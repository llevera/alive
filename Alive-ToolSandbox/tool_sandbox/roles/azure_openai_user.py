# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Simulated user role for Azure OpenAI models that conform to OpenAI tool use API"""

import os
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast

from openai import NOT_GIVEN, NotGiven, AzureOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tool
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole


class AzureOpenAIUser(BaseRole):
    """Simulated user role for Azure OpenAI models that conform to OpenAI tool use API"""

    role_type: RoleType = RoleType.USER
    model_name: str
    deployment_name: str

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
    ) -> None:
        """Initialize Azure OpenAI client with API key authentication.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL. If not provided, will use AZURE_OPENAI_ENDPOINT env var.
            api_key: API key for authentication. If not provided, will use AZURE_OPENAI_API_KEY env var.
            api_version: Azure OpenAI API version to use.
        """
        super().__init__()
        
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint must be provided either via parameter or AZURE_OPENAI_ENDPOINT environment variable"
            )
        
        # Initialize Azure OpenAI client with API key authentication
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be provided either via parameter or AZURE_OPENAI_API_KEY environment variable"
            )
        
        self.openai_client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=api_key,
            api_version=self.api_version,
        )

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system & agent messages, sends out valid followup responses back to agent

        Comparing to agents and execution environments. Users and user simulators have a unique challenge. Agents and
        execution environments passively accept messages from other roles, execute them and respond. However, a user
        has the autonomy to decide when to stop the conversation. It must be able to, otherwise the conversation is
        never going to stop.

        The current idea is to instruct the user simulator to issue structured responses indicating end of conversation,
        1 such approach could be, we offer a tool to user simulator. The simulator could issue
        tool call to in order to terminate the conversation. This will be interpreted, and sent to execution env to
        execute.

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        response_messages: List[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return
        # Get OpenAI tools if most recent turn is from Agent (again, to terminate the conversation if needed)
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            [convert_to_openai_tool(tool) for tool in available_tools.values()]
            if messages[-1].sender == RoleType.AGENT
            else NOT_GIVEN
        )
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven],
            openai_tools,
        )
        # Convert to OpenAI messages
        openai_messages = self.to_openai_messages(messages=messages)
        # Call model
        response = self.model_inference(
            openai_messages=openai_messages, openai_tools=openai_tools
        )
        # Parse response
        openai_response_message = response.choices[0].message

        # Message contains no tool call, aka addressed to agent
        if openai_response_message.tool_calls is None:
            # Not sure why the content field `ChatCompletionMessage` has a type of
            # `str | None`.
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.AGENT,
                    content=openai_response_message.content,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=None,
                        ),
                    )
                )
        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3)
    )
    def model_inference(
        self,
        openai_messages: list[dict[Literal["role", "content"], str]],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        """Run Azure OpenAI model inference with retry logic and proper error handling

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
            
        Raises:
            HTTPError: When Azure OpenAI API returns an error after retries
        """
        try:
            with all_logging_disabled():
                return self.openai_client.chat.completions.create(
                    model=self.deployment_name,  # Use deployment name for Azure OpenAI
                    messages=cast(list[ChatCompletionMessageParam], openai_messages),
                    tools=openai_tools,
                )
        except Exception as e:
            # Log error for monitoring and troubleshooting
            print(f"Azure OpenAI API error: {e}")
            raise

    @staticmethod
    def to_openai_messages(
        messages: List[Message],
    ) -> List[Dict[Literal["role", "content"], str]]:
        """Converts a list of Tool Sandbox messages to OpenAI API messages, from the perspective of a simulated user

        Args:
            messages:   A list of Tool Sandbox messages

        Returns:
            A list of OpenAI API messages
        """
        openai_messages: List[Dict[Literal["role", "content"], str]] = []
        for message in messages:
            if message.sender == RoleType.SYSTEM and message.recipient == RoleType.USER:
                openai_messages.append({"role": "system", "content": message.content})
            elif (
                message.sender == RoleType.AGENT and message.recipient == RoleType.USER
            ):
                # The roles are in reverse
                # We are the user simulator, simulated response from OpenAI assistant role is the simulated user message
                # which means agent dialog is OpenAI user role
                openai_messages.append({"role": "user", "content": message.content})
            elif (
                message.sender == RoleType.USER and message.recipient == RoleType.AGENT
            ):
                openai_messages.append(
                    {"role": "assistant", "content": message.content}
                )
            elif (
                message.sender == RoleType.USER
                and message.recipient == RoleType.EXECUTION_ENVIRONMENT
            ) or (
                message.sender == RoleType.EXECUTION_ENVIRONMENT
                and message.recipient == RoleType.USER
            ):
                # These pairs are ignored.
                pass
            else:
                raise ValueError(
                    f"Unrecognized sender recipient pair {(message.sender, message.recipient)}"
                )
        return openai_messages


class AzureGPT4User(AzureOpenAIUser):
    """Azure OpenAI GPT-4 user with common deployment configuration"""
    
    def __init__(
        self,
        deployment_name: str = "gpt-4",
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
    ):
        super().__init__(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.model_name = "gpt-4"
        self.deployment_name = deployment_name


class AzureGPT4oUser(AzureOpenAIUser):
    """Azure OpenAI GPT-4o user with common deployment configuration"""
    
    def __init__(
        self,
        deployment_name: str = "gpt-4o",
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
    ):
        super().__init__(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.model_name = "gpt-4o"
        self.deployment_name = deployment_name


class AzureGPT4oMiniUser(AzureOpenAIUser):
    """Azure OpenAI GPT-4o-mini user with common deployment configuration"""
    
    def __init__(
        self,
        deployment_name: str = "gpt-4o-mini",
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
    ):
        super().__init__(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.model_name = "gpt-4o-mini"
        self.deployment_name = deployment_name


class AzureGPT35TurboUser(AzureOpenAIUser):
    """Azure OpenAI GPT-3.5 Turbo user with common deployment configuration"""
    
    def __init__(
        self,
        deployment_name: str = "gpt-35-turbo",
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
    ):
        super().__init__(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.model_name = "gpt-3.5-turbo"
        self.deployment_name = deployment_name
