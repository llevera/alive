# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for Azure OpenAI models that conform to OpenAI tool use API"""

import os
from typing import Any, Iterable, List, Literal, Optional, Union, cast

from openai import NOT_GIVEN, NotGiven, AzureOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from requests.exceptions import HTTPError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole


class AzureOpenAIAgent(BaseRole):
    """Agent role for Azure OpenAI models that conform to OpenAI tool use API"""

    role_type: RoleType = RoleType.AGENT
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

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

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
        # Get OpenAI tools if most recent message is from user
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        openai_tools = (
            convert_to_openai_tools(available_tools)
            if messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
            else NOT_GIVEN
        )
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven],
            openai_tools,
        )
        # Convert to OpenAI messages.
        current_context = get_current_context()
        openai_messages, _ = to_openai_messages(messages)
        # Call model
        response = self.model_inference(
            openai_messages=openai_messages, openai_tools=openai_tools
        )
        # Parse response
        openai_response_message = response.choices[0].message
        # Message contains no tool call, aka addressed to user
        if openai_response_message.tool_calls is None:
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(
        self,
        openai_messages: list[
            dict[
                Literal["role", "content", "tool_call_id", "name", "tool_calls"],
                Any,
            ]
        ],
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
                    tools=openai_tools 
                )
        except Exception as e:
            # Log error for monitoring and troubleshooting
            print(f"Azure OpenAI API error: {e}")
            raise


class AzureGPT4Agent(AzureOpenAIAgent):
    """Azure OpenAI GPT-4 agent with common deployment configuration"""
    
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


class AzureGPT4oAgent(AzureOpenAIAgent):
    """Azure OpenAI GPT-4o agent with common deployment configuration"""
    
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


class AzureGPT4oMiniAgent(AzureOpenAIAgent):
    """Azure OpenAI GPT-4o-mini agent with common deployment configuration"""
    
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


class AzureGPT35TurboAgent(AzureOpenAIAgent):
    """Azure OpenAI GPT-3.5 Turbo agent with common deployment configuration"""
    
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
