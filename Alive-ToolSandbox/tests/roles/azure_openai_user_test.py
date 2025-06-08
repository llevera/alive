# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Tests for Azure OpenAI user role implementations."""

import os
import pytest
from unittest.mock import patch, MagicMock

from tool_sandbox.roles.openai_api_user import (
    OpenAIAPIUser,
    AzureGPT4User,
    AzureGPT4oUser,
    AzureGPT35TurboUser,
)


class TestOpenAIAPIUserAzureIntegration:
    """Test cases for OpenAI API user with Azure OpenAI integration."""

    def test_openai_api_user_with_azure_config(self):
        """Test OpenAI API user automatically uses Azure client when Azure config is present."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test-endpoint.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-api-key'
        }):
            with patch('tool_sandbox.roles.openai_api_user.AzureOpenAI') as mock_azure_openai:
                user = OpenAIAPIUser()
                
                # Verify Azure client was created
                mock_azure_openai.assert_called_once_with(
                    azure_endpoint='https://test-endpoint.openai.azure.com/',
                    api_key='test-api-key',
                    api_version='2024-02-01',
                )

    def test_openai_api_user_with_custom_api_version(self):
        """Test OpenAI API user with custom Azure API version."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test-endpoint.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-api-key',
            'AZURE_OPENAI_API_VERSION': '2024-06-01'
        }):
            with patch('tool_sandbox.roles.openai_api_user.AzureOpenAI') as mock_azure_openai:
                user = OpenAIAPIUser()
                
                # Verify custom API version is used
                mock_azure_openai.assert_called_once_with(
                    azure_endpoint='https://test-endpoint.openai.azure.com/',
                    api_key='test-api-key',
                    api_version='2024-06-01',
                )

    def test_openai_api_user_without_azure_config(self):
        """Test OpenAI API user uses standard OpenAI client when no Azure config."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('tool_sandbox.roles.openai_api_user.OpenAI') as mock_openai:
                user = OpenAIAPIUser()
                
                # Verify standard OpenAI client was created
                mock_openai.assert_called_once_with(base_url="https://api.openai.com/v1")

    def test_openai_api_user_missing_azure_api_key(self):
        """Test OpenAI API user falls back to standard client when Azure endpoint but no API key."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test-endpoint.openai.azure.com/'
        }, clear=True):
            with patch('tool_sandbox.roles.openai_api_user.OpenAI') as mock_openai:
                user = OpenAIAPIUser()
                
                # Should fall back to standard OpenAI client
                mock_openai.assert_called_once_with(base_url="https://api.openai.com/v1")


class TestAzureGPT4oUser:
    """Test cases for Azure GPT-4o user class."""

    def test_azure_gpt4o_user_model_name(self):
        """Test Azure GPT-4o user has correct model name."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test-endpoint.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-api-key'
        }):
            with patch('tool_sandbox.roles.openai_api_user.AzureOpenAI'):
                user = AzureGPT4oUser()
                
                # Verify model configuration
                assert user.model_name == "gpt-4o"


class TestAzureGPT4User:
    """Test cases for Azure GPT-4 user class."""

    def test_azure_gpt4_user_model_name(self):
        """Test Azure GPT-4 user has correct model name."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test-endpoint.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-api-key'
        }):
            with patch('tool_sandbox.roles.openai_api_user.AzureOpenAI'):
                user = AzureGPT4User()
                
                # Verify model configuration
                assert user.model_name == "gpt-4"


class TestAzureGPT35TurboUser:
    """Test cases for Azure GPT-3.5 Turbo user class."""

    def test_azure_gpt35_turbo_user_model_name(self):
        """Test Azure GPT-3.5 Turbo user has correct model name."""
        with patch.dict(os.environ, {
            'AZURE_OPENAI_ENDPOINT': 'https://test-endpoint.openai.azure.com/',
            'AZURE_OPENAI_API_KEY': 'test-api-key'
        }):
            with patch('tool_sandbox.roles.openai_api_user.AzureOpenAI'):
                user = AzureGPT35TurboUser()
                
                # Verify model configuration
                assert user.model_name == "gpt-35-turbo"
