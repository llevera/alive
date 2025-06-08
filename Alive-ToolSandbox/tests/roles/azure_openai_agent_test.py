# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Test file for Azure OpenAI agent functionality"""

import unittest
from unittest.mock import Mock, patch

from tool_sandbox.roles.azure_openai_agent import (
    AzureOpenAIAgent,
    AzureGPT4Agent,
    AzureGPT4oAgent,
    AzureGPT35TurboAgent,
)


class TestAzureOpenAIAgent(unittest.TestCase):
    """Test cases for Azure OpenAI agent"""

    @patch.dict('os.environ', {'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/'})
    @patch('tool_sandbox.roles.azure_openai_agent.DefaultAzureCredential')
    def test_agent_initialization_with_managed_identity(self, mock_credential):
        """Test agent initialization with managed identity authentication"""
        # Mock the credential and token
        mock_token = Mock()
        mock_token.token = "test_token"
        mock_credential.return_value.get_token.return_value = mock_token
        
        with patch('tool_sandbox.roles.azure_openai_agent.AzureOpenAI') as mock_azure_openai:
            agent = AzureOpenAIAgent()
            
            # Verify the AzureOpenAI client was created with correct parameters
            mock_azure_openai.assert_called_once_with(
                azure_endpoint='https://test.openai.azure.com/',
                api_version='2024-02-15-preview',
                azure_ad_token='test_token',
            )

    @patch.dict('os.environ', {
        'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'test_api_key'
    })
    def test_agent_initialization_with_api_key(self):
        """Test agent initialization with API key authentication"""
        with patch('tool_sandbox.roles.azure_openai_agent.AzureOpenAI') as mock_azure_openai:
            agent = AzureOpenAIAgent(use_managed_identity=False)
            
            # Verify the AzureOpenAI client was created with correct parameters
            mock_azure_openai.assert_called_once_with(
                azure_endpoint='https://test.openai.azure.com/',
                api_key='test_api_key',
                api_version='2024-02-15-preview',
            )

    def test_agent_initialization_missing_endpoint(self):
        """Test that agent raises error when endpoint is missing"""
        with self.assertRaises(ValueError) as context:
            AzureOpenAIAgent()
        
        self.assertIn("Azure OpenAI endpoint must be provided", str(context.exception))

    @patch.dict('os.environ', {'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/'})
    def test_agent_initialization_missing_api_key(self):
        """Test that agent raises error when API key is missing and managed identity is disabled"""
        with self.assertRaises(ValueError) as context:
            AzureOpenAIAgent(use_managed_identity=False)
        
        self.assertIn("API key must be provided", str(context.exception))

    @patch.dict('os.environ', {'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/'})
    @patch('tool_sandbox.roles.azure_openai_agent.DefaultAzureCredential')
    def test_gpt4_agent_initialization(self, mock_credential):
        """Test GPT-4 specific agent initialization"""
        mock_token = Mock()
        mock_token.token = "test_token"
        mock_credential.return_value.get_token.return_value = mock_token
        
        with patch('tool_sandbox.roles.azure_openai_agent.AzureOpenAI'):
            agent = AzureGPT4Agent(deployment_name="my-gpt4-deployment")
            
            self.assertEqual(agent.model_name, "gpt-4")
            self.assertEqual(agent.deployment_name, "my-gpt4-deployment")

    @patch.dict('os.environ', {'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/'})
    @patch('tool_sandbox.roles.azure_openai_agent.DefaultAzureCredential')
    def test_gpt4o_agent_initialization(self, mock_credential):
        """Test GPT-4o specific agent initialization"""
        mock_token = Mock()
        mock_token.token = "test_token"
        mock_credential.return_value.get_token.return_value = mock_token
        
        with patch('tool_sandbox.roles.azure_openai_agent.AzureOpenAI'):
            agent = AzureGPT4oAgent(deployment_name="my-gpt4o-deployment")
            
            self.assertEqual(agent.model_name, "gpt-4o")
            self.assertEqual(agent.deployment_name, "my-gpt4o-deployment")

    @patch.dict('os.environ', {'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/'})
    @patch('tool_sandbox.roles.azure_openai_agent.DefaultAzureCredential')
    def test_gpt35_turbo_agent_initialization(self, mock_credential):
        """Test GPT-3.5 Turbo specific agent initialization"""
        mock_token = Mock()
        mock_token.token = "test_token"
        mock_credential.return_value.get_token.return_value = mock_token
        
        with patch('tool_sandbox.roles.azure_openai_agent.AzureOpenAI'):
            agent = AzureGPT35TurboAgent(deployment_name="my-gpt35-deployment")
            
            self.assertEqual(agent.model_name, "gpt-3.5-turbo")
            self.assertEqual(agent.deployment_name, "my-gpt35-deployment")


if __name__ == '__main__':
    unittest.main()
