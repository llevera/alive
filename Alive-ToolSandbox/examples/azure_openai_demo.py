#!/usr/bin/env python3
"""
Example script demonstrating Azure OpenAI agent usage.

This script shows how to set up and use Azure OpenAI agents with different
authentication methods and configurations.
"""

import os
from tool_sandbox.roles.azure_openai_agent import (
    AzureGPT4Agent,
    AzureGPT4oAgent,
    AzureGPT35TurboAgent,
)


def main():
    """Demonstrate Azure OpenAI agent setup and basic functionality."""
    
    print("Azure OpenAI Agent Demo")
    print("=" * 50)
    
    # Example 1: Using API Key Authentication (for development)
    print("\n1. API Key Authentication Example:")
    print("-" * 40)
    
    # Set up environment variables (in real usage, these would be set externally)
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://your-resource.openai.azure.com/'
    os.environ['AZURE_OPENAI_API_KEY'] = 'your-api-key-here'
    
    try:
        # Create a GPT-4 agent with API key authentication
        gpt4_agent = AzureGPT4Agent(
            deployment_name="your-gpt4-deployment-name",
            use_managed_identity=False  # Use API key instead
        )
        print(f"✓ GPT-4 Agent created: {gpt4_agent.model_name} -> {gpt4_agent.deployment_name}")
        
        # Create a GPT-4o agent
        gpt4o_agent = AzureGPT4oAgent(
            deployment_name="your-gpt4o-deployment-name",
            use_managed_identity=False
        )
        print(f"✓ GPT-4o Agent created: {gpt4o_agent.model_name} -> {gpt4o_agent.deployment_name}")
        
        # Create a GPT-3.5 Turbo agent
        gpt35_agent = AzureGPT35TurboAgent(
            deployment_name="your-gpt35-deployment-name",
            use_managed_identity=False
        )
        print(f"✓ GPT-3.5 Turbo Agent created: {gpt35_agent.model_name} -> {gpt35_agent.deployment_name}")
        
    except Exception as e:
        print(f"✗ Error creating agents with API key: {e}")
    
    # Example 2: Using Managed Identity (for production on Azure)
    print("\n2. Managed Identity Authentication Example:")
    print("-" * 40)
    
    try:
        # Note: This will work only when running on Azure with managed identity enabled
        # For this demo, we'll catch the expected error
        managed_identity_agent = AzureGPT4Agent(
            deployment_name="your-gpt4-deployment-name",
            use_managed_identity=True  # Default - use managed identity
        )
        print(f"✓ Managed Identity Agent created: {managed_identity_agent.model_name}")
        
    except Exception as e:
        print(f"✗ Expected error for managed identity (not on Azure): {type(e).__name__}")
        print("  This is normal when running locally - managed identity works only on Azure")
    
    # Example 3: Configuration Best Practices
    print("\n3. Configuration Best Practices:")
    print("-" * 40)
    
    print("Environment Variables to Set:")
    print("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
    print("  AZURE_OPENAI_API_KEY=your-api-key (only if not using managed identity)")
    
    print("\nRecommended Authentication Methods:")
    print("  • Local Development: API Key authentication")
    print("  • Production on Azure: Managed Identity")
    print("  • CI/CD Pipelines: Service Principal")
    
    print("\nSecurity Best Practices:")
    print("  • Never hardcode API keys in source code")
    print("  • Use Azure Key Vault for secret management")
    print("  • Enable managed identity when running on Azure")
    print("  • Assign least privilege roles (Cognitive Services OpenAI User)")
    print("  • Monitor access logs and usage")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
