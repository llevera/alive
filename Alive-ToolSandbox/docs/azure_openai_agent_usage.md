# Azure OpenAI Agent Usage Examples

This document provides examples of how to use the Azure OpenAI agent classes with different authentication methods and configurations.

## Basic Usage

### Using Managed Identity (Recommended for Azure-hosted applications)

```python
from tool_sandbox.roles.azure_openai_agent import AzureGPT4oAgent

# For Azure-hosted applications (App Service, Container Apps, Functions, etc.)
# Set environment variables:
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

agent = AzureGPT4oAgent(
    deployment_name="your-gpt4o-deployment",
    use_managed_identity=True  # Default
)
```

### Using API Key Authentication

```python
from tool_sandbox.roles.azure_openai_agent import AzureGPT4Agent

# Set environment variables:
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_KEY=your-api-key

agent = AzureGPT4Agent(
    deployment_name="your-gpt4-deployment",
    use_managed_identity=False
)

# Or pass values directly (not recommended for production)
agent = AzureGPT4Agent(
    deployment_name="your-gpt4-deployment",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-api-key",
    use_managed_identity=False
)
```

## Available Agent Classes

### AzureOpenAIAgent (Base Class)
Generic Azure OpenAI agent that can be configured for any model deployment.

```python
from tool_sandbox.roles.azure_openai_agent import AzureOpenAIAgent

agent = AzureOpenAIAgent(
    azure_endpoint="https://your-resource.openai.azure.com/",
    use_managed_identity=True,
    api_version="2024-02-15-preview"
)
agent.model_name = "gpt-4"
agent.deployment_name = "your-deployment-name"
```

### AzureGPT4Agent
Pre-configured for GPT-4 models.

```python
from tool_sandbox.roles.azure_openai_agent import AzureGPT4Agent

agent = AzureGPT4Agent(deployment_name="your-gpt4-deployment")
```

### AzureGPT4oAgent
Pre-configured for GPT-4o models.

```python
from tool_sandbox.roles.azure_openai_agent import AzureGPT4oAgent

agent = AzureGPT4oAgent(deployment_name="your-gpt4o-deployment")
```

### AzureGPT35TurboAgent
Pre-configured for GPT-3.5 Turbo models.

```python
from tool_sandbox.roles.azure_openai_agent import AzureGPT35TurboAgent

agent = AzureGPT35TurboAgent(deployment_name="your-gpt35-deployment")
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint | Yes |
| `AZURE_OPENAI_API_KEY` | API key for authentication | Only when `use_managed_identity=False` |

## Authentication Methods

### 1. Managed Identity (Recommended)
- **Use case**: Azure-hosted applications (App Service, Container Apps, Functions, AKS, etc.)
- **Benefits**: No secrets to manage, automatic credential rotation, least privilege access
- **Setup**: Assign the Azure OpenAI resource the "Cognitive Services OpenAI User" role to your managed identity

```bash
# Assign role to system-assigned managed identity
az role assignment create \
  --role "Cognitive Services OpenAI User" \
  --assignee-object-id <your-managed-identity-object-id> \
  --scope /subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.CognitiveServices/accounts/<openai-resource-name>
```

### 2. API Key Authentication
- **Use case**: Local development, non-Azure environments
- **Setup**: Get API key from Azure portal > your OpenAI resource > Keys and Endpoint

## Error Handling and Retry Logic

The agent includes built-in retry logic with exponential backoff for handling transient failures:

- **Retries**: Up to 3 attempts
- **Backoff**: Exponential with jitter (1-40 seconds)
- **Retry conditions**: HTTP errors from the Azure OpenAI service

## Security Best Practices

1. **Use Managed Identity** when running on Azure
2. **Store API keys in Azure Key Vault** if you must use key-based authentication
3. **Use least privilege access** - assign only necessary roles
4. **Monitor access logs** through Azure Monitor
5. **Rotate keys regularly** if using API key authentication

## Deployment Considerations

### Azure App Service
```python
# Enable managed identity in App Service
# Set AZURE_OPENAI_ENDPOINT in application settings
agent = AzureGPT4oAgent(deployment_name="your-deployment")
```

### Azure Container Apps
```python
# Enable managed identity in Container Apps
# Set AZURE_OPENAI_ENDPOINT as environment variable
agent = AzureGPT4oAgent(deployment_name="your-deployment")
```

### Azure Functions
```python
# Enable managed identity in Function App
# Set AZURE_OPENAI_ENDPOINT in application settings
agent = AzureGPT4oAgent(deployment_name="your-deployment")
```

### Local Development
```python
# Use API key for local development
# Set environment variables in .env file (use python-dotenv)
agent = AzureGPT4oAgent(
    deployment_name="your-deployment",
    use_managed_identity=False
)
```

## Performance Optimization

1. **Connection Pooling**: The Azure OpenAI client automatically handles connection pooling
2. **Concurrent Requests**: Use appropriate concurrency limits to avoid rate limiting
3. **Caching**: Consider caching responses for repeated queries
4. **Monitoring**: Use Azure Monitor to track performance metrics

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify managed identity is enabled and has correct role assignments
   - Check API key is valid if using key-based authentication

2. **Endpoint Errors**
   - Ensure the endpoint URL is correct and includes the protocol (https://)
   - Verify the resource exists and is in the correct region

3. **Deployment Errors**
   - Confirm the deployment name matches exactly (case-sensitive)
   - Verify the model is deployed and running

4. **Rate Limiting**
   - Implement backoff strategies for high-volume applications
   - Consider using multiple deployments for load distribution

### Logging and Monitoring

```python
import logging

# Enable debug logging for troubleshooting
logging.basicConfig(level=logging.DEBUG)

# The agent will log errors during model inference
agent = AzureGPT4oAgent(deployment_name="your-deployment")
```
