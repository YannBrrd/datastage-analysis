"""
LLM Provider Implementations

Each provider implements the LLMClient interface for a specific service.

Supported Providers:
- Anthropic: Direct Claude API
- Azure OpenAI: Azure-hosted OpenAI models
- Azure AI Foundry: Azure AI Studio models (Llama, Mistral, etc.)
- AWS Bedrock: Claude, Llama, Mistral, Titan on AWS
- GCP Vertex: Gemini, Claude on Google Cloud
- OpenRouter: Unified API for multiple providers
"""

from .anthropic import AnthropicClient
from .azure_openai import AzureOpenAIClient
from .azure_foundry import AzureFoundryClient
from .aws_bedrock import BedrockClient
from .gcp_vertex import VertexClient
from .openrouter import OpenRouterClient

__all__ = [
    'AnthropicClient',
    'AzureOpenAIClient',
    'AzureFoundryClient',
    'BedrockClient',
    'VertexClient',
    'OpenRouterClient',
]
