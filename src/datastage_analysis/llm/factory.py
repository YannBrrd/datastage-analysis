"""
LLM Client Factory

Creates LLM clients based on configuration.
"""

import logging
from typing import Optional, Dict, Any

from .client import LLMClient, ProviderError
from .providers.anthropic import AnthropicClient
from .providers.azure_openai import AzureOpenAIClient
from .providers.azure_foundry import AzureFoundryClient
from .providers.aws_bedrock import BedrockClient
from .providers.gcp_vertex import VertexClient
from .providers.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)

# Provider registry
PROVIDERS: Dict[str, type] = {
    'anthropic': AnthropicClient,
    'azure': AzureOpenAIClient,
    'azure_openai': AzureOpenAIClient,  # Alias
    'azure_foundry': AzureFoundryClient,
    'foundry': AzureFoundryClient,  # Alias
    'aws': BedrockClient,
    'bedrock': BedrockClient,  # Alias
    'gcp': VertexClient,
    'vertex': VertexClient,  # Alias
    'openrouter': OpenRouterClient,
}


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    use_cache: bool = True,
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: Provider name. Options:
                  - anthropic: Direct Anthropic API
                  - azure / azure_openai: Azure OpenAI Service
                  - azure_foundry / foundry: Azure AI Foundry
                  - aws / bedrock: AWS Bedrock
                  - gcp / vertex: GCP Vertex AI
                  - openrouter: OpenRouter unified API
        model: Model ID (provider-specific)
        use_cache: Whether to wrap client with caching layer
        **kwargs: Additional provider-specific configuration

    Returns:
        Configured LLMClient instance

    Example:
        # Use defaults from config
        client = get_llm_client()

        # Anthropic
        client = get_llm_client(provider="anthropic")

        # Azure OpenAI
        client = get_llm_client(provider="azure", deployment_name="gpt-4o")

        # Azure AI Foundry
        client = get_llm_client(provider="azure_foundry", model="meta-llama-3.1-70b-instruct")

        # AWS Bedrock
        client = get_llm_client(provider="aws", region="us-west-2")

        # GCP Vertex AI
        client = get_llm_client(provider="gcp", project_id="my-project")

        # OpenRouter
        client = get_llm_client(provider="openrouter", model="anthropic/claude-sonnet-4")
    """
    # Load config
    try:
        from ..config import get_config
        config = get_config()
    except ImportError:
        config = None

    # Determine provider
    if provider is None:
        if config:
            provider = config.get('llm', 'provider', default='anthropic')
        else:
            provider = 'anthropic'

    provider = provider.lower()

    # Check if LLM is enabled
    if config and not config.get('llm', 'enabled', default=True):
        raise ProviderError("LLM is disabled in configuration. Set llm.enabled=true to enable.")

    # Get provider class
    if provider not in PROVIDERS:
        available = ', '.join(sorted(set(PROVIDERS.keys())))
        raise ProviderError(
            f"Unknown provider: {provider}. Available: {available}"
        )

    provider_class = PROVIDERS[provider]

    # Determine model from config
    if model is None and config:
        # Map aliases to config keys
        config_key = provider
        if provider in ('azure_openai',):
            config_key = 'azure'
        elif provider in ('foundry',):
            config_key = 'azure_foundry'
        elif provider in ('bedrock',):
            config_key = 'aws'
        elif provider in ('vertex',):
            config_key = 'gcp'

        model = config.get('llm', 'models', config_key)

    # Get provider-specific config
    provider_config = {}
    if config:
        # Map aliases to config keys
        config_key = provider
        if provider in ('azure_openai',):
            config_key = 'azure'
        elif provider in ('foundry',):
            config_key = 'azure_foundry'
        elif provider in ('bedrock',):
            config_key = 'aws'
        elif provider in ('vertex',):
            config_key = 'gcp'

        provider_config = config.get('llm', 'providers', config_key, default={}) or {}

    # Merge configs (kwargs override config file)
    merged_config = {**provider_config, **kwargs}

    # Add settings from config
    if config:
        settings = config.get('llm', 'settings', default={}) or {}
        if 'timeout' not in merged_config and 'timeout_seconds' in settings:
            merged_config['timeout'] = settings['timeout_seconds']

    # Create client
    logger.info(f"Creating LLM client: provider={provider}, model={model}")
    client = provider_class(model=model, **merged_config)

    # Wrap with cache if enabled
    if use_cache:
        cache_enabled = True
        if config:
            cache_enabled = config.get('cache', 'enabled', default=True)

        if cache_enabled:
            from .cache import CachedLLMClient
            client = CachedLLMClient(client)
            logger.info("LLM client wrapped with caching layer")

    return client


def list_providers() -> Dict[str, Dict[str, Any]]:
    """
    List available providers with their info.

    Returns:
        Dict mapping provider name to info dict
    """
    # Deduplicate aliases
    unique_providers = {
        'anthropic': 'Direct Anthropic Claude API',
        'azure': 'Azure OpenAI Service',
        'azure_foundry': 'Azure AI Foundry (Llama, Mistral, etc.)',
        'aws': 'AWS Bedrock (Claude, Llama, Titan)',
        'gcp': 'GCP Vertex AI (Gemini, Claude)',
        'openrouter': 'OpenRouter unified API',
    }

    return {
        name: {
            'description': desc,
            'class': PROVIDERS[name].__name__,
        }
        for name, desc in unique_providers.items()
    }


def get_provider_models(provider: str) -> Dict[str, Dict[str, Any]]:
    """
    Get available models for a provider.

    Args:
        provider: Provider name

    Returns:
        Dict of model_id -> model_info
    """
    provider = provider.lower()

    if provider == 'anthropic':
        from .providers.anthropic import ANTHROPIC_MODELS
        return ANTHROPIC_MODELS
    elif provider in ('azure', 'azure_openai'):
        from .providers.azure_openai import AZURE_MODELS
        return AZURE_MODELS
    elif provider in ('azure_foundry', 'foundry'):
        from .providers.azure_foundry import FOUNDRY_MODELS
        return FOUNDRY_MODELS
    elif provider in ('aws', 'bedrock'):
        from .providers.aws_bedrock import BEDROCK_MODELS
        return BEDROCK_MODELS
    elif provider in ('gcp', 'vertex'):
        from .providers.gcp_vertex import VERTEX_MODELS
        return VERTEX_MODELS
    elif provider == 'openrouter':
        from .providers.openrouter import OPENROUTER_MODELS
        return OPENROUTER_MODELS

    return {}
