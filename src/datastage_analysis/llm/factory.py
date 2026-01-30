"""
LLM Client Factory

Creates LLM clients based on configuration.
"""

import logging
from typing import Optional, Dict, Any

from .client import LLMClient, ProviderError
from .providers.anthropic import AnthropicClient

logger = logging.getLogger(__name__)

# Provider registry
PROVIDERS: Dict[str, type] = {
    'anthropic': AnthropicClient,
    # Future providers:
    # 'azure': AzureOpenAIClient,
    # 'aws': BedrockClient,
    # 'gcp': VertexClient,
    # 'openrouter': OpenRouterClient,
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
        provider: Provider name (anthropic, azure, aws, gcp, openrouter)
                  Default: from config or 'anthropic'
        model: Model ID (provider-specific)
               Default: from config or provider default
        use_cache: Whether to wrap client with caching layer
        **kwargs: Additional provider-specific configuration

    Returns:
        Configured LLMClient instance

    Example:
        # Use defaults from config
        client = get_llm_client()

        # Specify provider
        client = get_llm_client(provider="anthropic")

        # Specify provider and model
        client = get_llm_client(
            provider="anthropic",
            model="claude-3-5-haiku-20241022"
        )

        # With custom configuration
        client = get_llm_client(
            provider="anthropic",
            api_key="sk-...",
            timeout=60.0
        )
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
        available = ', '.join(PROVIDERS.keys())
        raise ProviderError(
            f"Unknown provider: {provider}. Available: {available}"
        )

    provider_class = PROVIDERS[provider]

    # Determine model
    if model is None and config:
        model = config.get('llm', 'models', provider)

    # Get provider-specific config
    provider_config = {}
    if config:
        provider_config = config.get('llm', 'providers', provider, default={}) or {}

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


def list_providers() -> Dict[str, bool]:
    """
    List available providers and their availability.

    Returns:
        Dict mapping provider name to availability status
    """
    result = {}
    for name, cls in PROVIDERS.items():
        try:
            # Check if provider can be instantiated (has required deps)
            # This is a basic check - actual availability depends on credentials
            result[name] = True
        except Exception:
            result[name] = False
    return result


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

    # Add other providers as implemented
    return {}
