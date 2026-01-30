"""
Azure AI Foundry Provider

Implements LLMClient for Azure AI Foundry (formerly Azure AI Studio).
Supports multiple model types: OpenAI, Llama, Mistral, Cohere, etc.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any

from ..client import (
    LLMClient,
    LLMResponse,
    ModelInfo,
    RateLimitError,
    AuthenticationError,
    ProviderError,
)

logger = logging.getLogger(__name__)

# Model configurations for Azure AI Foundry
FOUNDRY_MODELS = {
    # OpenAI models
    'gpt-4o': {
        'display_name': 'GPT-4o',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.005,
        'cost_per_1k_output': 0.015,
    },
    # Meta Llama models
    'meta-llama-3.1-405b-instruct': {
        'display_name': 'Llama 3.1 405B Instruct',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00533,
        'cost_per_1k_output': 0.016,
    },
    'meta-llama-3.1-70b-instruct': {
        'display_name': 'Llama 3.1 70B Instruct',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00268,
        'cost_per_1k_output': 0.00354,
    },
    # Mistral models
    'mistral-large': {
        'display_name': 'Mistral Large',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.004,
        'cost_per_1k_output': 0.012,
    },
    'mistral-small': {
        'display_name': 'Mistral Small',
        'max_tokens': 32000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.001,
        'cost_per_1k_output': 0.003,
    },
    # Cohere models
    'cohere-command-r-plus': {
        'display_name': 'Cohere Command R+',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
    },
}

DEFAULT_MODEL = 'gpt-4o'


class AzureFoundryClient(LLMClient):
    """
    Azure AI Foundry client.

    Supports multiple model providers through Azure AI Foundry's unified API.

    Requires: uv install azure-ai-inference

    Environment variables:
        AZURE_FOUNDRY_ENDPOINT: Model endpoint URL
        AZURE_FOUNDRY_API_KEY: API key for authentication
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        """
        Initialize Azure AI Foundry client.

        Args:
            model: Model identifier
            api_key: API key (default: from AZURE_FOUNDRY_API_KEY)
            endpoint: Endpoint URL (default: from AZURE_FOUNDRY_ENDPOINT)
            timeout: Request timeout in seconds
        """
        super().__init__(model=model or DEFAULT_MODEL, **kwargs)

        self.api_key = api_key or os.environ.get('AZURE_FOUNDRY_API_KEY')
        self.endpoint = endpoint or os.environ.get('AZURE_FOUNDRY_ENDPOINT')

        if not self.api_key:
            raise AuthenticationError(
                "Azure AI Foundry API key not found. "
                "Set AZURE_FOUNDRY_API_KEY environment variable."
            )
        if not self.endpoint:
            raise AuthenticationError(
                "Azure AI Foundry endpoint not found. "
                "Set AZURE_FOUNDRY_ENDPOINT environment variable."
            )

        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of Azure AI Foundry client."""
        if self._client is None:
            try:
                from azure.ai.inference import ChatCompletionsClient
                from azure.core.credentials import AzureKeyCredential
            except ImportError:
                raise ProviderError(
                    "azure-ai-inference package not installed. "
                    "Run: uv install azure-ai-inference"
                )

            self._client = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )

        return self._client

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Azure AI Foundry."""
        client = self._get_client()
        start_time = time.time()

        try:
            from azure.ai.inference.models import (
                SystemMessage,
                UserMessage,
            )
        except ImportError:
            raise ProviderError("azure-ai-inference package not installed")

        # Build messages
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(UserMessage(content=prompt))

        try:
            response = client.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                model=self.model,
                **kwargs
            )

            latency_ms = (time.time() - start_time) * 1000

            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            # Extract usage if available
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                output_tokens = getattr(response.usage, 'completion_tokens', 0)

            return LLMResponse(
                content=content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached=False,
                latency_ms=latency_ms,
                finish_reason=str(finish_reason),
                metadata={
                    'id': getattr(response, 'id', ''),
                    'endpoint': self.endpoint,
                }
            )

        except Exception as e:
            error_str = str(e).lower()

            if 'rate' in error_str or '429' in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif 'auth' in error_str or '401' in error_str or 'key' in error_str:
                raise AuthenticationError(f"Authentication failed: {e}")
            else:
                raise ProviderError(f"Azure AI Foundry API error: {e}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Approximate: ~4 chars per token
        return len(text) // 4 + 1

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        model_config = FOUNDRY_MODELS.get(self.model, {
            'display_name': self.model,
            'max_tokens': 32000,
            'max_output_tokens': 4096,
            'cost_per_1k_input': 0.01,
            'cost_per_1k_output': 0.03,
        })

        return ModelInfo(
            provider='azure_foundry',
            model_id=self.model,
            display_name=model_config['display_name'],
            max_tokens=model_config['max_tokens'],
            max_output_tokens=model_config['max_output_tokens'],
            supports_system_prompt=True,
            supports_streaming=True,
            cost_per_1k_input=model_config['cost_per_1k_input'],
            cost_per_1k_output=model_config['cost_per_1k_output'],
        )
