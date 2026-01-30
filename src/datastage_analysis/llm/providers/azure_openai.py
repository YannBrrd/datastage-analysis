"""
Azure OpenAI Provider

Implements LLMClient for Azure OpenAI Service.
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

# Model configurations for Azure OpenAI
AZURE_MODELS = {
    'gpt-4o': {
        'display_name': 'GPT-4o',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.005,
        'cost_per_1k_output': 0.015,
    },
    'gpt-4o-mini': {
        'display_name': 'GPT-4o Mini',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00015,
        'cost_per_1k_output': 0.0006,
    },
    'gpt-4-turbo': {
        'display_name': 'GPT-4 Turbo',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.01,
        'cost_per_1k_output': 0.03,
    },
    'gpt-4': {
        'display_name': 'GPT-4',
        'max_tokens': 8192,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.03,
        'cost_per_1k_output': 0.06,
    },
    'gpt-35-turbo': {
        'display_name': 'GPT-3.5 Turbo',
        'max_tokens': 16384,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.0005,
        'cost_per_1k_output': 0.0015,
    },
}

DEFAULT_MODEL = 'gpt-4o'


class AzureOpenAIClient(LLMClient):
    """
    Azure OpenAI Service client.

    Requires the 'openai' package: pip install openai

    Environment variables:
        AZURE_OPENAI_API_KEY: API key for authentication
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        deployment_name: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        """
        Initialize Azure OpenAI client.

        Args:
            model: Model name (used for config lookup)
            api_key: API key (default: from AZURE_OPENAI_API_KEY)
            endpoint: Azure endpoint (default: from AZURE_OPENAI_ENDPOINT)
            api_version: API version string
            deployment_name: Deployment name (default: same as model)
            timeout: Request timeout in seconds
        """
        super().__init__(model=model or DEFAULT_MODEL, **kwargs)

        self.api_key = api_key or os.environ.get('AZURE_OPENAI_API_KEY')
        self.endpoint = endpoint or os.environ.get('AZURE_OPENAI_ENDPOINT')

        if not self.api_key:
            raise AuthenticationError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY environment variable."
            )
        if not self.endpoint:
            raise AuthenticationError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT environment variable."
            )

        self.api_version = api_version
        self.deployment_name = deployment_name or self.model
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ProviderError(
                    "openai package not installed. Run: pip install openai"
                )

            self._client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                timeout=self.timeout,
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
        """Generate completion using Azure OpenAI."""
        client = self._get_client()
        start_time = time.time()

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )

            latency_ms = (time.time() - start_time) * 1000

            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            return LLMResponse(
                content=content,
                model=response.model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cached=False,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                metadata={
                    'id': response.id,
                    'deployment': self.deployment_name,
                }
            )

        except Exception as e:
            error_str = str(e).lower()

            if 'rate' in error_str or '429' in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif 'auth' in error_str or '401' in error_str or 'key' in error_str:
                raise AuthenticationError(f"Authentication failed: {e}")
            else:
                raise ProviderError(f"Azure OpenAI API error: {e}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: ~4 chars per token
            return len(text) // 4 + 1

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        model_config = AZURE_MODELS.get(self.model, AZURE_MODELS[DEFAULT_MODEL])

        return ModelInfo(
            provider='azure',
            model_id=self.model,
            display_name=model_config['display_name'],
            max_tokens=model_config['max_tokens'],
            max_output_tokens=model_config['max_output_tokens'],
            supports_system_prompt=True,
            supports_streaming=True,
            cost_per_1k_input=model_config['cost_per_1k_input'],
            cost_per_1k_output=model_config['cost_per_1k_output'],
        )
