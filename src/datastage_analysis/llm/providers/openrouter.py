"""
OpenRouter Provider

Implements LLMClient for OpenRouter - a unified API for multiple LLM providers.
Supports Claude, GPT-4, Llama, Mistral, and many other models.
"""

import os
import time
import logging
import json
from typing import Optional, List, Dict, Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from ..client import (
    LLMClient,
    LLMResponse,
    ModelInfo,
    RateLimitError,
    AuthenticationError,
    ProviderError,
)

logger = logging.getLogger(__name__)

# Model configurations for OpenRouter
OPENROUTER_MODELS = {
    # Anthropic models
    'anthropic/claude-sonnet-4': {
        'display_name': 'Claude Sonnet 4',
        'max_tokens': 200000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
    },
    'anthropic/claude-3.5-sonnet': {
        'display_name': 'Claude 3.5 Sonnet',
        'max_tokens': 200000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
    },
    'anthropic/claude-3-haiku': {
        'display_name': 'Claude 3 Haiku',
        'max_tokens': 200000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00025,
        'cost_per_1k_output': 0.00125,
    },
    # OpenAI models
    'openai/gpt-4o': {
        'display_name': 'GPT-4o',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.005,
        'cost_per_1k_output': 0.015,
    },
    'openai/gpt-4o-mini': {
        'display_name': 'GPT-4o Mini',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00015,
        'cost_per_1k_output': 0.0006,
    },
    # Meta Llama models
    'meta-llama/llama-3.1-405b-instruct': {
        'display_name': 'Llama 3.1 405B',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.003,
    },
    'meta-llama/llama-3.1-70b-instruct': {
        'display_name': 'Llama 3.1 70B',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.0008,
        'cost_per_1k_output': 0.0008,
    },
    # Mistral models
    'mistralai/mistral-large': {
        'display_name': 'Mistral Large',
        'max_tokens': 128000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.009,
    },
    'mistralai/codestral-latest': {
        'display_name': 'Codestral',
        'max_tokens': 32000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.001,
        'cost_per_1k_output': 0.003,
    },
    # Google models
    'google/gemini-pro-1.5': {
        'display_name': 'Gemini 1.5 Pro',
        'max_tokens': 2097152,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.00125,
        'cost_per_1k_output': 0.005,
    },
    # DeepSeek models
    'deepseek/deepseek-coder': {
        'display_name': 'DeepSeek Coder',
        'max_tokens': 64000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00014,
        'cost_per_1k_output': 0.00028,
    },
}

DEFAULT_MODEL = 'anthropic/claude-sonnet-4'
BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(LLMClient):
    """
    OpenRouter client.

    OpenRouter provides a unified API to access models from multiple providers.
    Uses OpenAI-compatible API format.

    Environment variables:
        OPENROUTER_API_KEY: API key for authentication
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        timeout: float = 120.0,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenRouter client.

        Args:
            model: Model ID (e.g., 'anthropic/claude-sonnet-4')
            api_key: API key (default: from OPENROUTER_API_KEY)
            base_url: API base URL
            timeout: Request timeout in seconds
            site_url: Optional site URL for rankings
            app_name: Optional app name for rankings
        """
        super().__init__(model=model or DEFAULT_MODEL, **kwargs)

        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise AuthenticationError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
            )

        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.site_url = site_url
        self.app_name = app_name or "datastage-migration-analyzer"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenRouter."""
        start_time = time.time()

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build request body
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop_sequences:
            body["stop"] = stop_sequences

        # Add any extra kwargs
        body.update(kwargs)

        # Build headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url or "https://github.com/datastage-migration",
            "X-Title": self.app_name,
        }

        try:
            request = Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(body).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode('utf-8'))

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            choice = response_data.get('choices', [{}])[0]
            content = choice.get('message', {}).get('content', '')
            finish_reason = choice.get('finish_reason', 'stop')

            usage = response_data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)

            return LLMResponse(
                content=content,
                model=response_data.get('model', self.model),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached=False,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                metadata={
                    'id': response_data.get('id', ''),
                    'provider': self._get_provider_from_model(),
                }
            )

        except HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)

            if e.code == 429:
                raise RateLimitError(f"Rate limit exceeded: {error_body}")
            elif e.code in (401, 403):
                raise AuthenticationError(f"Authentication failed: {error_body}")
            else:
                raise ProviderError(f"OpenRouter API error ({e.code}): {error_body}")

        except URLError as e:
            raise ProviderError(f"Network error: {e}")

        except Exception as e:
            raise ProviderError(f"OpenRouter error: {e}")

    def _get_provider_from_model(self) -> str:
        """Extract provider name from model ID."""
        if '/' in self.model:
            return self.model.split('/')[0]
        return 'unknown'

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Approximate: ~4 chars per token
        return len(text) // 4 + 1

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        model_config = OPENROUTER_MODELS.get(self.model, {
            'display_name': self.model,
            'max_tokens': 32000,
            'max_output_tokens': 4096,
            'cost_per_1k_input': 0.01,
            'cost_per_1k_output': 0.03,
        })

        return ModelInfo(
            provider='openrouter',
            model_id=self.model,
            display_name=model_config['display_name'],
            max_tokens=model_config['max_tokens'],
            max_output_tokens=model_config['max_output_tokens'],
            supports_system_prompt=True,
            supports_streaming=True,
            cost_per_1k_input=model_config['cost_per_1k_input'],
            cost_per_1k_output=model_config['cost_per_1k_output'],
        )

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from OpenRouter API.

        Returns:
            List of model information dictionaries
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            request = Request(
                f"{self.base_url}/models",
                headers=headers,
                method='GET'
            )

            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data.get('data', [])

        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []
