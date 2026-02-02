"""
Anthropic Claude Provider

Implements LLMClient for Anthropic's Claude API.
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

# Model configurations
ANTHROPIC_MODELS = {
    'claude-sonnet-4-20250514': {
        'display_name': 'Claude Sonnet 4',
        'max_tokens': 200000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
    },
    'claude-opus-4-20250514': {
        'display_name': 'Claude Opus 4',
        'max_tokens': 200000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.015,
        'cost_per_1k_output': 0.075,
    },
    'claude-3-5-sonnet-20241022': {
        'display_name': 'Claude 3.5 Sonnet',
        'max_tokens': 200000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
    },
    'claude-3-5-haiku-20241022': {
        'display_name': 'Claude 3.5 Haiku',
        'max_tokens': 200000,
        'max_output_tokens': 8192,
        'cost_per_1k_input': 0.0008,
        'cost_per_1k_output': 0.004,
    },
    'claude-3-opus-20240229': {
        'display_name': 'Claude 3 Opus',
        'max_tokens': 200000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.015,
        'cost_per_1k_output': 0.075,
    },
    'claude-3-sonnet-20240229': {
        'display_name': 'Claude 3 Sonnet',
        'max_tokens': 200000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.003,
        'cost_per_1k_output': 0.015,
    },
    'claude-3-haiku-20240307': {
        'display_name': 'Claude 3 Haiku',
        'max_tokens': 200000,
        'max_output_tokens': 4096,
        'cost_per_1k_input': 0.00025,
        'cost_per_1k_output': 0.00125,
    },
}

DEFAULT_MODEL = 'claude-sonnet-4-20250514'


class AnthropicClient(LLMClient):
    """
    Anthropic Claude API client.

    Requires the 'anthropic' package: uv pip install anthropic

    Environment variables:
        ANTHROPIC_API_KEY: API key for authentication
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs
    ):
        """
        Initialize Anthropic client.

        Args:
            model: Model ID (default: claude-sonnet-4-20250514)
            api_key: API key (default: from ANTHROPIC_API_KEY env var)
            base_url: Optional API base URL override
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(model=model or DEFAULT_MODEL, **kwargs)

        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise AuthenticationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = base_url
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ProviderError(
                    "anthropic package not installed. Run: uv pip install anthropic"
                )

            kwargs = {
                'api_key': self.api_key,
                'timeout': self.timeout,
            }
            if self.base_url:
                kwargs['base_url'] = self.base_url

            self._client = anthropic.Anthropic(**kwargs)

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
        """
        Generate completion using Claude.

        Args:
            prompt: User message/prompt
            system: System prompt for context
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional parameters (passed to API)

        Returns:
            LLMResponse with generated content
        """
        client = self._get_client()
        start_time = time.time()

        # Build message
        messages = [{"role": "user", "content": prompt}]

        # Build request
        request_kwargs = {
            'model': self.model,
            'max_tokens': max_tokens,
            'messages': messages,
            'temperature': temperature,
        }

        if system:
            request_kwargs['system'] = system

        if stop_sequences:
            request_kwargs['stop_sequences'] = stop_sequences

        # Add any extra kwargs
        request_kwargs.update(kwargs)

        try:
            response = client.messages.create(**request_kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Extract content
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        content += block.text

            return LLMResponse(
                content=content,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cached=False,
                latency_ms=latency_ms,
                finish_reason=response.stop_reason or "stop",
                metadata={
                    'id': response.id,
                    'type': response.type,
                }
            )

        except Exception as e:
            error_str = str(e).lower()

            if 'rate' in error_str or '429' in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif 'auth' in error_str or '401' in error_str or 'api key' in error_str:
                raise AuthenticationError(f"Authentication failed: {e}")
            else:
                raise ProviderError(f"Anthropic API error: {e}")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token for English text.
        For more accuracy, use the anthropic tokenizer if available.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Try to use the official tokenizer
        try:
            client = self._get_client()
            if hasattr(client, 'count_tokens'):
                return client.count_tokens(text)
        except Exception:
            pass

        # Fallback: approximate 4 chars per token
        return len(text) // 4 + 1

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        model_config = ANTHROPIC_MODELS.get(self.model, ANTHROPIC_MODELS[DEFAULT_MODEL])

        return ModelInfo(
            provider='anthropic',
            model_id=self.model,
            display_name=model_config['display_name'],
            max_tokens=model_config['max_tokens'],
            max_output_tokens=model_config['max_output_tokens'],
            supports_system_prompt=True,
            supports_streaming=True,
            cost_per_1k_input=model_config['cost_per_1k_input'],
            cost_per_1k_output=model_config['cost_per_1k_output'],
        )

    def complete_code(
        self,
        prompt: str,
        language: str = "python",
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Convenience method for code generation.

        Adds appropriate system prompt for code generation if not provided.

        Args:
            prompt: Code generation prompt
            language: Target programming language
            system: Optional custom system prompt
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated code
        """
        if system is None:
            system = f"""You are an expert {language} programmer.
Generate clean, well-documented, production-ready code.
Follow best practices and include error handling.
Output only the code, no explanations unless specifically asked."""

        return self.complete(
            prompt=prompt,
            system=system,
            temperature=0.1,  # Low temperature for code
            **kwargs
        )
