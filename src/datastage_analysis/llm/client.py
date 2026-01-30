"""
LLM Client Base Classes

Defines the abstract interface for all LLM providers.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool = False
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'cached': self.cached,
            'latency_ms': self.latency_ms,
            'finish_reason': self.finish_reason,
        }


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    provider: str
    model_id: str
    display_name: str
    max_tokens: int
    max_output_tokens: int
    supports_system_prompt: bool
    supports_streaming: bool
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost


class LLMClient(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement this interface to ensure consistent
    behavior across different LLM services.
    """

    def __init__(self, model: Optional[str] = None, **kwargs):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier (provider-specific)
            **kwargs: Additional provider-specific configuration
        """
        self.model = model
        self._config = kwargs

    @abstractmethod
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
        Generate a completion from the model.

        Args:
            prompt: The user prompt/message
            system: Optional system prompt for context
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with the generated content and metadata
        """
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the current model.

        Returns:
            ModelInfo with model capabilities and pricing
        """
        pass

    def complete_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 2.0,
        **kwargs
    ) -> LLMResponse:
        """
        Complete with automatic retry on transient failures.

        Uses exponential backoff for rate limits and network errors.

        Args:
            prompt: The prompt to complete
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds (doubles each retry)
            **kwargs: Additional arguments passed to complete()

        Returns:
            LLMResponse from successful completion

        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        delay = base_delay

        for attempt in range(max_retries + 1):
            try:
                return self.complete(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check if error is retryable
                retryable = any(term in error_str for term in [
                    'rate limit', 'timeout', 'overloaded',
                    'connection', 'temporary', '529', '503', '502'
                ])

                if not retryable or attempt == max_retries:
                    logger.error(f"LLM request failed after {attempt + 1} attempts: {e}")
                    raise

                logger.warning(f"LLM request failed (attempt {attempt + 1}/{max_retries + 1}), "
                             f"retrying in {delay}s: {e}")
                time.sleep(delay)
                delay *= 2  # Exponential backoff

        raise last_exception

    def validate_response(self, response: LLMResponse) -> bool:
        """
        Validate that the response is usable.

        Override in subclasses for provider-specific validation.

        Args:
            response: The response to validate

        Returns:
            True if response is valid
        """
        if not response.content:
            return False
        if response.finish_reason not in ('stop', 'end_turn', 'length'):
            logger.warning(f"Unexpected finish reason: {response.finish_reason}")
        return True


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class TokenLimitError(LLMError):
    """Raised when token limit is exceeded."""
    pass


class AuthenticationError(LLMError):
    """Raised when authentication fails."""
    pass


class ProviderError(LLMError):
    """Raised for provider-specific errors."""
    pass
