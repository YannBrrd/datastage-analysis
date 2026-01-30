"""
Token Counter

Estimates token counts for different models and providers.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Average characters per token by model family
CHARS_PER_TOKEN = {
    'claude': 3.5,
    'gpt': 4.0,
    'llama': 3.8,
    'mistral': 3.8,
    'gemini': 4.0,
    'default': 4.0,
}


class TokenCounter:
    """
    Estimates token counts for text.

    Supports multiple tokenization strategies:
    1. tiktoken (for OpenAI models) - most accurate
    2. anthropic tokenizer (for Claude) - if available
    3. Character-based estimation - fallback
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize token counter.

        Args:
            model: Model identifier for accurate counting
        """
        self.model = model
        self._tiktoken_encoder = None
        self._model_family = self._detect_model_family(model)

    def _detect_model_family(self, model: Optional[str]) -> str:
        """Detect model family from model name."""
        if not model:
            return 'default'

        model_lower = model.lower()

        if 'claude' in model_lower or 'anthropic' in model_lower:
            return 'claude'
        elif 'gpt' in model_lower or 'openai' in model_lower:
            return 'gpt'
        elif 'llama' in model_lower or 'meta' in model_lower:
            return 'llama'
        elif 'mistral' in model_lower:
            return 'mistral'
        elif 'gemini' in model_lower or 'google' in model_lower:
            return 'gemini'

        return 'default'

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count (estimated or exact)
        """
        if not text:
            return 0

        # Try tiktoken for GPT models
        if self._model_family == 'gpt':
            try:
                return self._count_tiktoken(text)
            except Exception:
                pass

        # Fallback to character-based estimation
        return self._estimate_by_chars(text)

    def _count_tiktoken(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken

            if self._tiktoken_encoder is None:
                # Use cl100k_base for GPT-4 and newer
                self._tiktoken_encoder = tiktoken.get_encoding("cl100k_base")

            return len(self._tiktoken_encoder.encode(text))

        except ImportError:
            raise ValueError("tiktoken not installed")

    def _estimate_by_chars(self, text: str) -> int:
        """Estimate tokens by character count."""
        chars_per_token = CHARS_PER_TOKEN.get(
            self._model_family,
            CHARS_PER_TOKEN['default']
        )

        # Count characters excluding excessive whitespace
        normalized = ' '.join(text.split())
        char_count = len(normalized)

        return int(char_count / chars_per_token) + 1

    def count_messages(self, messages: list) -> int:
        """
        Count tokens for a list of messages.

        Accounts for message formatting overhead.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total token count
        """
        total = 0

        for msg in messages:
            # Add overhead for message structure
            total += 4  # <|start|>role<|end|>

            if isinstance(msg, dict):
                content = msg.get('content', '')
            else:
                content = str(msg)

            total += self.count(content)

        # Add final overhead
        total += 2

        return total


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Convenience function to estimate tokens.

    Args:
        text: Text to estimate
        model: Optional model for accurate estimation

    Returns:
        Estimated token count
    """
    counter = TokenCounter(model)
    return counter.count(text)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str
) -> float:
    """
    Estimate cost for token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier

    Returns:
        Estimated cost in USD
    """
    # Pricing per 1K tokens (approximate)
    PRICING = {
        'claude-sonnet-4': (0.003, 0.015),
        'claude-3-5-sonnet': (0.003, 0.015),
        'claude-3-haiku': (0.00025, 0.00125),
        'gpt-4o': (0.005, 0.015),
        'gpt-4o-mini': (0.00015, 0.0006),
        'gpt-4': (0.03, 0.06),
        'gemini-1.5-pro': (0.00125, 0.005),
        'gemini-1.5-flash': (0.000075, 0.0003),
        'llama-3.1-70b': (0.0008, 0.0008),
        'mistral-large': (0.003, 0.009),
        'default': (0.01, 0.03),
    }

    # Find matching pricing
    model_lower = model.lower()
    input_price, output_price = PRICING['default']

    for key, prices in PRICING.items():
        if key in model_lower:
            input_price, output_price = prices
            break

    input_cost = (input_tokens / 1000) * input_price
    output_cost = (output_tokens / 1000) * output_price

    return input_cost + output_cost
