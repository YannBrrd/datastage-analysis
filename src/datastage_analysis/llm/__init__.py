"""
LLM Abstraction Layer

Provides a unified interface for multiple LLM providers with caching support.

Usage:
    from datastage_analysis.llm import get_llm_client

    # Use default provider from config
    client = get_llm_client()

    # Or specify provider
    client = get_llm_client(provider="anthropic")
    client = get_llm_client(provider="azure")

    # Generate completion
    response = client.complete(
        prompt="Convert this DataStage transform to PySpark...",
        system="You are an expert in DataStage to AWS Glue migration.",
        temperature=0.2
    )
    print(response.content)
"""

from .client import LLMClient, LLMResponse, ModelInfo
from .factory import get_llm_client

__all__ = [
    'LLMClient',
    'LLMResponse',
    'ModelInfo',
    'get_llm_client',
]
