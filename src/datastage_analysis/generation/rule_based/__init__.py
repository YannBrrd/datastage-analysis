"""
Rule-Based Code Generator

Generates AWS Glue code using templates and pattern matching.
No LLM required - fully deterministic.
"""

from .generator import RuleBasedGenerator

__all__ = ['RuleBasedGenerator']
