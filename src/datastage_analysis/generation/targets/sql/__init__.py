"""
SQL Target Generator

Generates SQL scripts from DataStage job definitions.
Supports multiple SQL dialects (Teradata, PostgreSQL, etc.).
"""

from .generator import SQLTargetGenerator
from .config import SQLTargetConfig, SQLDialect
from .dialects import TeradataDialect, GenericDialect

__all__ = [
    'SQLTargetGenerator',
    'SQLTargetConfig',
    'SQLDialect',
    'TeradataDialect',
    'GenericDialect',
]
