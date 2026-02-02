"""
AWS Glue Target Generator

Generates AWS Glue PySpark code from DataStage job definitions.
"""

from .generator import GlueTargetGenerator
from .config import GlueTargetConfig

__all__ = [
    'GlueTargetGenerator',
    'GlueTargetConfig',
]
