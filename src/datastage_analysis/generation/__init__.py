"""
Migration Code Generation Module

Generates AWS Glue code from DataStage job definitions.

Usage:
    from datastage_analysis.generation import MigrationGenerator

    generator = MigrationGenerator()
    results = generator.generate(predictions, structures)
"""

from .generator import MigrationGenerator, GenerationResult, GeneratedJob

__all__ = [
    'MigrationGenerator',
    'GenerationResult',
    'GeneratedJob',
]
