"""
Migration Code Generation Module

Generates AWS Glue code from DataStage job definitions.

Usage:
    from datastage_analysis.generation import MigrationGenerator

    generator = MigrationGenerator()
    results = generator.generate(predictions, structures)

Dry-run mode:
    from datastage_analysis.generation import DryRunEstimator

    estimator = DryRunEstimator(provider='anthropic')
    estimate = estimator.estimate(predictions, cluster_info)
"""

from .generator import MigrationGenerator, GenerationResult, GeneratedJob
from .dry_run import DryRunEstimator, DryRunResult, format_dry_run_report

__all__ = [
    'MigrationGenerator',
    'GenerationResult',
    'GeneratedJob',
    'DryRunEstimator',
    'DryRunResult',
    'format_dry_run_report',
]
