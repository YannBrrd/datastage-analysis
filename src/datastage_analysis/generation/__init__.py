"""
Migration Code Generation Module

Generates migration code from DataStage job definitions.
Supports multiple targets: AWS Glue, SQL (Teradata, PostgreSQL, etc.)

Usage:
    # AWS Glue (default)
    from datastage_analysis.generation import MigrationGenerator

    generator = MigrationGenerator()
    results = generator.generate(predictions, structures)

    # SQL/Teradata
    generator = MigrationGenerator(target='sql', sql_dialect='teradata')
    results = generator.generate(predictions, structures)

Dry-run mode:
    from datastage_analysis.generation import DryRunEstimator

    estimator = DryRunEstimator(provider='anthropic')
    estimate = estimator.estimate(predictions, cluster_info)

Target generators:
    from datastage_analysis.generation.targets import get_target_generator

    glue_gen = get_target_generator('glue')
    sql_gen = get_target_generator('sql')
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
