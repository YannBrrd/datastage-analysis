"""
Migration Target Generators

This module provides target-specific code generation for DataStage migrations.
Each target (Glue, SQL, etc.) implements the BaseTargetGenerator interface.

Usage:
    from datastage_analysis.generation.targets import get_target_generator

    generator = get_target_generator('glue')
    result = generator.generate(prediction, structure)
"""

from .base import (
    BaseTargetGenerator,
    GeneratedOutput,
    TargetConfig,
)
from .registry import (
    TargetRegistry,
    get_target_generator,
    register_target,
    list_available_targets,
)

__all__ = [
    'BaseTargetGenerator',
    'GeneratedOutput',
    'TargetConfig',
    'TargetRegistry',
    'get_target_generator',
    'register_target',
    'list_available_targets',
]
