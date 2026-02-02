"""
AWS Glue Target Configuration

Glue-specific configuration options.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from ..base import TargetConfig


@dataclass
class GlueTargetConfig(TargetConfig):
    """
    Configuration for AWS Glue target generator.

    Extends TargetConfig with Glue-specific options.
    """
    target_name: str = "glue"

    # Glue version and runtime
    glue_version: str = "4.0"
    python_version: str = "3"

    # Worker configuration
    default_worker_type: str = "G.1X"
    default_num_workers: int = 2
    max_workers: int = 10

    # Job bookmarks
    enable_bookmarks: bool = True

    # Monitoring
    enable_metrics: bool = True
    enable_spark_ui: bool = True
    enable_continuous_logging: bool = True

    # Timeout (minutes)
    default_timeout: int = 120

    # S3 paths (templates)
    scripts_bucket_var: str = "var.scripts_bucket"
    temp_bucket_var: str = "var.temp_bucket"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlueTargetConfig':
        """Create config from dictionary."""
        return cls(
            target_name=data.get('target_name', 'glue'),
            enabled=data.get('enabled', True),
            generate_infrastructure=data.get('generate_infrastructure', True),
            generate_tests=data.get('generate_tests', True),
            generate_docs=data.get('generate_docs', True),
            options=data.get('options', {}),
            glue_version=data.get('glue_version', '4.0'),
            python_version=data.get('python_version', '3'),
            default_worker_type=data.get('default_worker_type', 'G.1X'),
            default_num_workers=data.get('default_num_workers', 2),
            max_workers=data.get('max_workers', 10),
            enable_bookmarks=data.get('enable_bookmarks', True),
            enable_metrics=data.get('enable_metrics', True),
            enable_spark_ui=data.get('enable_spark_ui', True),
            enable_continuous_logging=data.get('enable_continuous_logging', True),
            default_timeout=data.get('default_timeout', 120),
            scripts_bucket_var=data.get('scripts_bucket_var', 'var.scripts_bucket'),
            temp_bucket_var=data.get('temp_bucket_var', 'var.temp_bucket'),
        )

    def get_worker_config(self, complexity_score: float = 0.5) -> Dict[str, Any]:
        """
        Get worker configuration based on job complexity.

        Args:
            complexity_score: 0.0-1.0 score indicating job complexity

        Returns:
            Dict with worker_type and number_of_workers
        """
        if complexity_score > 0.7:
            return {
                'worker_type': 'G.2X',
                'number_of_workers': min(self.max_workers, 4),
            }
        elif complexity_score > 0.4:
            return {
                'worker_type': 'G.1X',
                'number_of_workers': min(self.max_workers, 3),
            }
        else:
            return {
                'worker_type': self.default_worker_type,
                'number_of_workers': self.default_num_workers,
            }
