"""
Base Target Generator Interface

Defines the abstract interface that all target generators must implement.
This allows the system to support multiple output targets (Glue, SQL, etc.)
with a consistent API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from ...prediction.migration_predictor import MigrationPrediction, MigrationCategory


class OutputType(Enum):
    """Types of outputs a target generator can produce."""
    MAIN_SCRIPT = "main_script"       # Primary migration code (Glue script, SQL script)
    INFRASTRUCTURE = "infrastructure"  # Infrastructure code (Terraform, DDL)
    UNIT_TESTS = "unit_tests"         # Test files
    DOCUMENTATION = "documentation"    # Documentation files


@dataclass
class GeneratedOutput:
    """
    Result of generating code for a single job.

    This is the target-agnostic output format that all generators produce.
    """
    job_name: str
    category: MigrationCategory
    success: bool
    target: str  # 'glue', 'sql', etc.

    # Primary output
    main_script: Optional[str] = None
    main_script_extension: str = ".py"  # .py for Glue, .sql for SQL

    # Infrastructure
    infrastructure: Optional[str] = None
    infrastructure_extension: str = ".tf"  # .tf for Terraform, .sql for DDL
    infrastructure_type: str = "terraform"  # terraform, ddl, etc.

    # Secondary outputs
    unit_tests: Optional[str] = None
    documentation: Optional[str] = None

    # Metadata
    error: Optional[str] = None
    generator_type: str = "rule_based"  # rule_based, llm_based, hybrid, batch_variation
    llm_tokens_used: int = 0
    generation_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # Batch processing metadata
    batch_id: Optional[str] = None
    template_job: Optional[str] = None

    # Target-specific metadata
    target_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_name': self.job_name,
            'category': self.category.value if self.category else None,
            'success': self.success,
            'target': self.target,
            'generator_type': self.generator_type,
            'llm_tokens_used': self.llm_tokens_used,
            'generation_time_ms': self.generation_time_ms,
            'has_main_script': self.main_script is not None,
            'has_infrastructure': self.infrastructure is not None,
            'has_unit_tests': self.unit_tests is not None,
            'has_documentation': self.documentation is not None,
            'error': self.error,
            'warnings': self.warnings,
            'batch_id': self.batch_id,
            'template_job': self.template_job,
            'target_metadata': self.target_metadata,
        }


@dataclass
class TargetConfig:
    """
    Configuration for a target generator.

    Each target can have specific configuration options.
    """
    target_name: str
    enabled: bool = True

    # Output options
    generate_infrastructure: bool = True
    generate_tests: bool = True
    generate_docs: bool = True

    # Target-specific options (subclasses add their own)
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TargetConfig':
        """Create config from dictionary."""
        return cls(
            target_name=data.get('target_name', 'unknown'),
            enabled=data.get('enabled', True),
            generate_infrastructure=data.get('generate_infrastructure', True),
            generate_tests=data.get('generate_tests', True),
            generate_docs=data.get('generate_docs', True),
            options=data.get('options', {}),
        )


class BaseTargetGenerator(ABC):
    """
    Abstract base class for target-specific code generators.

    Each target (Glue, SQL, etc.) must implement this interface
    to provide consistent code generation capabilities.
    """

    # Class attributes that subclasses should override
    TARGET_NAME: str = "base"  # Unique identifier for this target
    TARGET_DISPLAY_NAME: str = "Base Target"  # Human-readable name
    MAIN_SCRIPT_EXTENSION: str = ".txt"
    INFRASTRUCTURE_EXTENSION: str = ".txt"
    INFRASTRUCTURE_TYPE: str = "generic"

    def __init__(self, config: Optional[TargetConfig] = None):
        """
        Initialize the target generator.

        Args:
            config: Optional configuration for this target
        """
        self.config = config or TargetConfig(target_name=self.TARGET_NAME)

    @abstractmethod
    def generate(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> GeneratedOutput:
        """
        Generate migration code for a single job.

        This is the main entry point for code generation.

        Args:
            prediction: Migration prediction with category and analysis
            structure: Parsed job structure from DSX parser
            **kwargs: Additional target-specific arguments

        Returns:
            GeneratedOutput with all generated code
        """
        pass

    @abstractmethod
    def generate_main_script(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> Tuple[str, List[str]]:
        """
        Generate the main migration script.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            **kwargs: Additional arguments

        Returns:
            Tuple of (script_code, warnings_list)
        """
        pass

    @abstractmethod
    def generate_infrastructure(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> Optional[str]:
        """
        Generate infrastructure code (Terraform, DDL, etc.).

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            **kwargs: Additional arguments

        Returns:
            Infrastructure code or None if not applicable
        """
        pass

    def generate_tests(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        main_script: str,
        **kwargs
    ) -> Optional[str]:
        """
        Generate unit tests for the migration code.

        Default implementation returns None. Override in subclasses.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            main_script: The generated main script to test
            **kwargs: Additional arguments

        Returns:
            Test code or None if not applicable
        """
        return None

    def generate_documentation(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        main_script: str,
        **kwargs
    ) -> Optional[str]:
        """
        Generate documentation for the migration.

        Default implementation returns None. Override in subclasses.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            main_script: The generated main script to document
            **kwargs: Additional arguments

        Returns:
            Documentation markdown or None
        """
        return None

    def supports_batch_generation(self) -> bool:
        """
        Check if this target supports batch generation.

        Batch generation allows generating code for similar jobs
        efficiently by using a template job.

        Returns:
            True if batch generation is supported
        """
        return False

    def generate_batch(
        self,
        template_prediction: MigrationPrediction,
        template_structure: Dict[str, Any],
        similar_jobs: List[Tuple[str, MigrationPrediction, Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, GeneratedOutput]:
        """
        Generate code for a batch of similar jobs.

        Default implementation generates each job individually.
        Override in subclasses for optimized batch processing.

        Args:
            template_prediction: Prediction for the template job
            template_structure: Structure of the template job
            similar_jobs: List of (job_name, prediction, structure) tuples
            **kwargs: Additional arguments

        Returns:
            Dict mapping job names to GeneratedOutput
        """
        results = {}

        # Generate template
        template_result = self.generate(template_prediction, template_structure, **kwargs)
        results[template_prediction.job_name] = template_result

        # Generate each similar job individually (subclasses can optimize)
        for job_name, prediction, structure in similar_jobs:
            result = self.generate(prediction, structure, **kwargs)
            results[job_name] = result

        return results

    def get_output_paths(self, job_name: str, base_dir: str) -> Dict[OutputType, str]:
        """
        Get the output file paths for a job.

        Args:
            job_name: Name of the job
            base_dir: Base output directory

        Returns:
            Dict mapping OutputType to file paths
        """
        from pathlib import Path

        safe_name = self._safe_filename(job_name)
        base = Path(base_dir)

        return {
            OutputType.MAIN_SCRIPT: str(base / self.TARGET_NAME / f'{safe_name}{self.MAIN_SCRIPT_EXTENSION}'),
            OutputType.INFRASTRUCTURE: str(base / 'infrastructure' / f'{safe_name}{self.INFRASTRUCTURE_EXTENSION}'),
            OutputType.UNIT_TESTS: str(base / 'tests' / f'test_{safe_name}.py'),
            OutputType.DOCUMENTATION: str(base / 'docs' / f'{safe_name}.md'),
        }

    def _safe_filename(self, name: str) -> str:
        """Convert job name to safe filename."""
        safe = name.lower()
        for char in ' /\\:*?"<>|':
            safe = safe.replace(char, '_')
        while '__' in safe:
            safe = safe.replace('__', '_')
        return safe.strip('_')

    def validate_structure(self, structure: Dict[str, Any]) -> List[str]:
        """
        Validate the job structure for this target.

        Args:
            structure: Parsed job structure

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not structure:
            errors.append("Empty job structure")
            return errors

        if 'stages' not in structure and 'links' not in structure:
            errors.append("Job structure missing stages and links")

        return errors

    def get_complexity_factors(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get target-specific complexity factors.

        These factors can be used for migration planning and estimation.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure

        Returns:
            Dict with complexity factors
        """
        return {
            'category': prediction.category.value,
            'automation_blockers': prediction.automation_blockers,
            'success_probability': prediction.success_probability,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target={self.TARGET_NAME})"
