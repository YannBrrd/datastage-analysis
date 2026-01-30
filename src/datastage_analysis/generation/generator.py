"""
Migration Generator

Main orchestrator for generating AWS Glue code from DataStage jobs.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config import get_config
from ..prediction.migration_predictor import MigrationPrediction, MigrationCategory

logger = logging.getLogger(__name__)


@dataclass
class GeneratedJob:
    """Result of generating code for a single job."""
    job_name: str
    category: MigrationCategory
    success: bool
    glue_script: Optional[str] = None
    terraform: Optional[str] = None
    unit_tests: Optional[str] = None
    documentation: Optional[str] = None
    error: Optional[str] = None
    generator_type: str = "rule_based"  # rule_based, llm_based, hybrid
    llm_tokens_used: int = 0
    generation_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_name': self.job_name,
            'category': self.category.value,
            'success': self.success,
            'generator_type': self.generator_type,
            'llm_tokens_used': self.llm_tokens_used,
            'generation_time_ms': self.generation_time_ms,
            'has_glue_script': self.glue_script is not None,
            'has_terraform': self.terraform is not None,
            'has_unit_tests': self.unit_tests is not None,
            'has_documentation': self.documentation is not None,
            'error': self.error,
            'warnings': self.warnings,
        }


@dataclass
class GenerationResult:
    """Result of generating code for multiple jobs."""
    jobs: Dict[str, GeneratedJob] = field(default_factory=dict)
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    total_llm_tokens: int = 0
    total_time_ms: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add(self, job_name: str, result: GeneratedJob):
        """Add a job result."""
        self.jobs[job_name] = result
        self.total_jobs += 1
        if result.success:
            self.successful_jobs += 1
        else:
            self.failed_jobs += 1
        self.total_llm_tokens += result.llm_tokens_used
        self.total_time_ms += result.generation_time_ms

    def merge(self, other: 'GenerationResult'):
        """Merge another result into this one."""
        for job_name, result in other.jobs.items():
            self.add(job_name, result)

    def get_summary(self) -> Dict[str, Any]:
        """Get generation summary."""
        return {
            'total_jobs': self.total_jobs,
            'successful_jobs': self.successful_jobs,
            'failed_jobs': self.failed_jobs,
            'success_rate': round(self.successful_jobs / self.total_jobs * 100, 1) if self.total_jobs > 0 else 0,
            'total_llm_tokens': self.total_llm_tokens,
            'total_time_ms': round(self.total_time_ms, 1),
            'by_generator': self._count_by_generator(),
            'generated_at': self.generated_at,
        }

    def _count_by_generator(self) -> Dict[str, int]:
        """Count jobs by generator type."""
        counts = {}
        for job in self.jobs.values():
            gen_type = job.generator_type
            counts[gen_type] = counts.get(gen_type, 0) + 1
        return counts


class MigrationGenerator:
    """
    Orchestrates migration code generation.

    Uses rule-based generation for AUTO jobs, LLM-assisted for others.
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the migration generator.

        Args:
            use_llm: Whether to use LLM for SEMI-AUTO and MANUAL jobs
        """
        self.config = get_config()
        self.use_llm = use_llm and self.config.get('llm', 'enabled', default=True)

        # Initialize generators
        from .rule_based import RuleBasedGenerator
        self.rule_based = RuleBasedGenerator()

        # Initialize LLM generator if enabled
        self.llm_generator = None
        if self.use_llm:
            try:
                from .llm_based import LLMGenerator
                from ..llm import get_llm_client
                client = get_llm_client()
                self.llm_generator = LLMGenerator(client)
                logger.info("LLM generator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM generator: {e}")
                self.use_llm = False

    def generate(
        self,
        predictions: List[MigrationPrediction],
        structures: Dict[str, Dict],
        jobs_filter: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate migration code for analyzed jobs.

        Args:
            predictions: Analysis results from MigrationPredictor
            structures: Job structures from parser
            jobs_filter: Optional list of job names to generate (None = all)
            output_dir: Directory to write generated files

        Returns:
            GenerationResult with all generated code
        """
        results = GenerationResult()
        output_dir = output_dir or self.config.get('generation', 'output_dir', default='./generated')

        # Filter predictions if needed
        if jobs_filter:
            predictions = [p for p in predictions if p.job_name in jobs_filter]

        logger.info(f"Generating code for {len(predictions)} jobs")

        # Group by category
        auto_jobs = [p for p in predictions if p.category == MigrationCategory.AUTO]
        semi_jobs = [p for p in predictions if p.category == MigrationCategory.SEMI_AUTO]
        manual_jobs = [p for p in predictions if p.category == MigrationCategory.MANUAL]

        logger.info(f"  AUTO: {len(auto_jobs)}, SEMI-AUTO: {len(semi_jobs)}, MANUAL: {len(manual_jobs)}")

        # Phase 1: Rule-based generation for AUTO jobs
        print(f"🔧 Generating AUTO jobs ({len(auto_jobs)}) with rule-based generator...")
        for pred in auto_jobs:
            structure = structures.get(pred.job_name, {})
            result = self._generate_auto(pred, structure)
            results.add(pred.job_name, result)

        # Phase 2: Hybrid generation for SEMI-AUTO jobs
        if semi_jobs:
            if self.use_llm and self.llm_generator:
                print(f"🤖 Generating SEMI-AUTO jobs ({len(semi_jobs)}) with hybrid generator...")
                for pred in semi_jobs:
                    structure = structures.get(pred.job_name, {})
                    result = self._generate_semi_auto(pred, structure)
                    results.add(pred.job_name, result)
            else:
                print(f"🔧 Generating SEMI-AUTO jobs ({len(semi_jobs)}) with rule-based (LLM disabled)...")
                for pred in semi_jobs:
                    structure = structures.get(pred.job_name, {})
                    result = self._generate_auto(pred, structure)
                    result.warnings.append("Generated with rule-based due to LLM disabled")
                    results.add(pred.job_name, result)

        # Phase 3: LLM generation for MANUAL jobs
        if manual_jobs:
            if self.use_llm and self.llm_generator:
                print(f"🤖 Generating MANUAL jobs ({len(manual_jobs)}) with LLM generator...")
                for pred in manual_jobs:
                    structure = structures.get(pred.job_name, {})
                    result = self._generate_manual(pred, structure)
                    results.add(pred.job_name, result)
            else:
                print(f"⚠️  Skipping MANUAL jobs ({len(manual_jobs)}) - LLM required but disabled")
                for pred in manual_jobs:
                    results.add(pred.job_name, GeneratedJob(
                        job_name=pred.job_name,
                        category=pred.category,
                        success=False,
                        error="LLM required for MANUAL jobs but is disabled",
                        generator_type="skipped",
                    ))

        # Write outputs
        if output_dir:
            self._write_outputs(results, output_dir)

        return results

    def _generate_auto(
        self,
        prediction: MigrationPrediction,
        structure: Dict
    ) -> GeneratedJob:
        """Generate code for AUTO job using rule-based generator."""
        import time
        start = time.time()

        try:
            result = self.rule_based.generate(prediction, structure)
            result.generation_time_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            logger.error(f"Error generating {prediction.job_name}: {e}")
            return GeneratedJob(
                job_name=prediction.job_name,
                category=prediction.category,
                success=False,
                error=str(e),
                generator_type="rule_based",
                generation_time_ms=(time.time() - start) * 1000,
            )

    def _generate_semi_auto(
        self,
        prediction: MigrationPrediction,
        structure: Dict
    ) -> GeneratedJob:
        """Generate code for SEMI-AUTO job using hybrid approach."""
        import time
        start = time.time()

        try:
            # First try rule-based for the basic structure
            result = self.rule_based.generate(prediction, structure)

            # Then use LLM for complex parts if available
            if self.llm_generator and prediction.automation_blockers:
                llm_result = self.llm_generator.enhance(
                    result,
                    prediction,
                    structure
                )
                result = llm_result
                result.generator_type = "hybrid"

            result.generation_time_ms = (time.time() - start) * 1000
            return result

        except Exception as e:
            logger.error(f"Error generating {prediction.job_name}: {e}")
            return GeneratedJob(
                job_name=prediction.job_name,
                category=prediction.category,
                success=False,
                error=str(e),
                generator_type="hybrid",
                generation_time_ms=(time.time() - start) * 1000,
            )

    def _generate_manual(
        self,
        prediction: MigrationPrediction,
        structure: Dict
    ) -> GeneratedJob:
        """Generate skeleton code for MANUAL job using LLM."""
        import time
        start = time.time()

        try:
            result = self.llm_generator.generate_skeleton(prediction, structure)
            result.generation_time_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            logger.error(f"Error generating {prediction.job_name}: {e}")
            return GeneratedJob(
                job_name=prediction.job_name,
                category=prediction.category,
                success=False,
                error=str(e),
                generator_type="llm_based",
                generation_time_ms=(time.time() - start) * 1000,
            )

    def _write_outputs(self, results: GenerationResult, output_dir: str):
        """Write generated files to disk."""
        base_path = Path(output_dir)

        # Create directories
        (base_path / 'glue_jobs').mkdir(parents=True, exist_ok=True)
        (base_path / 'terraform').mkdir(parents=True, exist_ok=True)
        (base_path / 'tests').mkdir(parents=True, exist_ok=True)
        (base_path / 'docs').mkdir(parents=True, exist_ok=True)

        written = 0
        for job_name, result in results.jobs.items():
            if not result.success:
                continue

            safe_name = self._safe_filename(job_name)

            if result.glue_script:
                path = base_path / 'glue_jobs' / f'{safe_name}.py'
                path.write_text(result.glue_script)
                written += 1

            if result.terraform:
                path = base_path / 'terraform' / f'{safe_name}.tf'
                path.write_text(result.terraform)

            if result.unit_tests:
                path = base_path / 'tests' / f'test_{safe_name}.py'
                path.write_text(result.unit_tests)

            if result.documentation:
                path = base_path / 'docs' / f'{safe_name}.md'
                path.write_text(result.documentation)

        print(f"📁 Written {written} Glue scripts to {output_dir}/glue_jobs/")

    def _safe_filename(self, name: str) -> str:
        """Convert job name to safe filename."""
        # Replace unsafe characters
        safe = name.lower()
        for char in ' /\\:*?"<>|':
            safe = safe.replace(char, '_')
        # Remove consecutive underscores
        while '__' in safe:
            safe = safe.replace('__', '_')
        return safe.strip('_')
