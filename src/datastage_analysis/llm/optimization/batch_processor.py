"""
Batch Processor for Similar Jobs

Groups similar jobs together and processes them efficiently
using template-based generation to reduce LLM calls.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class JobBatch:
    """A batch of similar jobs."""
    batch_id: str
    template_job: str
    jobs: List[str]
    similarity_score: float
    pattern_type: str
    estimated_savings: float


class BatchProcessor:
    """
    Processes similar jobs in batches to reduce LLM calls.

    Strategy:
    1. Group jobs by similarity cluster or pattern family
    2. Select best template job from each group
    3. Generate code for template
    4. Apply template with variations to other jobs
    """

    def __init__(
        self,
        min_batch_size: int = 2,
        max_batch_size: int = 20,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize batch processor.

        Args:
            min_batch_size: Minimum jobs to form a batch
            max_batch_size: Maximum jobs in one batch
            similarity_threshold: Minimum similarity for batching
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.similarity_threshold = similarity_threshold

    def create_batches(
        self,
        predictions: List[Dict[str, Any]],
        structures: Dict[str, Dict[str, Any]]
    ) -> List[JobBatch]:
        """
        Create batches from job predictions.

        Args:
            predictions: List of job predictions with similarity info
            structures: Job structures dict

        Returns:
            List of JobBatch objects
        """
        batches = []

        # Group by similarity cluster
        cluster_groups = self._group_by_cluster(predictions)

        for cluster_id, jobs in cluster_groups.items():
            if len(jobs) < self.min_batch_size:
                continue

            # Select template (job with best structure representation)
            template_job = self._select_template(jobs, structures)

            # Split large groups
            for i in range(0, len(jobs), self.max_batch_size):
                batch_jobs = jobs[i:i + self.max_batch_size]

                if len(batch_jobs) < self.min_batch_size:
                    continue

                # Calculate savings
                savings = self._estimate_savings(batch_jobs)

                batch = JobBatch(
                    batch_id=f"batch_{cluster_id}_{i // self.max_batch_size}",
                    template_job=template_job,
                    jobs=[j['job_name'] for j in batch_jobs],
                    similarity_score=batch_jobs[0].get('similarity_score', 0.9),
                    pattern_type=batch_jobs[0].get('pattern_family', 'unknown'),
                    estimated_savings=savings
                )
                batches.append(batch)

        # Also group by pattern family for remaining jobs
        pattern_groups = self._group_by_pattern(predictions,
                                                 [j for b in batches for j in b.jobs])

        for pattern, jobs in pattern_groups.items():
            if len(jobs) < self.min_batch_size:
                continue

            template_job = self._select_template(jobs, structures)

            for i in range(0, len(jobs), self.max_batch_size):
                batch_jobs = jobs[i:i + self.max_batch_size]

                if len(batch_jobs) < self.min_batch_size:
                    continue

                savings = self._estimate_savings(batch_jobs)

                batch = JobBatch(
                    batch_id=f"pattern_{pattern}_{i // self.max_batch_size}",
                    template_job=template_job,
                    jobs=[j['job_name'] for j in batch_jobs],
                    similarity_score=0.8,
                    pattern_type=pattern,
                    estimated_savings=savings
                )
                batches.append(batch)

        logger.info(f"Created {len(batches)} batches from {len(predictions)} jobs")
        return batches

    def _group_by_cluster(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group jobs by similarity cluster."""
        groups = defaultdict(list)

        for pred in predictions:
            cluster = pred.get('similarity_cluster')
            if cluster and cluster != '-':
                groups[cluster].append(pred)

        return dict(groups)

    def _group_by_pattern(
        self,
        predictions: List[Dict[str, Any]],
        exclude_jobs: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group remaining jobs by pattern family."""
        groups = defaultdict(list)
        exclude_set = set(exclude_jobs)

        for pred in predictions:
            job_name = pred.get('job_name')
            if job_name in exclude_set:
                continue

            pattern = pred.get('pattern_family')
            if pattern and pattern != 'other':
                groups[pattern].append(pred)

        return dict(groups)

    def _select_template(
        self,
        jobs: List[Dict[str, Any]],
        structures: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Select best template job from a group.

        Criteria:
        - Has complete structure
        - Representative complexity
        - Good success probability
        """
        best_job = None
        best_score = -1

        for job in jobs:
            job_name = job.get('job_name')
            structure = structures.get(job_name, {})

            # Calculate template score
            score = 0

            # Prefer jobs with complete structures
            if structure:
                score += 10
                score += len(structure.get('stages', [])) * 0.1
                score += len(structure.get('links', [])) * 0.05

            # Prefer higher success probability
            score += job.get('success_probability', 0) * 5

            # Prefer moderate complexity (not too simple, not too complex)
            complexity = job.get('complexity_score', 5)
            if 3 <= complexity <= 7:
                score += 2

            if score > best_score:
                best_score = score
                best_job = job_name

        return best_job or jobs[0].get('job_name')

    def _estimate_savings(self, jobs: List[Dict[str, Any]]) -> float:
        """
        Estimate LLM call savings from batching.

        Without batching: N LLM calls
        With batching: 1 template call + N-1 variation applications
        """
        n_jobs = len(jobs)

        # Estimate tokens per job (average)
        avg_input_tokens = 2000
        avg_output_tokens = 1500

        # Cost without batching
        without_batch = n_jobs * (avg_input_tokens + avg_output_tokens)

        # Cost with batching (1 full call + small variation calls)
        template_cost = avg_input_tokens + avg_output_tokens
        variation_cost = (n_jobs - 1) * 500  # Much smaller variation requests
        with_batch = template_cost + variation_cost

        savings = 1 - (with_batch / without_batch) if without_batch > 0 else 0
        return round(savings, 2)

    def get_batch_prompt(
        self,
        batch: JobBatch,
        template_structure: Dict[str, Any],
        job_structures: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate a batch processing prompt.

        Args:
            batch: The job batch
            template_structure: Structure of template job
            job_structures: All job structures

        Returns:
            Optimized prompt for batch processing
        """
        lines = [
            f"# Batch Migration: {batch.batch_id}",
            f"Pattern: {batch.pattern_type}",
            f"Jobs: {len(batch.jobs)}",
            "",
            "## Template Job",
            f"Name: {batch.template_job}",
            "",
            self._format_structure(template_structure),
            "",
            "## Variations",
            "Generate AWS Glue code for these similar jobs:",
            ""
        ]

        for job_name in batch.jobs:
            if job_name == batch.template_job:
                continue

            job_struct = job_structures.get(job_name, {})
            diff = self._get_structure_diff(template_structure, job_struct)

            lines.append(f"### {job_name}")
            if diff:
                lines.extend(diff)
            else:
                lines.append("  (identical to template)")
            lines.append("")

        return '\n'.join(lines)

    def _format_structure(self, structure: Dict[str, Any]) -> str:
        """Format job structure for prompt."""
        lines = []

        stages = structure.get('stages', [])
        lines.append(f"Stages ({len(stages)}):")

        for stage in stages[:20]:  # Limit stages shown
            sname = stage.get('name', 'unnamed')
            stype = stage.get('type', 'Unknown')
            lines.append(f"  - {sname}: {stype}")

        if len(stages) > 20:
            lines.append(f"  ... and {len(stages) - 20} more stages")

        return '\n'.join(lines)

    def _get_structure_diff(
        self,
        template: Dict[str, Any],
        job: Dict[str, Any]
    ) -> List[str]:
        """Get differences between template and job structure."""
        diffs = []

        template_stages = {s.get('name'): s for s in template.get('stages', [])}
        job_stages = {s.get('name'): s for s in job.get('stages', [])}

        # Added stages
        added = set(job_stages.keys()) - set(template_stages.keys())
        if added:
            diffs.append(f"  Added: {', '.join(list(added)[:5])}")

        # Removed stages
        removed = set(template_stages.keys()) - set(job_stages.keys())
        if removed:
            diffs.append(f"  Removed: {', '.join(list(removed)[:5])}")

        # Different properties in common stages
        common = set(template_stages.keys()) & set(job_stages.keys())
        prop_diffs = []
        for name in list(common)[:10]:
            t_props = template_stages[name].get('properties', {})
            j_props = job_stages[name].get('properties', {})

            for key in ['table', 'file', 'connection', 'query']:
                if t_props.get(key) != j_props.get(key) and key in j_props:
                    prop_diffs.append(f"  {name}.{key}: {str(j_props[key])[:50]}")

        diffs.extend(prop_diffs[:5])

        return diffs


class BatchResult:
    """Result of batch processing."""

    def __init__(self):
        self.batches_processed = 0
        self.jobs_processed = 0
        self.llm_calls_saved = 0
        self.tokens_saved = 0
        self.results: Dict[str, Any] = {}

    def add_batch_result(
        self,
        batch: JobBatch,
        template_result: Any,
        job_results: Dict[str, Any],
        tokens_used: int
    ):
        """Record result of processing a batch."""
        self.batches_processed += 1
        self.jobs_processed += len(batch.jobs)

        # Calculate savings
        estimated_individual_tokens = len(batch.jobs) * 3500  # Average tokens per job
        self.tokens_saved += estimated_individual_tokens - tokens_used
        self.llm_calls_saved += len(batch.jobs) - 1  # Saved N-1 calls

        # Store results
        for job_name, result in job_results.items():
            self.results[job_name] = result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of batch processing."""
        return {
            'batches_processed': self.batches_processed,
            'jobs_processed': self.jobs_processed,
            'llm_calls_saved': self.llm_calls_saved,
            'tokens_saved': self.tokens_saved,
            'estimated_cost_saved': self.tokens_saved * 0.00001,  # Rough estimate
        }
