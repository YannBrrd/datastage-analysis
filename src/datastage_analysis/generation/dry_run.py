"""
Dry-Run Mode for Migration Generation

Estimates costs, previews batches, and validates configuration
without making actual LLM calls or generating code.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..prediction.migration_predictor import MigrationPrediction, MigrationCategory
from ..llm.optimization import TokenCounter, CostTracker

logger = logging.getLogger(__name__)


# Average tokens per job type (based on typical DataStage jobs)
ESTIMATED_TOKENS = {
    MigrationCategory.AUTO: {'input': 500, 'output': 800},
    MigrationCategory.SEMI_AUTO: {'input': 2000, 'output': 1500},
    MigrationCategory.MANUAL: {'input': 3000, 'output': 2500},
}

# Cost per 1K tokens by provider (simplified)
PROVIDER_COSTS = {
    'anthropic': {'input': 0.003, 'output': 0.015},
    'azure': {'input': 0.005, 'output': 0.015},
    'azure_foundry': {'input': 0.003, 'output': 0.01},
    'aws': {'input': 0.003, 'output': 0.015},
    'gcp': {'input': 0.00125, 'output': 0.005},
    'openrouter': {'input': 0.003, 'output': 0.015},
}


@dataclass
class BatchPreview:
    """Preview of a batch to be processed."""
    batch_id: str
    template_job: str
    jobs: List[str]
    job_count: int
    estimated_llm_calls: int
    estimated_tokens: int
    estimated_cost: float
    savings_vs_individual: float


@dataclass
class DryRunResult:
    """Result of dry-run analysis."""
    # Job counts
    total_jobs: int = 0
    auto_jobs: int = 0
    semi_auto_jobs: int = 0
    manual_jobs: int = 0

    # Batch analysis
    batches: List[BatchPreview] = field(default_factory=list)
    individual_jobs: List[str] = field(default_factory=list)

    # Cost estimates
    estimated_llm_calls: int = 0
    estimated_tokens: int = 0
    estimated_cost: float = 0.0

    # Savings from batching
    cost_without_batching: float = 0.0
    cost_with_batching: float = 0.0
    savings_amount: float = 0.0
    savings_percent: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)
    budget_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_jobs': self.total_jobs,
            'by_category': {
                'auto': self.auto_jobs,
                'semi_auto': self.semi_auto_jobs,
                'manual': self.manual_jobs,
            },
            'batches': [
                {
                    'batch_id': b.batch_id,
                    'template_job': b.template_job,
                    'job_count': b.job_count,
                    'jobs': b.jobs[:5] + (['...'] if len(b.jobs) > 5 else []),
                    'estimated_llm_calls': b.estimated_llm_calls,
                    'estimated_cost': round(b.estimated_cost, 4),
                }
                for b in self.batches
            ],
            'individual_jobs_count': len(self.individual_jobs),
            'cost_estimate': {
                'llm_calls': self.estimated_llm_calls,
                'tokens': self.estimated_tokens,
                'cost': round(self.estimated_cost, 4),
            },
            'batch_savings': {
                'without_batching': round(self.cost_without_batching, 4),
                'with_batching': round(self.cost_with_batching, 4),
                'savings_amount': round(self.savings_amount, 4),
                'savings_percent': round(self.savings_percent, 1),
            },
            'warnings': self.warnings,
            'budget_exceeded': self.budget_exceeded,
        }


class DryRunEstimator:
    """
    Estimates migration costs and previews batches without execution.
    """

    def __init__(
        self,
        provider: str = 'anthropic',
        budget_limit: Optional[float] = None
    ):
        """
        Initialize dry-run estimator.

        Args:
            provider: LLM provider to estimate costs for
            budget_limit: Optional budget limit for warnings
        """
        self.provider = provider
        self.budget_limit = budget_limit
        self.costs = PROVIDER_COSTS.get(provider, PROVIDER_COSTS['anthropic'])

    def estimate(
        self,
        predictions: List[MigrationPrediction],
        cluster_info: Optional[Dict[str, Dict]] = None,
        use_batching: bool = True
    ) -> DryRunResult:
        """
        Estimate costs and preview generation plan.

        Args:
            predictions: List of migration predictions
            cluster_info: Optional cluster information for batching
            use_batching: Whether to use batch optimization

        Returns:
            DryRunResult with estimates and previews
        """
        result = DryRunResult()
        result.total_jobs = len(predictions)

        # Count by category
        for pred in predictions:
            if pred.category == MigrationCategory.AUTO:
                result.auto_jobs += 1
            elif pred.category == MigrationCategory.SEMI_AUTO:
                result.semi_auto_jobs += 1
            else:
                result.manual_jobs += 1

        # AUTO jobs don't need LLM
        llm_jobs = [p for p in predictions
                    if p.category in (MigrationCategory.SEMI_AUTO, MigrationCategory.MANUAL)]

        if not llm_jobs:
            result.warnings.append("No jobs require LLM - all can be rule-based generated")
            return result

        # Calculate cost without batching (baseline)
        result.cost_without_batching = self._estimate_individual_cost(llm_jobs)

        # Calculate with batching if enabled and cluster info available
        if use_batching and cluster_info:
            self._estimate_with_batching(llm_jobs, cluster_info, result)
        else:
            # No batching - all individual
            result.individual_jobs = [p.job_name for p in llm_jobs]
            result.cost_with_batching = result.cost_without_batching
            result.estimated_llm_calls = len(llm_jobs)
            result.estimated_tokens = sum(
                ESTIMATED_TOKENS[p.category]['input'] + ESTIMATED_TOKENS[p.category]['output']
                for p in llm_jobs
            )

        result.estimated_cost = result.cost_with_batching
        result.savings_amount = result.cost_without_batching - result.cost_with_batching
        result.savings_percent = (
            (result.savings_amount / result.cost_without_batching * 100)
            if result.cost_without_batching > 0 else 0
        )

        # Budget check
        if self.budget_limit and result.estimated_cost > self.budget_limit:
            result.budget_exceeded = True
            result.warnings.append(
                f"Estimated cost ${result.estimated_cost:.2f} exceeds budget ${self.budget_limit:.2f}"
            )

        # Add informational warnings
        if result.manual_jobs > 0:
            result.warnings.append(
                f"{result.manual_jobs} MANUAL jobs will generate skeleton code requiring review"
            )

        return result

    def _estimate_individual_cost(self, predictions: List[MigrationPrediction]) -> float:
        """Estimate cost if all jobs processed individually."""
        total = 0.0
        for pred in predictions:
            tokens = ESTIMATED_TOKENS[pred.category]
            input_cost = (tokens['input'] / 1000) * self.costs['input']
            output_cost = (tokens['output'] / 1000) * self.costs['output']
            total += input_cost + output_cost
        return total

    def _estimate_with_batching(
        self,
        predictions: List[MigrationPrediction],
        cluster_info: Dict[str, Dict],
        result: DryRunResult
    ):
        """Estimate cost with batch optimization."""
        # Group jobs by cluster
        clusters = defaultdict(list)
        no_cluster = []

        for pred in predictions:
            info = cluster_info.get(pred.job_name, {})
            cluster_id = info.get('similarity_cluster') or info.get('duplicate_group')

            if cluster_id:
                clusters[cluster_id].append(pred)
            else:
                no_cluster.append(pred)

        total_cost = 0.0
        total_tokens = 0
        total_calls = 0

        # Process batches
        for cluster_id, jobs in clusters.items():
            if len(jobs) >= 2:
                # Create batch preview
                batch = self._create_batch_preview(cluster_id, jobs)
                result.batches.append(batch)
                total_cost += batch.estimated_cost
                total_tokens += batch.estimated_tokens
                total_calls += batch.estimated_llm_calls
            else:
                # Single job in cluster - treat as individual
                no_cluster.extend(jobs)

        # Process individual jobs
        for pred in no_cluster:
            result.individual_jobs.append(pred.job_name)
            tokens = ESTIMATED_TOKENS[pred.category]
            input_cost = (tokens['input'] / 1000) * self.costs['input']
            output_cost = (tokens['output'] / 1000) * self.costs['output']
            total_cost += input_cost + output_cost
            total_tokens += tokens['input'] + tokens['output']
            total_calls += 1

        result.cost_with_batching = total_cost
        result.estimated_tokens = total_tokens
        result.estimated_llm_calls = total_calls

    def _create_batch_preview(
        self,
        cluster_id: str,
        jobs: List[MigrationPrediction]
    ) -> BatchPreview:
        """Create preview for a batch."""
        # Template uses full tokens
        template = jobs[0]
        template_tokens = ESTIMATED_TOKENS[template.category]

        # Variations use reduced tokens (simple string replacement = 0, complex = 30%)
        variation_tokens = int(
            (template_tokens['input'] + template_tokens['output']) * 0.15
        )

        # Estimate: 1 full call + ~20% of variations need LLM
        full_calls = 1
        variation_calls = max(1, int((len(jobs) - 1) * 0.2))
        total_calls = full_calls + variation_calls

        # Token estimate
        total_tokens = (
            template_tokens['input'] + template_tokens['output'] +
            variation_calls * variation_tokens
        )

        # Cost estimate
        cost = (
            (template_tokens['input'] / 1000) * self.costs['input'] +
            (template_tokens['output'] / 1000) * self.costs['output'] +
            (variation_calls * variation_tokens / 1000) * self.costs['output']
        )

        # Individual cost for comparison
        individual_cost = sum(
            (ESTIMATED_TOKENS[j.category]['input'] / 1000) * self.costs['input'] +
            (ESTIMATED_TOKENS[j.category]['output'] / 1000) * self.costs['output']
            for j in jobs
        )

        return BatchPreview(
            batch_id=cluster_id,
            template_job=template.job_name,
            jobs=[j.job_name for j in jobs],
            job_count=len(jobs),
            estimated_llm_calls=total_calls,
            estimated_tokens=total_tokens,
            estimated_cost=cost,
            savings_vs_individual=individual_cost - cost,
        )


def format_dry_run_report(result: DryRunResult, provider: str = 'anthropic') -> str:
    """Format dry-run result as readable report."""
    lines = [
        "=" * 70,
        "DRY-RUN: MIGRATION COST ESTIMATE",
        "=" * 70,
        "",
        f"Provider: {provider}",
        "",
        "JOB SUMMARY",
        "-" * 40,
        f"Total Jobs:      {result.total_jobs}",
        f"  AUTO:          {result.auto_jobs} (rule-based, no LLM)",
        f"  SEMI-AUTO:     {result.semi_auto_jobs} (hybrid, LLM enhanced)",
        f"  MANUAL:        {result.manual_jobs} (LLM skeleton generation)",
        "",
    ]

    if result.batches:
        lines.extend([
            "BATCH OPTIMIZATION",
            "-" * 40,
            f"Batches identified: {len(result.batches)}",
            "",
        ])

        for batch in result.batches[:10]:
            lines.append(f"  {batch.batch_id}:")
            lines.append(f"    Template: {batch.template_job}")
            lines.append(f"    Jobs: {batch.job_count}")
            lines.append(f"    Est. LLM calls: {batch.estimated_llm_calls} (vs {batch.job_count} without batching)")
            lines.append(f"    Est. savings: ${batch.savings_vs_individual:.4f}")
            lines.append("")

        if len(result.batches) > 10:
            lines.append(f"  ... and {len(result.batches) - 10} more batches")
            lines.append("")

    if result.individual_jobs:
        lines.extend([
            f"Individual jobs (no cluster): {len(result.individual_jobs)}",
            "",
        ])

    lines.extend([
        "COST ESTIMATE",
        "-" * 40,
        f"Estimated LLM calls:    {result.estimated_llm_calls:,}",
        f"Estimated tokens:       {result.estimated_tokens:,}",
        "",
        f"Cost WITHOUT batching:  ${result.cost_without_batching:.4f}",
        f"Cost WITH batching:     ${result.cost_with_batching:.4f}",
        f"SAVINGS:                ${result.savings_amount:.4f} ({result.savings_percent:.1f}%)",
        "",
    ])

    if result.warnings:
        lines.extend([
            "WARNINGS",
            "-" * 40,
        ])
        for warning in result.warnings:
            lines.append(f"  ⚠️  {warning}")
        lines.append("")

    if result.budget_exceeded:
        lines.extend([
            "🚨 BUDGET EXCEEDED - Generation will not proceed without --force flag",
            "",
        ])

    lines.extend([
        "=" * 70,
        "To proceed with generation, remove --dry-run flag",
        "=" * 70,
    ])

    return '\n'.join(lines)
