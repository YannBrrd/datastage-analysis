"""
Prompt Optimizer

Compresses and optimizes prompts to reduce token usage.
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    Optimizes prompts to reduce token usage while preserving meaning.

    Techniques:
    - Remove excessive whitespace
    - Deduplicate repeated content
    - Compress stage definitions
    - Use abbreviations for common patterns
    """

    # Common abbreviations for DataStage terms
    ABBREVIATIONS = {
        'OracleConnectorPX': 'OraclePX',
        'DB2ConnectorPX': 'DB2PX',
        'TeradataConnectorPX': 'TeradataPX',
        'SQLServerConnectorPX': 'SQLServerPX',
        'PxSequentialFile': 'SeqFile',
        'SequentialFile': 'SeqFile',
        'CTransformerStage': 'Transformer',
        'PxTransform': 'Transform',
        'transformation_ctx': 'ctx',
        'DynamicFrame': 'DynFrame',
        'glueContext': 'glueCx',
    }

    def __init__(self, target_reduction: float = 0.3):
        """
        Initialize optimizer.

        Args:
            target_reduction: Target reduction ratio (0.3 = 30% smaller)
        """
        self.target_reduction = target_reduction

    def optimize(self, prompt: str) -> str:
        """
        Optimize prompt to reduce token count.

        Args:
            prompt: Original prompt

        Returns:
            Optimized prompt
        """
        original_length = len(prompt)

        # Apply optimization techniques
        prompt = self._normalize_whitespace(prompt)
        prompt = self._compress_json(prompt)
        prompt = self._deduplicate_stages(prompt)
        prompt = self._abbreviate_terms(prompt)
        prompt = self._compress_properties(prompt)

        optimized_length = len(prompt)
        reduction = 1 - (optimized_length / original_length) if original_length > 0 else 0

        logger.debug(f"Prompt optimized: {original_length} -> {optimized_length} chars ({reduction:.1%} reduction)")

        return prompt

    def _normalize_whitespace(self, text: str) -> str:
        """Remove excessive whitespace."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        return text.strip()

    def _compress_json(self, text: str) -> str:
        """Compress JSON-like structures."""
        # Remove unnecessary spaces in JSON
        text = re.sub(r':\s+', ': ', text)
        text = re.sub(r',\s+', ', ', text)
        text = re.sub(r'\{\s+', '{', text)
        text = re.sub(r'\s+\}', '}', text)
        text = re.sub(r'\[\s+', '[', text)
        text = re.sub(r'\s+\]', ']', text)

        return text

    def _deduplicate_stages(self, text: str) -> str:
        """
        Deduplicate repeated stage definitions.

        If the same stage type appears multiple times, reference the first.
        """
        # Find stage definitions (simplified pattern)
        stage_pattern = r'Stage:\s*(\w+)\s*\((\w+)\)'
        seen_types = {}
        dedup_count = 0

        def replace_duplicate(match):
            nonlocal dedup_count
            name, stage_type = match.group(1), match.group(2)

            if stage_type in seen_types and dedup_count < 5:
                # Already seen this type, abbreviate
                dedup_count += 1
                return f"Stage: {name} (same as {seen_types[stage_type]})"

            seen_types[stage_type] = name
            return match.group(0)

        return re.sub(stage_pattern, replace_duplicate, text)

    def _abbreviate_terms(self, text: str) -> str:
        """Replace common terms with abbreviations."""
        for full, abbrev in self.ABBREVIATIONS.items():
            text = text.replace(full, abbrev)

        return text

    def _compress_properties(self, text: str) -> str:
        """Compress property listings."""
        # Compress property: value patterns
        text = re.sub(
            r'property:\s*(\w+)\s*=\s*"([^"]*)"',
            r'\1="\2"',
            text
        )

        # Compress common property patterns
        text = re.sub(r'name\s*=\s*', 'n=', text)
        text = re.sub(r'type\s*=\s*', 't=', text)
        text = re.sub(r'value\s*=\s*', 'v=', text)

        return text

    def optimize_for_model(self, prompt: str, model: str, max_tokens: int) -> str:
        """
        Optimize prompt for specific model context window.

        Args:
            prompt: Original prompt
            model: Model identifier
            max_tokens: Maximum context tokens

        Returns:
            Optimized prompt that fits context
        """
        from .token_counter import TokenCounter

        counter = TokenCounter(model)
        current_tokens = counter.count(prompt)

        if current_tokens <= max_tokens:
            return self.optimize(prompt)

        # Need aggressive optimization
        optimized = self.optimize(prompt)
        current_tokens = counter.count(optimized)

        if current_tokens <= max_tokens:
            return optimized

        # Still too large - truncate with summary
        reduction_needed = current_tokens / max_tokens
        target_chars = int(len(optimized) / reduction_needed * 0.9)

        logger.warning(f"Prompt too large ({current_tokens} tokens), truncating to ~{max_tokens} tokens")

        # Keep beginning and end, summarize middle
        keep_start = target_chars // 2
        keep_end = target_chars // 3

        truncated = (
            optimized[:keep_start] +
            f"\n\n[... {len(optimized) - keep_start - keep_end} chars truncated ...]\n\n" +
            optimized[-keep_end:]
        )

        return truncated


class ContextBuilder:
    """
    Builds optimized context for LLM prompts.

    Intelligently selects and compresses relevant information
    to fit within token limits.
    """

    def __init__(self, max_tokens: int = 4000):
        """
        Initialize context builder.

        Args:
            max_tokens: Maximum tokens for context
        """
        self.max_tokens = max_tokens
        self.optimizer = PromptOptimizer()

    def build_job_context(
        self,
        job_name: str,
        structure: Dict[str, Any],
        include_properties: bool = True,
        max_stages: int = 50
    ) -> str:
        """
        Build optimized context for a job.

        Args:
            job_name: Job name
            structure: Job structure dict
            include_properties: Include stage properties
            max_stages: Maximum stages to include

        Returns:
            Optimized context string
        """
        lines = [f"Job: {job_name}"]

        stages = structure.get('stages', [])[:max_stages]

        if len(structure.get('stages', [])) > max_stages:
            lines.append(f"Note: Showing {max_stages} of {len(structure['stages'])} stages")

        # Group stages by type
        stage_types = {}
        for stage in stages:
            stype = stage.get('type', 'Unknown')
            if stype not in stage_types:
                stage_types[stype] = []
            stage_types[stype].append(stage)

        lines.append(f"\nStage Types: {', '.join(stage_types.keys())}")

        # List stages with limited properties
        lines.append("\nStages:")
        for stage in stages:
            sname = stage.get('name', 'unnamed')
            stype = stage.get('type', 'Unknown')
            lines.append(f"  - {sname} ({stype})")

            if include_properties:
                props = stage.get('properties', {})
                # Only include key properties
                key_props = {k: v for k, v in list(props.items())[:5]
                            if k in ('table', 'file', 'connection', 'sql', 'query')}
                if key_props:
                    for k, v in key_props.items():
                        v_str = str(v)[:50] + '...' if len(str(v)) > 50 else str(v)
                        lines.append(f"      {k}: {v_str}")

        context = '\n'.join(lines)
        return self.optimizer.optimize(context)

    def build_batch_context(
        self,
        jobs: List[Dict[str, Any]],
        template_job: str
    ) -> str:
        """
        Build context for batch processing similar jobs.

        Args:
            jobs: List of job structures
            template_job: Name of job to use as template

        Returns:
            Optimized batch context
        """
        lines = [
            f"Batch of {len(jobs)} similar jobs.",
            f"Template job: {template_job}",
            "",
            "Differences from template:"
        ]

        # Find template
        template = next((j for j in jobs if j.get('name') == template_job), jobs[0])

        for job in jobs:
            if job.get('name') == template_job:
                continue

            diffs = self._find_differences(template, job)
            if diffs:
                lines.append(f"\n{job.get('name')}:")
                for diff in diffs[:5]:  # Limit differences shown
                    lines.append(f"  - {diff}")

        return '\n'.join(lines)

    def _find_differences(
        self,
        template: Dict[str, Any],
        job: Dict[str, Any]
    ) -> List[str]:
        """Find key differences between template and job."""
        diffs = []

        template_stages = {s.get('name'): s for s in template.get('stages', [])}
        job_stages = {s.get('name'): s for s in job.get('stages', [])}

        # Find added/removed stages
        added = set(job_stages.keys()) - set(template_stages.keys())
        removed = set(template_stages.keys()) - set(job_stages.keys())

        if added:
            diffs.append(f"Added stages: {', '.join(list(added)[:3])}")
        if removed:
            diffs.append(f"Removed stages: {', '.join(list(removed)[:3])}")

        # Find property differences in common stages
        common = set(template_stages.keys()) & set(job_stages.keys())
        for name in list(common)[:5]:
            t_props = template_stages[name].get('properties', {})
            j_props = job_stages[name].get('properties', {})

            for key in set(t_props.keys()) | set(j_props.keys()):
                if t_props.get(key) != j_props.get(key):
                    diffs.append(f"{name}.{key} differs")
                    break

        return diffs
