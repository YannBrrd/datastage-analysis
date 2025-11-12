"""
Smart Job Summarizer - Extract minimal but meaningful job representations
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class JobSummary:
    """Compact representation of a DataStage job for LLM analysis."""
    name: str
    complexity: float
    stage_types: List[str]  # Unique types only
    stage_count: int
    source_systems: List[str]
    target_systems: List[str]
    transformation_patterns: List[str]
    business_logic_keywords: List[str]
    
    def to_prompt_text(self) -> str:
        """Convert to compact text for LLM prompt (< 500 tokens)."""
        return f"""Job: {self.name}
Complexity: {self.complexity}/100
Architecture: {self.stage_count} stages
Sources: {', '.join(self.source_systems[:3])}
Targets: {', '.join(self.target_systems[:3])}
Transforms: {', '.join(self.transformation_patterns[:5])}
Logic: {', '.join(self.business_logic_keywords[:5])}"""


class SmartJobSummarizer:
    """Create minimal but meaningful job summaries for LLM processing."""
    
    # Keywords to extract business logic
    BUSINESS_KEYWORDS = {
        'customer', 'order', 'payment', 'invoice', 'product', 'sale',
        'transaction', 'account', 'balance', 'aggregate', 'calculate',
        'validate', 'enrich', 'deduplicate', 'merge', 'reconcile'
    }
    
    def summarize_job(self, job, pattern: Any) -> JobSummary:
        """Create a compact summary from full job object."""
        structure = job.structure
        
        # Extract unique stage types (not counts)
        stage_types = list(set(s.get('type', 'Unknown') for s in structure.get('stages', [])))
        
        # Extract business keywords from job name and properties
        business_keywords = self._extract_business_keywords(job, structure)
        
        return JobSummary(
            name=job.name,
            complexity=pattern.complexity_score,
            stage_types=stage_types[:10],  # Max 10 types
            stage_count=pattern.stage_count,
            source_systems=pattern.source_types[:3],  # Max 3
            target_systems=pattern.target_types[:3],
            transformation_patterns=pattern.transformation_types[:5],  # Max 5
            business_logic_keywords=business_keywords[:5]
        )
    
    def _extract_business_keywords(self, job, structure: Dict) -> List[str]:
        """Extract business-relevant keywords from job."""
        keywords = set()
        
        # From job name
        name_lower = job.name.lower()
        for keyword in self.BUSINESS_KEYWORDS:
            if keyword in name_lower:
                keywords.add(keyword)
        
        # From stage properties (table names, queries)
        for stage in structure.get('stages', []):
            props = stage.get('properties', {})
            for key in ['Table', 'TableName', 'FileName']:
                value = props.get(key, '').lower()
                for keyword in self.BUSINESS_KEYWORDS:
                    if keyword in value:
                        keywords.add(keyword)
        
        return list(keywords)
    
    def create_batch_for_llm(self, summaries: List[JobSummary], max_tokens: int = 15000) -> List[List[JobSummary]]:
        """
        Group job summaries into batches that fit within token limits.
        
        Strategy: ~500 tokens per job summary, so ~30 jobs per batch for 15K tokens.
        """
        batches = []
        current_batch = []
        current_tokens = 0
        tokens_per_job = 500  # Conservative estimate
        
        for summary in summaries:
            if current_tokens + tokens_per_job > max_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [summary]
                current_tokens = tokens_per_job
            else:
                current_batch.append(summary)
                current_tokens += tokens_per_job
        
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches from {len(summaries)} job summaries")
        return batches
