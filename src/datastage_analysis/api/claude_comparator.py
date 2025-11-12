"""
Claude API Comparator Module

Handles fine-grained comparison using Claude AI with prompt caching for efficiency.
Optimized for large-scale batch processing with 90% token cost reduction.
"""

import asyncio
import logging
from typing import List, Dict, Any, Tuple
import json
from anthropic import AsyncAnthropic
from dataclasses import dataclass

from ..cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two jobs."""
    job1_idx: int
    job2_idx: int
    similarity_score: float
    differences: List[str]
    reasoning: str


class ClaudeComparator:
    """
    Compares jobs using Claude AI with advanced optimizations:
    - Prompt caching (90% cost reduction on repeated context)
    - Batch processing (multiple comparisons per API call)
    - Smart summarization (send only essential metadata, not full XML)
    """

    def __init__(self, api_key: str = None, cache: RedisCache = None, model: str = "claude-3-5-sonnet-20241022"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.cache = cache
        self.model = model  # Sonnet for better analysis, Haiku for speed
        self.system_prompt_cached = self._create_cached_system_prompt()

    def _create_cached_system_prompt(self) -> str:
        """
        System prompt with cache_control - reused across ALL batches.
        Claude caches prompts > 1024 tokens, reducing cost by 90%.
        """
        return """You are a DataStage to PySpark migration expert analyzing job patterns.

CONTEXT:
- DataStage: IBM ETL tool with proprietary stages (Oracle Connector, Transformer, etc.)
- PySpark: Apache Spark Python API for distributed data processing
- Migration Goal: Identify reusable patterns, complexity factors, and risks

DATASTAGE STAGE TYPES:
SOURCES: OracleConnectorPX, TeradataConnectorPX, DB2ConnectorPX, SequentialFile, ODBC
TARGETS: Same connectors in write mode, DataSet, Peek
TRANSFORMERS: Transformer (business logic), Aggregator, Joiner, Lookup, Funnel, Sort, Remove Duplicates

PYSPARK MIGRATION COMPLEXITY:
- Oracle/Teradata Connectors → spark.read.jdbc() [Medium - 3/5]
- Transformer (simple) → DataFrame.withColumn() [Simple - 1/5]
- Transformer (complex SQL) → Custom UDFs, testing needed [Hard - 4/5]
- Aggregator → groupBy().agg() [Simple - 2/5]
- Joiner → DataFrame.join() [Medium - 3/5]
- Lookup → broadcast join or merge [Medium - 3/5]
- Remove Duplicates → dropDuplicates() [Simple - 1/5]

ANALYSIS FRAMEWORK:
1. Structural Similarity: Do jobs share source→transform→target pattern?
2. Reusability: Can PySpark code be templated/shared?
3. Migration Risks: Complex transformations, performance bottlenecks, data quality
4. Effort: Factor in testing, validation, tuning

OUTPUT FORMAT (per comparison):
- Similarity: 0-100
- Patterns: Common reusable elements
- Risks: Specific challenges
- Advice: Actionable migration recommendations

Analyze concisely - focus on actionable migration insights."""

    async def compare_batch_optimized(self, job_summaries: List[Tuple[Any, Any]]) -> List[Dict]:
        """
        Compare multiple job pairs with smart batching and caching.
        
        Strategy:
        1. Check Redis cache for each pair
        2. Group uncached pairs into batches of 10-15
        3. Use prompt caching to reuse system context
        4. Parse responses and cache results
        
        Args:
            job_summaries: List of (JobSummary1, JobSummary2) tuples
            
        Returns:
            List of comparison dicts with similarity, patterns, risks
        """
        from .job_summarizer import JobSummary
        
        results = []
        uncached_pairs = []
        uncached_indices = []
        
        # Phase 1: Check cache
        for i, (summary1, summary2) in enumerate(job_summaries):
            cache_key = f"comparison_v2:{summary1.name}:{summary2.name}"
            cached = await self.cache.get(cache_key) if self.cache else None
            
            if cached:
                logger.info(f"✓ Cache hit: {summary1.name} vs {summary2.name}")
                results.append({"cached": True, **cached})
            else:
                results.append(None)  # Placeholder
                uncached_pairs.append((summary1, summary2))
                uncached_indices.append(i)
        
        if not uncached_pairs:
            logger.info("All comparisons cached!")
            return results
        
        # Phase 2: Batch process uncached pairs
        logger.info(f"Processing {len(uncached_pairs)} uncached comparisons...")
        batch_size = 12  # ~500 tokens/job × 2 jobs × 12 = 12K tokens per batch
        
        for batch_start in range(0, len(uncached_pairs), batch_size):
            batch = uncached_pairs[batch_start:batch_start + batch_size]
            batch_results = await self._process_batch_with_cache(batch)
            
            # Store results
            for local_idx, result in enumerate(batch_results):
                global_idx = uncached_indices[batch_start + local_idx]
                results[global_idx] = result
                
                # Cache it
                if self.cache:
                    summary1, summary2 = batch[local_idx]
                    cache_key = f"comparison_v2:{summary1.name}:{summary2.name}"
                    await self.cache.set(cache_key, result)
        
        return results

    async def _process_batch_with_cache(self, batch: List[Tuple]) -> List[Dict]:
        """Process a batch of job pairs with prompt caching."""
        from .job_summarizer import JobSummary
        
        # Build user prompt with all comparisons
        comparisons_text = []
        for i, (sum1, sum2) in enumerate(batch, 1):
            comparisons_text.append(f"""
=== PAIR {i} ===
JOB A: {sum1.to_prompt_text()}

JOB B: {sum2.to_prompt_text()}

Analyze: similarity score (0-100), reusable patterns, migration risks, recommendations.
---""")
        
        user_prompt = "\n".join(comparisons_text)
        
        try:
            # Call Claude with cached system prompt
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=0.1,
                system=[{
                    "type": "text",
                    "text": self.system_prompt_cached,
                    "cache_control": {"type": "ephemeral"}  # Cache this!
                }],
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Log cache efficiency
            usage = message.usage
            cache_read = getattr(usage, 'cache_read_input_tokens', 0)
            cache_write = getattr(usage, 'cache_creation_input_tokens', 0)
            logger.info(f"📊 Tokens - Input: {usage.input_tokens}, "
                       f"Cache Read: {cache_read} (💰 saved!), "
                       f"Cache Write: {cache_write}, "
                       f"Output: {usage.output_tokens}")
            
            # Parse response
            response_text = message.content[0].text
            return self._parse_batch_response(response_text, len(batch))
            
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            return [{"error": str(e), "cached": False} for _ in batch]

    def _parse_batch_response(self, response: str, expected_count: int) -> List[Dict]:
        """Parse Claude's batch response into individual comparison dicts."""
        # Split by pair markers
        pairs = response.split("=== PAIR")
        results = []
        
        for i in range(1, min(len(pairs), expected_count + 1)):
            pair_text = pairs[i].strip()
            
            # Simple parsing - extract key info
            result = {
                "cached": False,
                "similarity_score": self._extract_score(pair_text),
                "reusable_patterns": self._extract_section(pair_text, "pattern"),
                "migration_risks": self._extract_section(pair_text, "risk"),
                "recommendations": self._extract_section(pair_text, "recommend"),
                "raw_analysis": pair_text[:500]  # First 500 chars
            }
            results.append(result)
        
        # Fill missing
        while len(results) < expected_count:
            results.append({"error": "Incomplete response", "cached": False})
        
        return results

    def _extract_score(self, text: str) -> float:
        """Extract similarity score from response text."""
        import re
        match = re.search(r'similarity[:\s]+(\d+)', text.lower())
        if match:
            return float(match.group(1))
        return 50.0  # Default

    def _extract_section(self, text: str, keyword: str) -> str:
        """Extract section containing keyword."""
        lines = text.lower().split('\n')
        relevant = [line for line in lines if keyword in line]
        return ' '.join(relevant[:3]) if relevant else "N/A"

    async def compare_batch(self, job_indices: List[int], jobs: List[Any] = None) -> List[ComparisonResult]:
        """
        Compare all pairs in the batch using Claude AI.

        Args:
            job_indices: Indices of jobs to compare
            jobs: Full list of jobs (needed for structures)

        Returns:
            List of comparison results
        """
        if not jobs:
            raise ValueError("Jobs list required for comparison")

        results = []

        # Compare each pair
        for i in range(len(job_indices)):
            for j in range(i + 1, len(job_indices)):
                idx1, idx2 = job_indices[i], job_indices[j]

                # Check cache first
                cache_key = f"comparison:{idx1}:{idx2}"
                cached_result = await self.cache.get(cache_key) if self.cache else None

                if cached_result:
                    result = ComparisonResult(**cached_result)
                else:
                    result = await self._compare_single(jobs[idx1], jobs[idx2], idx1, idx2)

                    # Cache the result
                    if self.cache:
                        await self.cache.set(cache_key, {
                            'job1_idx': result.job1_idx,
                            'job2_idx': result.job2_idx,
                            'similarity_score': result.similarity_score,
                            'differences': result.differences,
                            'reasoning': result.reasoning
                        })

                results.append(result)

        logger.info(f"Completed {len(results)} comparisons")
        return results

    async def _compare_single(self, job1: Any, job2: Any, idx1: int, idx2: int) -> ComparisonResult:
        """Compare two individual jobs using Claude."""
        # Prepare prompt
        prompt = self.prompt_template.format(
            job1_structure=json.dumps(job1.structure, indent=2),
            job2_structure=json.dumps(job2.structure, indent=2)
        )

        # Call Claude API
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.1,
            system="You are an expert DataStage ETL analyst.",
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        try:
            content = response.content[0].text
            result_data = json.loads(content)

            return ComparisonResult(
                job1_idx=idx1,
                job2_idx=idx2,
                similarity_score=result_data['similarity_score'],
                differences=result_data['differences'],
                reasoning=result_data['reasoning']
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse Claude response: {e}")
            # Return default result
            return ComparisonResult(
                job1_idx=idx1,
                job2_idx=idx2,
                similarity_score=0.5,
                differences=["Failed to parse comparison"],
                reasoning="Error in AI analysis"
            )