#!/usr/bin/env python3
"""
DataStage Job Comparison Pipeline

This script implements a 6-phase pipeline for comparing 9000 DataStage jobs:
1. Structural fingerprint extraction
2. Structural clustering by hash
3. Semantic clustering with embeddings
4. Representative selection
5. Fine comparison with Claude AI
6. Interactive report generation
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datastage_analysis.parsers.dsx_parser import DSXParser
from datastage_analysis.clustering.structural_clusterer import StructuralClusterer
from datastage_analysis.embeddings.semantic_embedder import SemanticEmbedder
from datastage_analysis.clustering.semantic_clusterer import SemanticClusterer
from datastage_analysis.api.claude_comparator import ClaudeComparator
from datastage_analysis.report.interactive_report import InteractiveReport
from datastage_analysis.cache.redis_cache import RedisCache
from datastage_analysis.analysis.pattern_analyzer import PatternAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStageAnalysisPipeline:
    def __init__(self, data_dir: Path, output_dir: Path, redis_url: str = "redis://localhost:6379", skip_genai: bool = False, n_semantic_clusters: int = 10):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.skip_genai = skip_genai
        self.cache = RedisCache(redis_url)
        self.parser = DSXParser()
        self.structural_clusterer = StructuralClusterer()
        self.embedder = SemanticEmbedder()
        self.semantic_clusterer = SemanticClusterer(n_clusters=n_semantic_clusters)
        self.pattern_analyzer = PatternAnalyzer()
        if not skip_genai:
            self.comparator = ClaudeComparator(cache=self.cache)
        self.report = InteractiveReport(output_dir)

    async def run_pipeline(self) -> None:
        """Run the complete 6-phase pipeline."""
        logger.info("Starting DataStage analysis pipeline...")

        # Phase 1: Extract structural fingerprints
        logger.info("Phase 1: Extracting structural fingerprints...")
        jobs = await self.parser.parse_all_jobs(self.data_dir)
        fingerprints = [self.parser.extract_fingerprint(job) for job in jobs]

        # Phase 2: Structural clustering
        logger.info("Phase 2: Structural clustering...")
        structural_clusters = self.structural_clusterer.cluster_by_hash(fingerprints)

        # Phase 3: Semantic clustering
        logger.info("Phase 3: Semantic clustering...")
        embeddings = await self.embedder.generate_embeddings(jobs)
        semantic_clusters = self.semantic_clusterer.cluster_embeddings(embeddings)

        # Phase 4: Select representatives
        logger.info("Phase 4: Selecting representatives...")
        representatives = self.semantic_clusterer.select_representatives(semantic_clusters, target_count=1000)

        # Phase 4.5: Pattern analysis for migration
        logger.info("Phase 4.5: Analyzing patterns for migration...")
        patterns = [self.pattern_analyzer.analyze_job(job) for job in jobs]
        migration_report = self.pattern_analyzer.generate_migration_report(patterns)
        
        # Save migration report
        import json
        with open(self.output_dir / "migration_report.json", 'w') as f:
            json.dump(migration_report, f, indent=2)
        
        logger.info(f"Migration Analysis:")
        logger.info(f"  - Average Complexity: {migration_report['avg_complexity']}/100")
        logger.info(f"  - Categories: {migration_report['categories']}")
        for rec in migration_report['migration_recommendations']:
            logger.info(f"  - {rec}")

        # Phase 5: Fine comparison (optional)
        comparisons = []
        if not self.skip_genai:
            logger.info("Phase 5: Fine comparison with Claude AI...")
            comparisons = await self.comparator.compare_batch(representatives)
        else:
            logger.info("Phase 5: Skipping GenAI comparison...")

        # Phase 6: Generate report
        logger.info("Phase 6: Generating interactive report...")
        await self.report.generate_report(jobs, structural_clusters, semantic_clusters, comparisons)

        logger.info("Pipeline completed successfully!")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='DataStage Job Comparison Pipeline')
    parser.add_argument('--skip-genai', action='store_true', help='Skip GenAI comparison phase')
    parser.add_argument('--n-clusters', type=int, default=10, help='Number of semantic clusters (default: 10)')
    args = parser.parse_args()

    data_dir = Path("data")
    output_dir = Path("output")

    # Ensure directories exist
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    pipeline = DataStageAnalysisPipeline(data_dir, output_dir, skip_genai=args.skip_genai, n_semantic_clusters=args.n_clusters)
    await pipeline.run_pipeline()


if __name__ == "__main__":
    asyncio.run(main())