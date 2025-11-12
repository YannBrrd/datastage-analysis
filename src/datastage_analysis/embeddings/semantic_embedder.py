"""
Semantic Embeddings Module

Generates semantic embeddings for jobs using sentence-transformers.
"""

import logging
from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticEmbedder:
    """Generates semantic embeddings for DataStage jobs."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    async def generate_embeddings(self, jobs: List[Any]) -> np.ndarray:
        """
        Generate embeddings for a list of jobs.

        Args:
            jobs: List of DataStageJob objects

        Returns:
            Numpy array of embeddings
        """
        # Extract text representations from jobs
        texts = [self._job_to_text(job) for job in jobs]

        logger.info(f"Generating embeddings for {len(texts)} jobs")

        # Generate embeddings (this is CPU intensive, but sentence-transformers handles it)
        embeddings = self.model.encode(texts, show_progress_bar=True)

        return embeddings

    def _job_to_text(self, job: Any) -> str:
        """Convert a job to a text representation for embedding."""
        parts = []

        # Job name
        parts.append(f"Job: {job.name}")

        # Stages
        for stage in job.structure.get('stages', []):
            stage_text = f"Stage {stage['name']} of type {stage['type']}"
            parts.append(stage_text)

            # Include key properties
            props = stage.get('properties', {})
            for key, value in props.items():
                if isinstance(value, str) and len(value) < 100:  # Avoid very long values
                    parts.append(f"{key}: {value}")

        # Links
        for link in job.structure.get('links', []):
            parts.append(f"Link from {link['from']} to {link['to']}")

        return ' '.join(parts)