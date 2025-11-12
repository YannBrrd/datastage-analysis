"""
Semantic Clustering Module

Clusters jobs based on semantic embeddings using scikit-learn.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class SemanticClusterer:
    """Clusters jobs based on semantic embeddings."""

    def __init__(self, n_clusters: int = 100):
        self.n_clusters = n_clusters

    def cluster_embeddings(self, embeddings: np.ndarray) -> Dict[int, List[int]]:
        """
        Cluster embeddings using K-means.

        Args:
            embeddings: Numpy array of embeddings

        Returns:
            Dict mapping cluster ID to list of job indices
        """
        n_samples = len(embeddings)
        # Adjust n_clusters to be at most n_samples
        effective_n_clusters = min(self.n_clusters, n_samples)
        
        logger.info(f"Clustering {n_samples} embeddings into {effective_n_clusters} clusters")

        # Use K-means clustering
        kmeans = KMeans(n_clusters=effective_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Group by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Calculate silhouette score for quality assessment
        if n_samples > effective_n_clusters:
            silhouette = silhouette_score(embeddings, cluster_labels)
            logger.info(f"Silhouette score: {silhouette:.3f}")

        return clusters

    def select_representatives(self, clusters: Dict[int, List[int]], target_count: int = 1000) -> List[int]:
        """
        Select representative jobs from clusters.

        Args:
            clusters: Dict mapping cluster ID to list of job indices
            target_count: Target number of representatives

        Returns:
            List of job indices to use as representatives
        """
        representatives = []

        # Calculate how many to select from each cluster
        total_jobs = sum(len(members) for members in clusters.values())
        cluster_sizes = {cid: len(members) for cid, members in clusters.items()}

        for cluster_id, members in clusters.items():
            # Proportional selection
            proportion = len(members) / total_jobs
            count = max(1, int(proportion * target_count))

            # Select first 'count' members as representatives
            # In practice, might want to select most central ones
            representatives.extend(members[:count])

        # Trim to target count if exceeded
        if len(representatives) > target_count:
            representatives = representatives[:target_count]

        logger.info(f"Selected {len(representatives)} representatives from {len(clusters)} clusters")

        return representatives