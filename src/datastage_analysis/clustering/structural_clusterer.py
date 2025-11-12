"""
Structural Clustering Module

Clusters jobs based on structural hash fingerprints.
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class StructuralClusterer:
    """Clusters jobs by their structural fingerprints."""

    def cluster_by_hash(self, fingerprints: List[str]) -> Dict[str, List[int]]:
        """
        Group jobs by identical fingerprints.

        Args:
            fingerprints: List of fingerprint strings

        Returns:
            Dict mapping fingerprint to list of job indices
        """
        clusters = defaultdict(list)

        for idx, fingerprint in enumerate(fingerprints):
            clusters[fingerprint].append(idx)

        # Remove single-item clusters if desired, but keep for now
        logger.info(f"Created {len(clusters)} structural clusters from {len(fingerprints)} jobs")

        return dict(clusters)

    def get_cluster_stats(self, clusters: Dict[str, List[int]]) -> Dict[str, Any]:
        """Get statistics about the clusters."""
        cluster_sizes = [len(members) for members in clusters.values()]

        return {
            'total_clusters': len(clusters),
            'total_jobs': sum(cluster_sizes),
            'avg_cluster_size': sum(cluster_sizes) / len(clusters) if clusters else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'singleton_clusters': sum(1 for size in cluster_sizes if size == 1)
        }