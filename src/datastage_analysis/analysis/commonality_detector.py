"""
Commonality Detector Module

Detects duplicate and near-duplicate DataStage jobs to reduce migration effort.
Groups jobs by structural patterns and identifies parameterization opportunities.
"""

import logging
from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """A group of exact duplicate jobs."""
    fingerprint: str
    job_names: List[str]
    stage_pattern: List[str]  # Stage type sequence
    representative_job: str  # First job as template

    @property
    def count(self) -> int:
        return len(self.job_names)


@dataclass
class SimilarityCluster:
    """A cluster of similar (near-duplicate) jobs."""
    cluster_id: int
    job_names: List[str]
    similarity_score: float  # Average pairwise similarity
    pattern_signature: str  # Simplified pattern description
    parameterizable_fields: List[str]  # Fields that vary (table names, paths, etc.)

    @property
    def count(self) -> int:
        return len(self.job_names)


@dataclass
class PatternFamily:
    """A family of jobs following the same structural pattern."""
    pattern_id: str
    pattern_name: str  # Human-readable pattern name
    stage_sequence: List[str]  # Ordered stage types
    job_names: List[str]
    migration_template: str  # Suggested Glue template type
    effort_reduction: float  # Estimated effort reduction vs individual migration

    @property
    def count(self) -> int:
        return len(self.job_names)


@dataclass
class CommonalityReport:
    """Complete commonality analysis report."""
    total_jobs: int
    unique_patterns: int
    exact_duplicate_groups: List[DuplicateGroup]
    similarity_clusters: List[SimilarityCluster]
    pattern_families: List[PatternFamily]
    effective_unique_jobs: int  # Jobs after deduplication
    effort_reduction_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_jobs': self.total_jobs,
            'unique_patterns': self.unique_patterns,
            'exact_duplicate_groups': len(self.exact_duplicate_groups),
            'exact_duplicate_jobs': sum(g.count for g in self.exact_duplicate_groups),
            'similarity_clusters': len(self.similarity_clusters),
            'similar_jobs': sum(c.count for c in self.similarity_clusters),
            'pattern_families': len(self.pattern_families),
            'effective_unique_jobs': self.effective_unique_jobs,
            'effort_reduction_percent': round(self.effort_reduction_percent, 1),
            'top_patterns': [
                {
                    'name': p.pattern_name,
                    'count': p.count,
                    'template': p.migration_template
                }
                for p in sorted(self.pattern_families, key=lambda x: -x.count)[:10]
            ]
        }


class CommonalityDetector:
    """Detects duplicate and similar jobs for migration optimization."""

    # Similarity threshold for near-duplicates
    SIMILARITY_THRESHOLD = 0.85

    # Stage types to ignore when computing similarity (metadata stages)
    IGNORE_STAGES = {
        'CContainerView', 'CJobDefn', 'CAnnotation',
        'CTrxInput', 'CTrxOutput', 'CCustomInput', 'CCustomOutput',
        'CSeqInput', 'CSeqOutput', 'CHashedInput', 'CHashedOutput',
        'CJSJobActivity', 'CJSActivityInput', 'CJSActivityOutput',
        'CJSUserVarsActivity', 'CJSCondition', 'CJSSequencer',
    }

    # Pattern name mappings based on stage combinations
    PATTERN_NAMES = {
        ('source', 'transform', 'target'): 'ETL Pipeline',
        ('source', 'target'): 'Direct Copy',
        ('source', 'lookup', 'transform', 'target'): 'Lookup Enrichment',
        ('source', 'aggregate', 'target'): 'Aggregation Pipeline',
        ('source', 'join', 'transform', 'target'): 'Join Pipeline',
        ('source', 'transform', 'multiple_targets'): 'Fan-out Pipeline',
        ('multiple_sources', 'transform', 'target'): 'Fan-in Pipeline',
    }

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def analyze(self, jobs: Dict[str, Dict], fingerprints: Dict[str, str] = None) -> CommonalityReport:
        """
        Analyze jobs for commonalities and duplicates.

        Args:
            jobs: Dict of job_name -> job_structure
            fingerprints: Optional dict of job_name -> fingerprint (if pre-computed)

        Returns:
            CommonalityReport with analysis results
        """
        if not jobs:
            return CommonalityReport(
                total_jobs=0,
                unique_patterns=0,
                exact_duplicate_groups=[],
                similarity_clusters=[],
                pattern_families=[],
                effective_unique_jobs=0,
                effort_reduction_percent=0.0
            )

        total_jobs = len(jobs)
        logger.info(f"Analyzing commonalities in {total_jobs} jobs")

        # Phase 1: Extract stage sequences for each job
        job_sequences = {}
        for job_name, structure in jobs.items():
            sequence = self._extract_stage_sequence(structure)
            job_sequences[job_name] = sequence

        # Phase 2: Compute fingerprints if not provided
        if fingerprints is None:
            fingerprints = {}
            for job_name, sequence in job_sequences.items():
                fingerprints[job_name] = self._compute_sequence_fingerprint(sequence)

        # Phase 3: Find exact duplicates
        exact_groups = self._find_exact_duplicates(job_sequences, fingerprints)

        # Phase 4: Find similar jobs (near-duplicates)
        # Only compare unique patterns to avoid O(n^2) on all jobs
        unique_patterns = self._get_unique_patterns(job_sequences, fingerprints)
        similarity_clusters = self._find_similar_clusters(unique_patterns, job_sequences, fingerprints)

        # Phase 5: Identify pattern families
        pattern_families = self._identify_pattern_families(job_sequences)

        # Calculate effective unique jobs and effort reduction
        # Jobs in exact duplicate groups count as 1
        jobs_in_dup_groups = sum(g.count - 1 for g in exact_groups)  # -1 because we keep one

        # Jobs in similarity clusters (excluding overlaps with exact dups)
        clustered_jobs = set()
        for cluster in similarity_clusters:
            clustered_jobs.update(cluster.job_names)

        # Effective unique = total - duplicates saved - similar jobs saved
        effective_unique = total_jobs - jobs_in_dup_groups

        # Further reduction from similarity clusters (estimate 80% reduction per cluster)
        cluster_reduction = sum(
            max(0, c.count - 1) * 0.8
            for c in similarity_clusters
        )

        effective_unique = max(1, effective_unique - int(cluster_reduction))
        effort_reduction = ((total_jobs - effective_unique) / total_jobs) * 100 if total_jobs > 0 else 0

        return CommonalityReport(
            total_jobs=total_jobs,
            unique_patterns=len(unique_patterns),
            exact_duplicate_groups=exact_groups,
            similarity_clusters=similarity_clusters,
            pattern_families=pattern_families,
            effective_unique_jobs=effective_unique,
            effort_reduction_percent=effort_reduction
        )

    def _extract_stage_sequence(self, structure: Dict) -> List[str]:
        """Extract ordered stage type sequence from job structure."""
        stages = structure.get('stages', [])

        # Filter out metadata stages and extract types
        sequence = []
        for stage in stages:
            stage_type = stage.get('type', 'Unknown')
            if stage_type not in self.IGNORE_STAGES and stage_type:
                sequence.append(stage_type)

        return sequence

    def _compute_sequence_fingerprint(self, sequence: List[str]) -> str:
        """Compute a fingerprint for a stage sequence."""
        # Sort for order-independent matching, then hash
        sorted_seq = sorted(sequence)
        seq_str = '|'.join(sorted_seq)
        return hashlib.md5(seq_str.encode()).hexdigest()[:16]

    def _find_exact_duplicates(self, job_sequences: Dict[str, List[str]],
                                fingerprints: Dict[str, str]) -> List[DuplicateGroup]:
        """Find groups of jobs with identical structure."""
        # Group jobs by fingerprint
        fingerprint_groups: Dict[str, List[str]] = defaultdict(list)
        for job_name, fp in fingerprints.items():
            fingerprint_groups[fp].append(job_name)

        # Create DuplicateGroup for groups with more than one job
        groups = []
        for fp, job_names in fingerprint_groups.items():
            if len(job_names) > 1:
                # Get stage pattern from first job
                first_job = job_names[0]
                stage_pattern = job_sequences.get(first_job, [])

                groups.append(DuplicateGroup(
                    fingerprint=fp,
                    job_names=sorted(job_names),
                    stage_pattern=stage_pattern,
                    representative_job=first_job
                ))

        # Sort by group size descending
        return sorted(groups, key=lambda g: -g.count)

    def _get_unique_patterns(self, job_sequences: Dict[str, List[str]],
                             fingerprints: Dict[str, str]) -> Dict[str, str]:
        """Get one representative job per unique fingerprint."""
        seen_fps = set()
        unique = {}

        for job_name, fp in fingerprints.items():
            if fp not in seen_fps:
                seen_fps.add(fp)
                unique[job_name] = fp

        return unique

    def _compute_jaccard_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Compute Jaccard similarity between two stage sequences."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        set1 = set(seq1)
        set2 = set(seq2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _compute_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """
        Compute similarity between two stage sequences.
        Combines Jaccard similarity with length similarity.
        """
        jaccard = self._compute_jaccard_similarity(seq1, seq2)

        # Length similarity
        len1, len2 = len(seq1), len(seq2)
        if len1 == 0 and len2 == 0:
            length_sim = 1.0
        else:
            length_sim = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

        # Order similarity (for ordered sequences)
        order_sim = self._compute_order_similarity(seq1, seq2)

        # Weighted combination
        return 0.5 * jaccard + 0.3 * length_sim + 0.2 * order_sim

    def _compute_order_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Compute similarity based on stage ordering using LCS."""
        if not seq1 or not seq2:
            return 0.0

        # Longest common subsequence length
        m, n = len(seq1), len(seq2)

        # Use simple LCS for small sequences, approximate for large
        if m * n > 10000:
            # Approximate for large sequences
            common = set(seq1) & set(seq2)
            return len(common) / max(m, n)

        # Dynamic programming LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]
        return (2 * lcs_length) / (m + n)

    def _find_similar_clusters(self, unique_patterns: Dict[str, str],
                               job_sequences: Dict[str, List[str]],
                               fingerprints: Dict[str, str]) -> List[SimilarityCluster]:
        """Find clusters of similar jobs using greedy clustering."""
        if len(unique_patterns) < 2:
            return []

        # Get sequences for unique patterns only
        pattern_jobs = list(unique_patterns.keys())

        # Build similarity matrix for unique patterns
        n = len(pattern_jobs)
        similarities: Dict[Tuple[str, str], float] = {}

        for i in range(n):
            for j in range(i + 1, n):
                job1, job2 = pattern_jobs[i], pattern_jobs[j]
                sim = self._compute_sequence_similarity(
                    job_sequences[job1],
                    job_sequences[job2]
                )
                if sim >= self.similarity_threshold:
                    similarities[(job1, job2)] = sim

        # Greedy clustering
        clusters = []
        clustered = set()
        cluster_id = 0

        # Sort by similarity descending
        sorted_pairs = sorted(similarities.items(), key=lambda x: -x[1])

        for (job1, job2), sim in sorted_pairs:
            if job1 in clustered and job2 in clustered:
                continue

            # Start new cluster or extend existing
            cluster_jobs = {job1, job2}

            # Add all jobs with same fingerprint
            fp1, fp2 = unique_patterns[job1], unique_patterns[job2]
            for job_name, fp in fingerprints.items():
                if fp == fp1 or fp == fp2:
                    cluster_jobs.add(job_name)

            # Find other similar patterns to add
            for other_job in pattern_jobs:
                if other_job in cluster_jobs or other_job in clustered:
                    continue

                # Check similarity with cluster members
                min_sim = min(
                    self._compute_sequence_similarity(
                        job_sequences[other_job],
                        job_sequences[cj]
                    )
                    for cj in [job1, job2]
                )

                if min_sim >= self.similarity_threshold:
                    cluster_jobs.add(other_job)
                    # Add all jobs with same fingerprint
                    other_fp = unique_patterns[other_job]
                    for jn, fp in fingerprints.items():
                        if fp == other_fp:
                            cluster_jobs.add(jn)

            if len(cluster_jobs) > 1:
                clustered.update([job1, job2])

                # Determine pattern signature
                sample_seq = job_sequences[job1]
                pattern_sig = self._generate_pattern_signature(sample_seq)

                clusters.append(SimilarityCluster(
                    cluster_id=cluster_id,
                    job_names=sorted(cluster_jobs),
                    similarity_score=sim,
                    pattern_signature=pattern_sig,
                    parameterizable_fields=['table_name', 'file_path', 'connection']
                ))
                cluster_id += 1

        return sorted(clusters, key=lambda c: -c.count)

    def _generate_pattern_signature(self, sequence: List[str]) -> str:
        """Generate a human-readable pattern signature."""
        if not sequence:
            return "Empty"

        # Simplify to key stage types
        simplified = []
        for stage in sequence:
            if 'Transformer' in stage or 'Trx' in stage:
                simplified.append('Transform')
            elif 'Seq' in stage or 'File' in stage:
                simplified.append('File')
            elif 'Hash' in stage:
                simplified.append('Lookup')
            elif 'Oracle' in stage or 'DB2' in stage or 'SQL' in stage or 'ODBC' in stage:
                simplified.append('Database')
            elif 'Custom' in stage:
                simplified.append('Custom')
            else:
                simplified.append(stage)

        # Remove consecutive duplicates
        deduped = []
        for s in simplified:
            if not deduped or deduped[-1] != s:
                deduped.append(s)

        return ' → '.join(deduped[:5])  # Limit length

    def _identify_pattern_families(self, job_sequences: Dict[str, List[str]]) -> List[PatternFamily]:
        """Identify high-level pattern families."""
        # Classify each job by pattern type
        pattern_jobs: Dict[str, List[str]] = defaultdict(list)

        for job_name, sequence in job_sequences.items():
            pattern_type = self._classify_pattern(sequence)
            pattern_jobs[pattern_type].append(job_name)

        # Create pattern families
        families = []
        for pattern_type, jobs in pattern_jobs.items():
            if len(jobs) >= 1:  # Include all patterns
                # Get representative sequence
                sample_job = jobs[0]
                sample_seq = job_sequences[sample_job]

                families.append(PatternFamily(
                    pattern_id=hashlib.md5(pattern_type.encode()).hexdigest()[:8],
                    pattern_name=pattern_type,
                    stage_sequence=sample_seq[:10],  # Limit for readability
                    job_names=jobs,
                    migration_template=self._suggest_glue_template(pattern_type),
                    effort_reduction=self._estimate_effort_reduction(len(jobs))
                ))

        return sorted(families, key=lambda f: -f.count)

    def _classify_pattern(self, sequence: List[str]) -> str:
        """Classify a job into a pattern category."""
        if not sequence:
            return "Empty Job"

        has_transform = any('Transformer' in s or 'Trx' in s for s in sequence)
        has_file = any('Seq' in s or 'File' in s for s in sequence)
        has_db = any('Oracle' in s or 'DB2' in s or 'SQL' in s or 'ODBC' in s or 'JDBC' in s for s in sequence)
        has_lookup = any('Hash' in s or 'Lookup' in s for s in sequence)
        has_custom = any('Custom' in s for s in sequence)
        has_join = any('Join' in s for s in sequence)
        has_aggregate = any('Aggregat' in s for s in sequence)

        # Classify based on composition
        if has_custom:
            return "Custom Processing"
        elif has_db and has_file and has_transform:
            return "DB to File ETL"
        elif has_file and has_db and has_transform:
            return "File to DB ETL"
        elif has_db and has_transform:
            return "Database ETL"
        elif has_file and has_transform:
            return "File Processing"
        elif has_lookup and has_transform:
            return "Lookup Enrichment"
        elif has_join:
            return "Join Pipeline"
        elif has_aggregate:
            return "Aggregation Pipeline"
        elif has_transform:
            return "Transformation Only"
        elif has_file:
            return "File Copy"
        elif has_db:
            return "Database Copy"
        else:
            return "Other"

    def _suggest_glue_template(self, pattern_type: str) -> str:
        """Suggest appropriate Glue template for pattern."""
        templates = {
            "DB to File ETL": "jdbc_to_s3_etl",
            "File to DB ETL": "s3_to_jdbc_etl",
            "Database ETL": "jdbc_transform",
            "File Processing": "s3_to_s3_etl",
            "Lookup Enrichment": "join_lookup_etl",
            "Join Pipeline": "join_lookup_etl",
            "Aggregation Pipeline": "aggregation_etl",
            "Transformation Only": "s3_transform",
            "File Copy": "s3_copy",
            "Database Copy": "jdbc_copy",
            "Custom Processing": "custom_etl",
            "Empty Job": "skip",
            "Other": "custom_etl",
        }
        return templates.get(pattern_type, "custom_etl")

    def _estimate_effort_reduction(self, job_count: int) -> float:
        """Estimate effort reduction percentage for a pattern family."""
        if job_count <= 1:
            return 0.0

        # First job = full effort, subsequent jobs = parameterization effort (~10-20%)
        base_effort = job_count  # If done individually
        reduced_effort = 1 + (job_count - 1) * 0.15  # Template + parameterization

        return ((base_effort - reduced_effort) / base_effort) * 100
