"""
Pattern Analysis Module

Analyzes DataStage jobs to identify patterns and migration complexity.
"""

import logging
from typing import Dict, List, Any
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class JobPattern:
    """Represents patterns found in a DataStage job."""
    job_name: str
    source_types: List[str]
    target_types: List[str]
    transformation_types: List[str]
    stage_count: int
    link_count: int
    complexity_score: float
    migration_category: str
    
    
class PatternAnalyzer:
    """Analyzes jobs to identify patterns and migration complexity."""
    
    # Stage type categories for migration
    SOURCE_STAGES = {
        'OracleConnector', 'DB2Connector', 'SQLServerConnector', 'TeradataConnector',
        'SequentialFile', 'FileSet', 'Dataset', 'ODBCConnector', 'JDBCConnector'
    }
    
    TARGET_STAGES = {
        'OracleConnector', 'DB2Connector', 'SQLServerConnector', 'TeradataConnector',
        'SequentialFile', 'Dataset', 'ODBCConnector', 'JDBCConnector'
    }
    
    TRANSFORMATION_STAGES = {
        'Transformer', 'Join', 'Lookup', 'Aggregator', 'Sort', 'Funnel',
        'Remove Duplicates', 'Filter', 'Copy', 'Modify', 'Pivot', 'ChangeCapture',
        'SurrogateKeyGenerator', 'ColumnGenerator', 'Merge', 'Switch'
    }
    
    # AWS Glue equivalence difficulty (1=easy, 5=hard)
    GLUE_COMPLEXITY = {
        # File/S3 Sources - Native Glue support
        'SequentialFile': 1,      # Easy: S3 source with GlueContext
        'FileSet': 1,             # Easy: S3 partitioned dataset
        'Dataset': 1,             # Easy: Glue Data Catalog table

        # Database Connectors - JDBC via Glue Connections
        'OracleConnector': 2,     # Medium: Glue JDBC Connection
        'OracleConnectorPX': 2,   # Medium: Glue JDBC Connection
        'DB2Connector': 2,        # Medium: Glue JDBC Connection
        'DB2ConnectorPX': 2,      # Medium: Glue JDBC Connection
        'SQLServerConnector': 2,  # Medium: Glue JDBC Connection
        'SQLServerConnectorPX': 2,# Medium: Glue JDBC Connection
        'ODBCConnector': 2,       # Medium: Glue JDBC Connection
        'JDBCConnector': 2,       # Medium: Native Glue JDBC
        'TeradataConnector': 4,   # Hard: Custom connector or JDBC
        'TeradataConnectorPX': 4, # Hard: Custom connector or JDBC
        'NetezzaConnector': 3,    # Medium-Hard: JDBC with tuning
        'RedshiftConnector': 1,   # Easy: Native Glue support

        # Transformations - DynamicFrame operations
        'Transformer': 2,         # Medium: ApplyMapping + Map transform
        'Filter': 1,              # Easy: Filter.apply()
        'Join': 2,                # Medium: Join.apply()
        'Lookup': 3,              # Hard: Join with broadcast hint
        'Aggregator': 2,          # Medium: Spark groupBy via DynamicFrame
        'Sort': 1,                # Easy: native orderBy
        'Funnel': 1,              # Easy: Union/unionAll
        'Remove Duplicates': 2,   # Medium: dropDuplicates
        'Copy': 1,                # Easy: select/alias
        'Modify': 2,              # Medium: ApplyMapping
        'Pivot': 3,               # Hard: toDF() + pivot + toDynamicFrame
        'Merge': 2,               # Medium: Union with schema resolution
        'Switch': 3,              # Hard: Multiple filters with routing

        # Complex Patterns - Require custom logic
        'ChangeCapture': 5,       # Very Hard: Glue Bookmarks + Delta/Iceberg
        'ChangeApply': 5,         # Very Hard: SCD Type 2 logic
        'SurrogateKeyGenerator': 2,  # Medium: monotonically_increasing_id
        'ColumnGenerator': 1,     # Easy: withColumn
        'RowGenerator': 2,        # Medium: Custom DataFrame creation

        # Parallel/Advanced Stages
        'Peek': 1,                # Easy: show() / take()
        'Head': 1,                # Easy: limit()
        'Tail': 2,                # Medium: reverse sort + limit
        'Sample': 1,              # Easy: sample()
    }

    # Legacy alias for backward compatibility
    PYSPARK_COMPLEXITY = GLUE_COMPLEXITY

    # AWS Glue specific migration categories
    GLUE_MIGRATION_MAPPING = {
        # Source mappings
        'SequentialFile': 's3_source',
        'FileSet': 's3_partitioned',
        'Dataset': 'catalog_table',
        'OracleConnector': 'jdbc_oracle',
        'OracleConnectorPX': 'jdbc_oracle',
        'DB2Connector': 'jdbc_db2',
        'DB2ConnectorPX': 'jdbc_db2',
        'SQLServerConnector': 'jdbc_sqlserver',
        'SQLServerConnectorPX': 'jdbc_sqlserver',
        'TeradataConnector': 'jdbc_teradata',
        'TeradataConnectorPX': 'jdbc_teradata',
        'RedshiftConnector': 'redshift_native',
        'JDBCConnector': 'jdbc_generic',
        'ODBCConnector': 'jdbc_generic',

        # Transform mappings
        'Transformer': 'apply_mapping',
        'Filter': 'filter_transform',
        'Join': 'join_transform',
        'Lookup': 'join_broadcast',
        'Aggregator': 'group_by',
        'Sort': 'order_by',
        'Funnel': 'union_all',
        'Pivot': 'pivot_transform',
        'Merge': 'union_resolve',
        'ChangeCapture': 'cdc_bookmarks',
        'SurrogateKeyGenerator': 'surrogate_key',
    }

    def analyze_job(self, job) -> JobPattern:
        """Analyze a single job to extract patterns."""
        structure = job.structure
        
        # Extract stage types
        stages = structure.get('stages', [])
        stage_types = [s.get('type', 'Unknown') for s in stages]
        
        # Categorize stages
        sources = [st for st in stage_types if st in self.SOURCE_STAGES]
        targets = [st for st in stage_types if st in self.TARGET_STAGES]
        transforms = [st for st in stage_types if st in self.TRANSFORMATION_STAGES]
        
        # Count links
        links = structure.get('links', [])
        link_count = len(links)
        
        # Calculate complexity score
        complexity = self._calculate_complexity(stages, links)
        
        # Determine migration category
        category = self._categorize_for_migration(sources, targets, transforms, complexity)
        
        return JobPattern(
            job_name=job.name,
            source_types=list(set(sources)),
            target_types=list(set(targets)),
            transformation_types=list(set(transforms)),
            stage_count=len(stages),
            link_count=link_count,
            complexity_score=complexity,
            migration_category=category
        )
    
    def _calculate_complexity(self, stages: List[Dict], links: List[Dict]) -> float:
        """Calculate job complexity score (0-100) for AWS Glue migration."""
        score = 0.0

        # Stage count contribution (0-30 points)
        stage_count = len(stages)
        score += min(stage_count * 2, 30)

        # Stage type complexity (0-40 points) - using Glue complexity mapping
        for stage in stages:
            stage_type = stage.get('type', 'Unknown')
            score += self.GLUE_COMPLEXITY.get(stage_type, 2)

        # Link complexity (0-20 points)
        link_count = len(links)
        score += min(link_count * 1.5, 20)

        # Branching complexity (0-10 points)
        # Count stages with multiple outputs
        from_counts = Counter(link.get('from', '') for link in links)
        branches = sum(1 for count in from_counts.values() if count > 1)
        score += min(branches * 3, 10)

        return min(score, 100)

    def get_glue_migration_type(self, stage_type: str) -> str:
        """Get the AWS Glue migration type for a DataStage stage."""
        return self.GLUE_MIGRATION_MAPPING.get(stage_type, 'custom_transform')
    
    def _categorize_for_migration(self, sources: List[str], targets: List[str],
                                   transforms: List[str], complexity: float) -> str:
        """Categorize job for AWS Glue migration strategy."""

        # Simple file to S3
        if (any('File' in s or 'Dataset' in s for s in sources) and
            any('File' in t or 'Dataset' in t for t in targets) and
            complexity < 20):
            return "AUTO - S3 to S3 ETL"

        # Database read-only (reporting) - perfect for Glue + Athena
        if sources and not targets and complexity < 30:
            return "AUTO - Read-Only (Athena candidate)"

        # Simple DB to DB/S3
        if (any('Connector' in s for s in sources) and
            complexity < 40 and len(transforms) < 3):
            return "AUTO - Simple JDBC ETL"

        # CDC/SCD patterns - requires Glue Bookmarks + Delta/Iceberg
        if any('Change' in t for t in transforms):
            return "MANUAL - CDC/SCD Pattern (Delta Lake)"

        # Complex transformations
        if len(transforms) > 5 or complexity > 70:
            return "MANUAL - Complex Transformations"

        # Default categorization
        if complexity < 30:
            return "AUTO - Basic Glue Job"
        elif complexity < 50:
            return "SEMI-AUTO - Standard ETL"
        elif complexity < 70:
            return "SEMI-AUTO - Advanced ETL"
        else:
            return "MANUAL - Complex Job"

    def get_migration_automation_level(self, category: str) -> str:
        """Extract automation level from category."""
        if category.startswith("AUTO"):
            return "AUTO"
        elif category.startswith("SEMI-AUTO"):
            return "SEMI-AUTO"
        else:
            return "MANUAL"
    
    def generate_migration_report(self, patterns: List[JobPattern]) -> Dict[str, Any]:
        """Generate a migration complexity report."""
        
        # Count by category
        categories = Counter(p.migration_category for p in patterns)
        
        # Average complexity
        avg_complexity = sum(p.complexity_score for p in patterns) / len(patterns) if patterns else 0
        
        # Most common sources/targets/transforms
        all_sources = [s for p in patterns for s in p.source_types]
        all_targets = [t for p in patterns for t in p.target_types]
        all_transforms = [t for p in patterns for t in p.transformation_types]
        
        source_counts = Counter(all_sources)
        target_counts = Counter(all_targets)
        transform_counts = Counter(all_transforms)
        
        # Complexity distribution
        simple_jobs = sum(1 for p in patterns if p.complexity_score < 30)
        medium_jobs = sum(1 for p in patterns if 30 <= p.complexity_score < 60)
        complex_jobs = sum(1 for p in patterns if p.complexity_score >= 60)
        
        return {
            'total_jobs': len(patterns),
            'avg_complexity': round(avg_complexity, 2),
            'categories': dict(categories),
            'complexity_distribution': {
                'simple': simple_jobs,
                'medium': medium_jobs,
                'complex': complex_jobs
            },
            'top_sources': dict(source_counts.most_common(10)),
            'top_targets': dict(target_counts.most_common(10)),
            'top_transforms': dict(transform_counts.most_common(10)),
            'migration_recommendations': self._generate_recommendations(patterns)
        }
    
    def _generate_recommendations(self, patterns: List[JobPattern]) -> List[str]:
        """Generate AWS Glue migration recommendations based on patterns."""
        recommendations = []

        # Count stages types
        all_sources = [s for p in patterns for s in p.source_types]
        all_transforms = [t for p in patterns for t in p.transformation_types]

        # Automation level stats
        auto_count = sum(1 for p in patterns if p.migration_category.startswith("AUTO"))
        semi_count = sum(1 for p in patterns if p.migration_category.startswith("SEMI-AUTO"))
        manual_count = sum(1 for p in patterns if p.migration_category.startswith("MANUAL"))

        recommendations.append(
            f"🤖 Migration automatique: {auto_count} jobs AUTO, {semi_count} SEMI-AUTO, {manual_count} MANUAL"
        )

        # File/S3-heavy recommendation
        file_count = sum(1 for s in all_sources if 'File' in s or 'Dataset' in s)
        if file_count > len(patterns) * 0.5:
            recommendations.append(
                f"📁 {file_count} jobs utilisent des fichiers - Migration S3 native avec Glue DynamicFrames"
            )

        # JDBC-heavy recommendation
        jdbc_count = sum(1 for s in all_sources if 'Connector' in s)
        if jdbc_count > len(patterns) * 0.3:
            recommendations.append(
                f"🔌 {jdbc_count} jobs utilisent JDBC - Configurer Glue Connections dans le Data Catalog"
            )

        # Lookup-heavy recommendation
        lookup_count = sum(1 for t in all_transforms if 'Lookup' in t)
        if lookup_count > len(patterns) * 0.3:
            recommendations.append(
                f"🔍 {lookup_count} jobs utilisent Lookups - Utiliser Join avec broadcast hint dans Glue"
            )

        # CDC/SCD recommendation
        cdc_count = sum(1 for p in patterns if 'CDC' in p.migration_category or 'SCD' in p.migration_category)
        if cdc_count > 0:
            recommendations.append(
                f"⚠️ {cdc_count} jobs CDC/SCD - Utiliser Glue Bookmarks + Delta Lake/Iceberg sur S3"
            )

        # Complexity recommendation
        complex_count = sum(1 for p in patterns if p.complexity_score >= 60)
        if complex_count > len(patterns) * 0.3:
            recommendations.append(
                f"🔥 {complex_count} jobs complexes - Prévoir tests unitaires et validation manuelle"
            )

        # Glue-specific recommendations
        recommendations.append(
            "💡 Recommandation: Utiliser Glue Job Bookmarks pour le traitement incrémental"
        )

        # AWS infrastructure
        teradata_count = sum(1 for s in all_sources if 'Teradata' in s)
        if teradata_count > 0:
            recommendations.append(
                f"🔧 {teradata_count} jobs Teradata - Prévoir custom Glue connector ou JDBC optimisé"
            )

        # Overall estimate
        total_effort_days = sum(p.complexity_score / 10 for p in patterns)
        recommendations.append(
            f"⏱️ Effort estimé de migration: {round(total_effort_days, 1)} jours-développeur"
        )

        return recommendations
