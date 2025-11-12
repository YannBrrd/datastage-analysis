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
    
    # PySpark equivalence difficulty (1=easy, 5=hard)
    PYSPARK_COMPLEXITY = {
        'SequentialFile': 1,  # Easy: spark.read.csv()
        'Transformer': 2,     # Medium: .withColumn()
        'Filter': 1,          # Easy: .filter()
        'Join': 2,            # Medium: .join()
        'Aggregator': 2,      # Medium: .groupBy().agg()
        'Sort': 1,            # Easy: .orderBy()
        'Lookup': 3,          # Hard: broadcast join
        'Pivot': 3,           # Hard: .pivot()
        'ChangeCapture': 4,   # Very Hard: complex logic
        'SurrogateKeyGenerator': 3,  # Hard: window functions
        'OracleConnector': 2,  # Medium: JDBC
        'TeradataConnector': 3,  # Hard: specific connector
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
        """Calculate job complexity score (0-100)."""
        score = 0.0
        
        # Stage count contribution (0-30 points)
        stage_count = len(stages)
        score += min(stage_count * 2, 30)
        
        # Stage type complexity (0-40 points)
        for stage in stages:
            stage_type = stage.get('type', 'Unknown')
            score += self.PYSPARK_COMPLEXITY.get(stage_type, 2)
        
        # Link complexity (0-20 points)
        link_count = len(links)
        score += min(link_count * 1.5, 20)
        
        # Branching complexity (0-10 points)
        # Count stages with multiple outputs
        from_counts = Counter(link.get('from', '') for link in links)
        branches = sum(1 for count in from_counts.values() if count > 1)
        score += min(branches * 3, 10)
        
        return min(score, 100)
    
    def _categorize_for_migration(self, sources: List[str], targets: List[str], 
                                   transforms: List[str], complexity: float) -> str:
        """Categorize job for migration strategy."""
        
        # Simple file to file
        if (any('File' in s for s in sources) and 
            any('File' in t for t in targets) and 
            complexity < 20):
            return "Simple - File ETL"
        
        # Database read-only (reporting)
        if sources and not targets and complexity < 30:
            return "Medium - Read-Only Report"
        
        # Simple DB to DB
        if (any('Connector' in s for s in sources) and 
            any('Connector' in t for t in targets) and 
            complexity < 40 and len(transforms) < 3):
            return "Medium - Simple DB ETL"
        
        # Complex transformations
        if len(transforms) > 5 or complexity > 60:
            return "Hard - Complex Transformations"
        
        # CDC/SCD patterns
        if any('Change' in t for t in transforms):
            return "Very Hard - CDC/SCD Pattern"
        
        # Default
        if complexity < 30:
            return "Simple - Basic Job"
        elif complexity < 60:
            return "Medium - Standard ETL"
        else:
            return "Hard - Advanced ETL"
    
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
        """Generate migration recommendations based on patterns."""
        recommendations = []
        
        # Count stages types
        all_sources = [s for p in patterns for s in p.source_types]
        all_transforms = [t for p in patterns for t in p.transformation_types]
        
        # File-heavy recommendation
        file_count = sum(1 for s in all_sources if 'File' in s)
        if file_count > len(patterns) * 0.5:
            recommendations.append(
                f"📁 {file_count} jobs use file sources - PySpark native file I/O will work well"
            )
        
        # Lookup-heavy recommendation
        lookup_count = sum(1 for t in all_transforms if 'Lookup' in t)
        if lookup_count > len(patterns) * 0.3:
            recommendations.append(
                f"🔍 {lookup_count} jobs use Lookups - Consider broadcast joins in PySpark"
            )
        
        # CDC/SCD recommendation
        cdc_count = sum(1 for p in patterns if 'CDC' in p.migration_category or 'SCD' in p.migration_category)
        if cdc_count > 0:
            recommendations.append(
                f"⚠️ {cdc_count} jobs use CDC/SCD - Complex migration, consider Delta Lake"
            )
        
        # Complexity recommendation
        complex_count = sum(1 for p in patterns if p.complexity_score >= 60)
        if complex_count > len(patterns) * 0.3:
            recommendations.append(
                f"🔥 {complex_count} complex jobs - Plan for detailed testing and refactoring"
            )
        
        # Overall estimate
        total_effort_days = sum(p.complexity_score / 10 for p in patterns)
        recommendations.append(
            f"⏱️ Estimated migration effort: {round(total_effort_days, 1)} developer-days"
        )
        
        return recommendations
