"""
Migration Predictor Module

Predicts migration outcomes and classifies jobs into automation categories.
Uses rule-based scoring with optional ML enhancement.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class MigrationCategory(Enum):
    """Categories for migration automation level."""
    AUTO = "AUTO"           # Fully automatable, generate code directly
    SEMI_AUTO = "SEMI_AUTO" # Needs template + manual adjustments
    MANUAL = "MANUAL"       # Requires significant manual work


class MigrationRisk(Enum):
    """Risk levels for migration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class MigrationPrediction:
    """Prediction result for a single job."""
    job_name: str
    category: MigrationCategory
    confidence: float  # 0.0 to 1.0
    success_probability: float  # 0.0 to 1.0
    estimated_hours: float
    risk_level: MigrationRisk
    risk_factors: List[str]
    recommendations: List[str]
    automation_blockers: List[str]
    glue_features_needed: List[str]
    unknown_stages: List[str] = field(default_factory=list)  # Unrecognized DataStage stages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'job_name': self.job_name,
            'category': self.category.value,
            'confidence': round(self.confidence, 2),
            'success_probability': round(self.success_probability, 2),
            'estimated_hours': round(self.estimated_hours, 1),
            'risk_level': self.risk_level.value,
            'risk_factors': self.risk_factors,
            'recommendations': self.recommendations,
            'automation_blockers': self.automation_blockers,
            'glue_features_needed': self.glue_features_needed,
            'unknown_stages': self.unknown_stages,
        }


@dataclass
class BatchPredictionReport:
    """Summary report for batch predictions."""
    total_jobs: int
    auto_count: int
    semi_auto_count: int
    manual_count: int
    avg_success_probability: float
    total_estimated_hours: float
    high_risk_jobs: List[str]
    critical_risk_jobs: List[str]
    common_blockers: Dict[str, int]
    glue_infrastructure_needs: Dict[str, int]
    unknown_stages: Dict[str, int] = field(default_factory=dict)  # Unknown stage types with counts
    jobs_with_unknown_stages: List[str] = field(default_factory=list)  # Jobs containing unknown stages


class MigrationPredictor:
    """
    Predicts migration outcomes for DataStage jobs to AWS Glue.

    Uses a hybrid approach:
    1. Rule-based scoring for deterministic factors
    2. Complexity-weighted probability calculations
    3. Risk assessment based on known problematic patterns
    """

    # Stage types that are fully supported in Glue (high automation potential)
    FULLY_SUPPORTED_STAGES = {
        'SequentialFile', 'FileSet', 'Dataset',
        'Filter', 'Sort', 'Copy', 'Funnel',
        'JDBCConnector', 'ODBCConnector',
    }

    # Stage types that need some manual configuration
    PARTIALLY_SUPPORTED_STAGES = {
        'Transformer', 'Join', 'Aggregator', 'Lookup',
        'OracleConnector', 'OracleConnectorPX',
        'DB2Connector', 'DB2ConnectorPX',
        'SQLServerConnector', 'SQLServerConnectorPX',
        'Modify', 'ColumnGenerator', 'SurrogateKeyGenerator',
        'Remove Duplicates', 'Merge',
    }

    # Stage types that require significant manual work
    MANUAL_STAGES = {
        'ChangeCapture', 'ChangeApply', 'Pivot', 'Switch',
        'TeradataConnector', 'TeradataConnectorPX',
        'NetezzaConnector', 'CustomStage',
    }

    # Known problematic patterns
    RISK_PATTERNS = {
        'nested_lookup': {'description': 'Multiple nested lookups', 'risk_increase': 0.2},
        'complex_sql': {'description': 'Complex embedded SQL', 'risk_increase': 0.25},
        'cdc_scd': {'description': 'CDC/SCD patterns', 'risk_increase': 0.3},
        'parallel_complex': {'description': 'Complex parallel processing', 'risk_increase': 0.15},
        'custom_routine': {'description': 'Custom routines/procedures', 'risk_increase': 0.35},
        'external_call': {'description': 'External system calls', 'risk_increase': 0.25},
    }

    # Effort multipliers by stage type
    EFFORT_MULTIPLIERS = {
        'SequentialFile': 0.5,
        'Transformer': 2.0,
        'Join': 1.5,
        'Lookup': 2.0,
        'Aggregator': 1.5,
        'ChangeCapture': 4.0,
        'Pivot': 2.5,
        'TeradataConnector': 3.0,
        'default': 1.0,
    }

    def __init__(self, calibration_data: Optional[Dict] = None):
        """
        Initialize the predictor.

        Args:
            calibration_data: Optional historical data for calibration
        """
        self.calibration_data = calibration_data or {}
        self._load_calibration()

    def _load_calibration(self):
        """Load calibration adjustments from historical data."""
        # Default calibration factors (can be adjusted based on actual migration results)
        self.calibration = {
            'effort_factor': 1.0,  # Multiply all effort estimates
            'confidence_adjustment': 0.0,  # Add to confidence scores
            'success_baseline': 0.85,  # Base success probability
        }

        if self.calibration_data:
            # Adjust based on actual historical results
            if 'actual_vs_predicted_effort' in self.calibration_data:
                ratio = self.calibration_data['actual_vs_predicted_effort']
                self.calibration['effort_factor'] = ratio

            if 'actual_success_rate' in self.calibration_data:
                actual = self.calibration_data['actual_success_rate']
                self.calibration['success_baseline'] = actual

    def predict(self, job_pattern: Any, job_structure: Dict) -> MigrationPrediction:
        """
        Predict migration outcome for a single job.

        Args:
            job_pattern: JobPattern from PatternAnalyzer
            job_structure: Raw job structure with stages and links

        Returns:
            MigrationPrediction with category, probability, and recommendations
        """
        stages = job_structure.get('stages', [])
        links = job_structure.get('links', [])

        # Analyze stage composition
        stage_analysis = self._analyze_stages(stages)

        # Detect risk patterns
        risk_factors, risk_score = self._detect_risks(stages, links, job_pattern)

        # Calculate category and confidence
        category, confidence = self._classify_job(
            stage_analysis, risk_score, job_pattern.complexity_score
        )

        # Calculate success probability
        success_prob = self._calculate_success_probability(
            stage_analysis, risk_score, job_pattern.complexity_score
        )

        # Estimate effort
        estimated_hours = self._estimate_effort(
            stages, job_pattern.complexity_score, category
        )

        # Determine risk level
        risk_level = self._determine_risk_level(risk_score, risk_factors)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            stage_analysis, risk_factors, category
        )

        # Identify automation blockers
        blockers = self._identify_blockers(stages, risk_factors)

        # Identify Glue features needed
        glue_features = self._identify_glue_features(stages, job_pattern)

        # Get unknown stages from analysis
        unknown_stages = stage_analysis.get('unknown_stage_types', [])

        # Add warning for unknown stages
        if unknown_stages:
            risk_factors.append(f"Unknown stage types: {', '.join(unknown_stages)}")
            recommendations.append(f"⚠️ Review unknown stages: {', '.join(unknown_stages)}")

        return MigrationPrediction(
            job_name=job_pattern.job_name,
            category=category,
            confidence=confidence,
            success_probability=success_prob,
            estimated_hours=estimated_hours,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations,
            automation_blockers=blockers,
            glue_features_needed=glue_features,
            unknown_stages=unknown_stages,
        )

    def predict_batch(self, patterns: List[Any],
                      structures: Dict[str, Dict]) -> Tuple[List[MigrationPrediction], BatchPredictionReport]:
        """
        Predict migration outcomes for multiple jobs.

        Args:
            patterns: List of JobPattern objects
            structures: Dict mapping job names to structures

        Returns:
            Tuple of (predictions list, summary report)
        """
        predictions = []

        for pattern in patterns:
            structure = structures.get(pattern.job_name, {'stages': [], 'links': []})
            prediction = self.predict(pattern, structure)
            predictions.append(prediction)

        # Generate summary report
        report = self._generate_batch_report(predictions)

        return predictions, report

    def _analyze_stages(self, stages: List[Dict]) -> Dict[str, Any]:
        """Analyze stage composition for classification."""
        stage_types = [s.get('type', 'Unknown') for s in stages]

        fully_supported = sum(1 for t in stage_types if t in self.FULLY_SUPPORTED_STAGES)
        partially_supported = sum(1 for t in stage_types if t in self.PARTIALLY_SUPPORTED_STAGES)
        manual_required = sum(1 for t in stage_types if t in self.MANUAL_STAGES)

        # Identify unknown/unrecognized stage types
        all_known_stages = (
            self.FULLY_SUPPORTED_STAGES |
            self.PARTIALLY_SUPPORTED_STAGES |
            self.MANUAL_STAGES
        )
        unknown_stage_types = [
            t for t in stage_types
            if t not in all_known_stages and t != 'Unknown' and t != ''
        ]
        unknown = len(unknown_stage_types)

        total = len(stage_types) or 1  # Avoid division by zero

        return {
            'total_stages': len(stage_types),
            'fully_supported': fully_supported,
            'partially_supported': partially_supported,
            'manual_required': manual_required,
            'unknown': unknown,
            'unknown_stage_types': list(set(unknown_stage_types)),  # Unique unknown types
            'automation_ratio': (fully_supported + partially_supported * 0.5) / total,
            'stage_types': stage_types,
            'unique_types': list(set(stage_types)),
        }

    def _detect_risks(self, stages: List[Dict], links: List[Dict],
                      pattern: Any) -> Tuple[List[str], float]:
        """Detect risk patterns in the job."""
        risk_factors = []
        risk_score = 0.0

        stage_types = [s.get('type', '') for s in stages]

        # Check for nested lookups
        lookup_count = sum(1 for t in stage_types if 'Lookup' in t)
        if lookup_count > 2:
            risk_factors.append(f"Multiple lookups ({lookup_count}) - complex data enrichment")
            risk_score += self.RISK_PATTERNS['nested_lookup']['risk_increase']

        # Check for CDC/SCD patterns
        if any('Change' in t for t in stage_types):
            risk_factors.append("CDC/SCD pattern detected - requires Glue Bookmarks + custom logic")
            risk_score += self.RISK_PATTERNS['cdc_scd']['risk_increase']

        # Check for complex branching
        from_counts = {}
        for link in links:
            from_stage = link.get('from', '')
            from_counts[from_stage] = from_counts.get(from_stage, 0) + 1

        complex_branches = sum(1 for count in from_counts.values() if count > 2)
        if complex_branches > 0:
            risk_factors.append(f"Complex branching ({complex_branches} multi-output stages)")
            risk_score += self.RISK_PATTERNS['parallel_complex']['risk_increase']

        # Check for problematic connectors
        if any('Teradata' in t for t in stage_types):
            risk_factors.append("Teradata connector - requires custom JDBC configuration")
            risk_score += 0.15

        if any('Netezza' in t for t in stage_types):
            risk_factors.append("Netezza connector - legacy system, verify compatibility")
            risk_score += 0.15

        # Check complexity threshold
        if pattern.complexity_score > 80:
            risk_factors.append(f"High complexity score ({pattern.complexity_score}/100)")
            risk_score += 0.1

        # Check for Pivot operations
        if any('Pivot' in t for t in stage_types):
            risk_factors.append("Pivot transformation - requires DataFrame API")
            risk_score += 0.1

        return risk_factors, min(risk_score, 1.0)

    def _classify_job(self, stage_analysis: Dict, risk_score: float,
                      complexity_score: float) -> Tuple[MigrationCategory, float]:
        """Classify job into migration category."""
        automation_ratio = stage_analysis['automation_ratio']
        manual_required = stage_analysis['manual_required']

        # Decision logic
        if manual_required > 0 or risk_score > 0.4:
            category = MigrationCategory.MANUAL
            # Confidence based on how clear-cut the decision is
            confidence = min(0.95, 0.7 + risk_score * 0.3)

        elif automation_ratio > 0.8 and complexity_score < 40 and risk_score < 0.2:
            category = MigrationCategory.AUTO
            confidence = min(0.95, automation_ratio - risk_score)

        elif automation_ratio > 0.6 and complexity_score < 60:
            category = MigrationCategory.SEMI_AUTO
            confidence = min(0.9, 0.5 + automation_ratio * 0.3 - risk_score * 0.2)

        else:
            # Default to SEMI_AUTO for uncertain cases
            if complexity_score > 70:
                category = MigrationCategory.MANUAL
            else:
                category = MigrationCategory.SEMI_AUTO
            confidence = 0.6

        return category, confidence

    def _calculate_success_probability(self, stage_analysis: Dict,
                                       risk_score: float,
                                       complexity_score: float) -> float:
        """Calculate probability of successful migration."""
        base_prob = self.calibration['success_baseline']

        # Adjust based on automation ratio
        automation_factor = stage_analysis['automation_ratio'] * 0.15

        # Penalize for risk
        risk_penalty = risk_score * 0.3

        # Penalize for complexity
        complexity_penalty = (complexity_score / 100) * 0.2

        # Penalize for unknown stages
        unknown_penalty = (stage_analysis['unknown'] / max(stage_analysis['total_stages'], 1)) * 0.1

        success_prob = base_prob + automation_factor - risk_penalty - complexity_penalty - unknown_penalty

        return max(0.1, min(0.99, success_prob))

    def _estimate_effort(self, stages: List[Dict], complexity_score: float,
                         category: MigrationCategory) -> float:
        """Estimate migration effort in hours."""
        base_hours = 2.0  # Minimum for any job

        # Add hours per stage based on type
        stage_hours = 0.0
        for stage in stages:
            stage_type = stage.get('type', 'Unknown')
            multiplier = self.EFFORT_MULTIPLIERS.get(
                stage_type,
                self.EFFORT_MULTIPLIERS['default']
            )
            stage_hours += multiplier

        # Complexity factor
        complexity_factor = 1 + (complexity_score / 100)

        # Category multiplier
        category_multipliers = {
            MigrationCategory.AUTO: 0.5,
            MigrationCategory.SEMI_AUTO: 1.0,
            MigrationCategory.MANUAL: 2.0,
        }
        category_mult = category_multipliers.get(category, 1.0)

        # Calculate total
        total_hours = (base_hours + stage_hours) * complexity_factor * category_mult

        # Apply calibration
        total_hours *= self.calibration['effort_factor']

        return round(total_hours, 1)

    def _determine_risk_level(self, risk_score: float,
                              risk_factors: List[str]) -> MigrationRisk:
        """Determine overall risk level."""
        # Check for critical patterns
        critical_patterns = ['CDC/SCD', 'custom routine', 'external call']
        has_critical = any(
            pattern.lower() in factor.lower()
            for factor in risk_factors
            for pattern in critical_patterns
        )

        if has_critical or risk_score > 0.6:
            return MigrationRisk.CRITICAL
        elif risk_score > 0.4:
            return MigrationRisk.HIGH
        elif risk_score > 0.2:
            return MigrationRisk.MEDIUM
        else:
            return MigrationRisk.LOW

    def _generate_recommendations(self, stage_analysis: Dict,
                                  risk_factors: List[str],
                                  category: MigrationCategory) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Category-specific recommendations
        if category == MigrationCategory.AUTO:
            recommendations.append("✅ Good candidate for automated migration")
            recommendations.append("Run generated script in Glue Dev Endpoint for validation")

        elif category == MigrationCategory.SEMI_AUTO:
            recommendations.append("📝 Review generated template and complete TODOs")
            recommendations.append("Test with sample data before full migration")

        else:  # MANUAL
            recommendations.append("⚠️ Requires manual analysis and custom implementation")
            recommendations.append("Consider breaking into smaller, simpler jobs")

        # Risk-based recommendations
        if any('CDC' in f for f in risk_factors):
            recommendations.append("💡 Use Glue Bookmarks for incremental processing")
            recommendations.append("Consider Delta Lake or Apache Iceberg for CDC")

        if any('Lookup' in f for f in risk_factors):
            recommendations.append("💡 Use broadcast joins for small lookup tables")
            recommendations.append("Consider caching lookup data in Glue Data Catalog")

        if any('Teradata' in f for f in risk_factors):
            recommendations.append("🔧 Configure Teradata JDBC driver in Glue")
            recommendations.append("Consider migrating Teradata tables to Redshift first")

        # Complexity recommendations
        if stage_analysis['total_stages'] > 10:
            recommendations.append("📊 Complex job - consider modularizing into multiple Glue jobs")

        return recommendations

    def _identify_blockers(self, stages: List[Dict],
                          risk_factors: List[str]) -> List[str]:
        """Identify factors blocking full automation."""
        blockers = []

        stage_types = [s.get('type', '') for s in stages]

        # Stage-based blockers
        for stage_type in set(stage_types):
            if stage_type in self.MANUAL_STAGES:
                blockers.append(f"Stage type '{stage_type}' requires manual implementation")

        # Risk-based blockers
        if any('CDC' in f or 'SCD' in f for f in risk_factors):
            blockers.append("CDC/SCD logic must be manually validated")

        if any('custom' in f.lower() for f in risk_factors):
            blockers.append("Custom code/routines need manual conversion")

        # Check for complex transformers
        transformer_count = sum(1 for t in stage_types if t == 'Transformer')
        if transformer_count > 3:
            blockers.append(f"Multiple Transformers ({transformer_count}) - review SQL/logic")

        return blockers

    def _identify_glue_features(self, stages: List[Dict],
                                pattern: Any) -> List[str]:
        """Identify AWS Glue features needed for migration."""
        features = set()

        stage_types = [s.get('type', '') for s in stages]

        # Always needed
        features.add("Glue ETL Job")
        features.add("IAM Role for Glue")

        # Conditional features
        if any('Connector' in t for t in stage_types):
            features.add("Glue Connection (JDBC)")
            features.add("VPC Configuration")
            features.add("Secrets Manager (credentials)")

        if any('File' in t or 'Dataset' in t for t in stage_types):
            features.add("S3 Bucket")
            features.add("Glue Data Catalog")

        if any('Change' in t for t in stage_types):
            features.add("Job Bookmarks")
            features.add("Delta Lake / Iceberg (recommended)")

        if pattern.complexity_score > 60:
            features.add("Glue Dev Endpoint (testing)")
            features.add("CloudWatch Logs")
            features.add("Spark UI")

        if any('Oracle' in t or 'DB2' in t or 'SQLServer' in t for t in stage_types):
            features.add("JDBC Drivers")

        if any('Teradata' in t for t in stage_types):
            features.add("Teradata JDBC Driver (custom)")
            features.add("Glue Marketplace Connector (optional)")

        return sorted(list(features))

    def _generate_batch_report(self,
                               predictions: List[MigrationPrediction]) -> BatchPredictionReport:
        """Generate summary report for batch predictions."""
        auto_count = sum(1 for p in predictions if p.category == MigrationCategory.AUTO)
        semi_count = sum(1 for p in predictions if p.category == MigrationCategory.SEMI_AUTO)
        manual_count = sum(1 for p in predictions if p.category == MigrationCategory.MANUAL)

        avg_success = sum(p.success_probability for p in predictions) / len(predictions) if predictions else 0
        total_hours = sum(p.estimated_hours for p in predictions)

        high_risk = [p.job_name for p in predictions if p.risk_level == MigrationRisk.HIGH]
        critical_risk = [p.job_name for p in predictions if p.risk_level == MigrationRisk.CRITICAL]

        # Count common blockers
        blocker_counts = {}
        for p in predictions:
            for blocker in p.automation_blockers:
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1

        # Count infrastructure needs
        infra_counts = {}
        for p in predictions:
            for feature in p.glue_features_needed:
                infra_counts[feature] = infra_counts.get(feature, 0) + 1

        # Count unknown/unrecognized stage types
        unknown_stage_counts = {}
        jobs_with_unknown = []
        for p in predictions:
            if p.unknown_stages:
                jobs_with_unknown.append(p.job_name)
                for stage_type in p.unknown_stages:
                    unknown_stage_counts[stage_type] = unknown_stage_counts.get(stage_type, 0) + 1

        return BatchPredictionReport(
            total_jobs=len(predictions),
            auto_count=auto_count,
            semi_auto_count=semi_count,
            manual_count=manual_count,
            avg_success_probability=round(avg_success, 2),
            total_estimated_hours=round(total_hours, 1),
            high_risk_jobs=high_risk,
            critical_risk_jobs=critical_risk,
            common_blockers=dict(sorted(blocker_counts.items(), key=lambda x: -x[1])[:10]),
            glue_infrastructure_needs=dict(sorted(infra_counts.items(), key=lambda x: -x[1])),
            unknown_stages=dict(sorted(unknown_stage_counts.items(), key=lambda x: -x[1])),
            jobs_with_unknown_stages=jobs_with_unknown,
        )

    def calibrate(self, actual_results: List[Dict]):
        """
        Calibrate predictor based on actual migration results.

        Args:
            actual_results: List of dicts with 'job_name', 'actual_hours', 'success' keys
        """
        if not actual_results:
            return

        # Calculate effort ratio
        predicted_hours = []
        actual_hours = []
        successes = []

        for result in actual_results:
            # Re-predict (if we have the patterns cached)
            predicted_hours.append(result.get('predicted_hours', 0))
            actual_hours.append(result.get('actual_hours', 0))
            successes.append(1 if result.get('success', False) else 0)

        if predicted_hours and actual_hours:
            total_predicted = sum(predicted_hours)
            total_actual = sum(actual_hours)
            if total_predicted > 0:
                self.calibration['effort_factor'] = total_actual / total_predicted

        if successes:
            self.calibration['success_baseline'] = sum(successes) / len(successes)

        logger.info(f"Calibration updated: {self.calibration}")

    def export_predictions(self, predictions: List[MigrationPrediction],
                          output_path: str, format: str = 'json'):
        """Export predictions to file."""
        data = [p.to_dict() for p in predictions]

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'csv':
            import csv
            if data:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    for row in data:
                        # Flatten lists to strings
                        flat_row = {
                            k: '; '.join(v) if isinstance(v, list) else v
                            for k, v in row.items()
                        }
                        writer.writerow(flat_row)

        logger.info(f"Exported {len(predictions)} predictions to {output_path}")


class MigrationPriorityRanker:
    """Ranks jobs for migration priority based on multiple factors."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ranker with custom weights.

        Args:
            weights: Dict of factor weights (should sum to 1.0)
        """
        self.weights = weights or {
            'automation_potential': 0.3,  # Higher = better (AUTO jobs first)
            'success_probability': 0.25,  # Higher = better
            'business_impact': 0.2,       # Higher = migrate sooner
            'effort_inverse': 0.15,       # Lower effort = higher priority
            'risk_inverse': 0.1,          # Lower risk = higher priority
        }

    def rank_jobs(self, predictions: List[MigrationPrediction],
                  business_priorities: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        Rank jobs by migration priority.

        Args:
            predictions: List of MigrationPrediction objects
            business_priorities: Optional dict of job_name -> priority score (0-1)

        Returns:
            List of (job_name, priority_score) tuples, sorted descending
        """
        business_priorities = business_priorities or {}
        scores = []

        # Handle empty predictions list
        if not predictions:
            return []

        # Normalize effort values
        max_effort = max((p.estimated_hours for p in predictions), default=1) or 1

        for pred in predictions:
            # Automation potential (AUTO=1.0, SEMI=0.5, MANUAL=0.2)
            auto_scores = {
                MigrationCategory.AUTO: 1.0,
                MigrationCategory.SEMI_AUTO: 0.5,
                MigrationCategory.MANUAL: 0.2,
            }
            auto_score = auto_scores.get(pred.category, 0.3)

            # Success probability (already 0-1)
            success_score = pred.success_probability

            # Business impact (default to 0.5 if not provided)
            business_score = business_priorities.get(pred.job_name, 0.5)

            # Effort inverse (lower effort = higher score)
            effort_score = 1 - (pred.estimated_hours / max_effort)

            # Risk inverse
            risk_scores = {
                MigrationRisk.LOW: 1.0,
                MigrationRisk.MEDIUM: 0.7,
                MigrationRisk.HIGH: 0.4,
                MigrationRisk.CRITICAL: 0.1,
            }
            risk_score = risk_scores.get(pred.risk_level, 0.5)

            # Calculate weighted score
            priority = (
                self.weights['automation_potential'] * auto_score +
                self.weights['success_probability'] * success_score +
                self.weights['business_impact'] * business_score +
                self.weights['effort_inverse'] * effort_score +
                self.weights['risk_inverse'] * risk_score
            )

            scores.append((pred.job_name, round(priority, 3)))

        # Sort by priority descending
        return sorted(scores, key=lambda x: -x[1])

    def create_migration_waves(self, ranked_jobs: List[Tuple[str, float]],
                               wave_size: int = 50) -> List[List[str]]:
        """
        Group ranked jobs into migration waves.

        Args:
            ranked_jobs: Sorted list of (job_name, priority) tuples
            wave_size: Maximum jobs per wave

        Returns:
            List of waves, each containing job names
        """
        waves = []
        current_wave = []

        for job_name, _ in ranked_jobs:
            current_wave.append(job_name)
            if len(current_wave) >= wave_size:
                waves.append(current_wave)
                current_wave = []

        if current_wave:
            waves.append(current_wave)

        return waves
