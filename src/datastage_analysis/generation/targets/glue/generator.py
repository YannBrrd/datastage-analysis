"""
AWS Glue Target Generator

Implements the BaseTargetGenerator interface for AWS Glue.
Generates PySpark-based Glue ETL scripts with Terraform infrastructure.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ..base import BaseTargetGenerator, GeneratedOutput, TargetConfig
from .config import GlueTargetConfig
from ....prediction.migration_predictor import MigrationPrediction, MigrationCategory

logger = logging.getLogger(__name__)


# Stage type mappings to Glue equivalents
STAGE_MAPPINGS = {
    # Database connectors
    'OracleConnectorPX': {'type': 'jdbc', 'driver': 'oracle'},
    'OracleODBC': {'type': 'jdbc', 'driver': 'oracle'},
    'DB2ConnectorPX': {'type': 'jdbc', 'driver': 'db2'},
    'TeradataConnectorPX': {'type': 'jdbc', 'driver': 'teradata'},
    'SQLServerConnectorPX': {'type': 'jdbc', 'driver': 'sqlserver'},
    'PostgreSQLConnectorPX': {'type': 'jdbc', 'driver': 'postgresql'},
    'ODBCConnectorPX': {'type': 'jdbc', 'driver': 'generic'},
    'DSDB2PX': {'type': 'jdbc', 'driver': 'db2'},

    # File stages
    'PxSequentialFile': {'type': 's3', 'format': 'csv'},
    'SequentialFile': {'type': 's3', 'format': 'csv'},
    'PxDataSet': {'type': 's3', 'format': 'parquet'},
    'FileConnectorPX': {'type': 's3', 'format': 'csv'},
    'ParquetFileConnectorPX': {'type': 's3', 'format': 'parquet'},

    # Transform stages
    'CTransformerStage': {'type': 'transform'},
    'PxTransform': {'type': 'transform'},
    'Transformer': {'type': 'transform'},

    # Lookup/Join stages
    'PxLookup': {'type': 'lookup'},
    'PxJoin': {'type': 'join'},
    'PxMerge': {'type': 'join'},

    # Aggregation
    'PxAggregator': {'type': 'aggregate'},
    'Aggregator': {'type': 'aggregate'},

    # Filter/Sort
    'PxFilter': {'type': 'filter'},
    'PxSort': {'type': 'sort'},
    'PxRemoveDuplicates': {'type': 'distinct'},

    # Funnel/Copy
    'PxFunnel': {'type': 'union'},
    'PxCopy': {'type': 'copy'},
}


class GlueTargetGenerator(BaseTargetGenerator):
    """
    AWS Glue target generator.

    Generates PySpark-based AWS Glue ETL scripts from DataStage jobs.
    Also generates Terraform infrastructure code for Glue job deployment.
    """

    TARGET_NAME = "glue"
    TARGET_DISPLAY_NAME = "AWS Glue"
    MAIN_SCRIPT_EXTENSION = ".py"
    INFRASTRUCTURE_EXTENSION = ".tf"
    INFRASTRUCTURE_TYPE = "terraform"

    def __init__(self, config: Optional[TargetConfig] = None):
        """
        Initialize the Glue target generator.

        Args:
            config: Optional GlueTargetConfig (or TargetConfig)
        """
        if config is None:
            config = GlueTargetConfig()
        elif not isinstance(config, GlueTargetConfig):
            # Convert base TargetConfig to GlueTargetConfig
            config = GlueTargetConfig.from_dict(config.__dict__)

        super().__init__(config)
        self.glue_config: GlueTargetConfig = config

    def generate(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> GeneratedOutput:
        """
        Generate AWS Glue code for a DataStage job.

        Args:
            prediction: Migration prediction with category and analysis
            structure: Parsed job structure from DSX parser
            **kwargs: Additional arguments

        Returns:
            GeneratedOutput with Glue script, Terraform, tests, and docs
        """
        job_name = prediction.job_name
        warnings = []

        # Analyze job structure
        analysis = self._analyze_structure(structure)

        # Generate main script
        main_script, script_warnings = self.generate_main_script(
            prediction, structure, analysis=analysis
        )
        warnings.extend(script_warnings)

        # Generate infrastructure (Terraform)
        infrastructure = None
        if self.config.generate_infrastructure:
            infrastructure = self.generate_infrastructure(
                prediction, structure, analysis=analysis
            )

        # Generate tests
        unit_tests = None
        if self.config.generate_tests:
            unit_tests = self.generate_tests(
                prediction, structure, main_script, analysis=analysis
            )

        # Generate documentation
        documentation = None
        if self.config.generate_docs:
            documentation = self.generate_documentation(
                prediction, structure, main_script, analysis=analysis
            )

        return GeneratedOutput(
            job_name=job_name,
            category=prediction.category,
            success=True,
            target=self.TARGET_NAME,
            main_script=main_script,
            main_script_extension=self.MAIN_SCRIPT_EXTENSION,
            infrastructure=infrastructure,
            infrastructure_extension=self.INFRASTRUCTURE_EXTENSION,
            infrastructure_type=self.INFRASTRUCTURE_TYPE,
            unit_tests=unit_tests,
            documentation=documentation,
            generator_type="rule_based",
            warnings=warnings,
            target_metadata={
                'pattern': analysis.get('pattern', 'unknown'),
                'glue_version': self.glue_config.glue_version,
                'worker_type': self.glue_config.default_worker_type,
            },
        )

    def generate_main_script(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> Tuple[str, List[str]]:
        """
        Generate the AWS Glue PySpark script.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            **kwargs: May contain 'analysis' from generate()

        Returns:
            Tuple of (script_code, warnings_list)
        """
        warnings = []
        analysis = kwargs.get('analysis') or self._analyze_structure(structure)
        job_name = prediction.job_name

        # Check for unknown stages
        if analysis.get('unknown_stages'):
            warnings.append(f"Unknown stages: {', '.join(analysis['unknown_stages'])}")
        if analysis.get('complex_transforms'):
            warnings.append("Contains complex transforms - manual review recommended")

        script = self._build_glue_script(job_name, analysis)
        return script, warnings

    def generate_infrastructure(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> Optional[str]:
        """
        Generate Terraform configuration for the Glue job.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            **kwargs: May contain 'analysis'

        Returns:
            Terraform configuration string
        """
        analysis = kwargs.get('analysis') or self._analyze_structure(structure)
        job_name = prediction.job_name
        safe_name = self._safe_filename(job_name)

        # Determine worker configuration based on complexity
        complexity_score = 0.5
        if hasattr(prediction, 'complexity_score'):
            complexity_score = prediction.complexity_score / 10.0
        elif analysis.get('complex_transforms'):
            complexity_score = 0.7

        worker_config = self.glue_config.get_worker_config(complexity_score)

        return self._build_terraform(
            job_name=job_name,
            safe_name=safe_name,
            prediction=prediction,
            analysis=analysis,
            worker_config=worker_config,
        )

    def generate_tests(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        main_script: str,
        **kwargs
    ) -> Optional[str]:
        """
        Generate pytest-based unit tests for the Glue job.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            main_script: The generated Glue script
            **kwargs: Additional arguments

        Returns:
            Unit test code
        """
        job_name = prediction.job_name
        safe_name = self._safe_filename(job_name)

        return self._build_unit_tests(job_name, safe_name)

    def generate_documentation(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        main_script: str,
        **kwargs
    ) -> Optional[str]:
        """
        Generate markdown documentation for the migration.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            main_script: The generated Glue script
            **kwargs: May contain 'analysis'

        Returns:
            Markdown documentation
        """
        analysis = kwargs.get('analysis') or self._analyze_structure(structure)
        job_name = prediction.job_name

        return self._build_documentation(job_name, prediction, analysis)

    def supports_batch_generation(self) -> bool:
        """Glue target supports batch generation."""
        return True

    def _analyze_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze job structure for Glue generation.

        Returns analysis including sources, targets, transforms, and pattern.
        """
        stages = structure.get('stages', [])

        sources = []
        targets = []
        transforms = []
        lookups = []
        unknown_stages = []

        for stage in stages:
            stage_type = stage.get('type', 'Unknown')
            stage_name = stage.get('name', 'unnamed')

            mapping = STAGE_MAPPINGS.get(stage_type)

            if mapping:
                stage_info = {
                    'name': stage_name,
                    'original_type': stage_type,
                    **mapping,
                    'properties': stage.get('properties', {}),
                }

                # Categorize by role
                if mapping['type'] in ('jdbc', 's3'):
                    name_lower = stage_name.lower()
                    if any(x in name_lower for x in ['src', 'source', 'in', 'read']):
                        sources.append(stage_info)
                    elif any(x in name_lower for x in ['tgt', 'target', 'out', 'write', 'dest']):
                        targets.append(stage_info)
                    else:
                        if not sources:
                            sources.append(stage_info)
                        else:
                            targets.append(stage_info)

                elif mapping['type'] == 'transform':
                    transforms.append(stage_info)
                elif mapping['type'] in ('lookup', 'join'):
                    lookups.append(stage_info)
                else:
                    transforms.append(stage_info)
            else:
                if not stage_type.startswith('C') or stage_type in ('CTransformerStage',):
                    unknown_stages.append(stage_type)

        pattern = self._determine_pattern(sources, targets, transforms, lookups)

        return {
            'sources': sources,
            'targets': targets,
            'transforms': transforms,
            'lookups': lookups,
            'unknown_stages': list(set(unknown_stages)),
            'pattern': pattern,
            'has_jdbc': any(s.get('type') == 'jdbc' for s in sources + targets),
            'has_s3': any(s.get('type') == 's3' for s in sources + targets),
            'complex_transforms': len(transforms) > 2,
        }

    def _determine_pattern(
        self,
        sources: List[Dict],
        targets: List[Dict],
        transforms: List[Dict],
        lookups: List[Dict]
    ) -> str:
        """Determine the job pattern type."""
        has_jdbc_source = any(s.get('type') == 'jdbc' for s in sources)
        has_jdbc_target = any(t.get('type') == 'jdbc' for t in targets)
        has_s3_source = any(s.get('type') == 's3' for s in sources)
        has_s3_target = any(t.get('type') == 's3' for t in targets)
        has_lookup = len(lookups) > 0

        if has_jdbc_source and has_s3_target:
            return 'jdbc_to_s3'
        elif has_s3_source and has_jdbc_target:
            return 's3_to_jdbc'
        elif has_jdbc_source and has_jdbc_target:
            return 'jdbc_to_jdbc'
        elif has_s3_source and has_s3_target:
            return 's3_to_s3'
        elif has_lookup:
            return 'lookup_transform'
        elif transforms:
            return 'transform_only'
        else:
            return 'simple_copy'

    def _build_glue_script(self, job_name: str, analysis: Dict[str, Any]) -> str:
        """Build the AWS Glue PySpark script."""
        sources = analysis.get('sources', [])
        targets = analysis.get('targets', [])
        transforms = analysis.get('transforms', [])

        # Build source reading code
        source_code = []
        for src in sources:
            var_name = f"df_{self._safe_filename(src['name'])}"
            if src.get('type') == 'jdbc':
                source_code.append(f'''
    # Read from {src['original_type']}: {src['name']}
    {var_name} = glueContext.create_dynamic_frame.from_catalog(
        database=args["source_database"],
        table_name="{src['name'].lower()}",
        transformation_ctx="{var_name}_ctx"
    ).toDF()''')
            else:
                source_code.append(f'''
    # Read from S3: {src['name']}
    {var_name} = glueContext.create_dynamic_frame.from_options(
        connection_type="s3",
        connection_options={{"paths": [args["source_path"]]}},
        format="{src.get('format', 'csv')}",
        transformation_ctx="{var_name}_ctx"
    ).toDF()''')

        # Build transform code
        transform_code = []
        if transforms:
            transform_code.append('''
    # Apply transformations
    # TODO: Review and customize transformation logic''')
            for t in transforms:
                transform_code.append(f'    # Transform: {t["name"]} ({t["original_type"]})')

        # Build target writing code
        target_code = []
        for tgt in targets:
            if tgt.get('type') == 'jdbc':
                target_code.append(f'''
    # Write to {tgt['original_type']}: {tgt['name']}
    glueContext.write_dynamic_frame.from_catalog(
        frame=DynamicFrame.fromDF(df_output, glueContext, "output"),
        database=args["target_database"],
        table_name="{tgt['name'].lower()}",
        transformation_ctx="write_{self._safe_filename(tgt['name'])}_ctx"
    )''')
            else:
                target_code.append(f'''
    # Write to S3: {tgt['name']}
    glueContext.write_dynamic_frame.from_options(
        frame=DynamicFrame.fromDF(df_output, glueContext, "output"),
        connection_type="s3",
        connection_options={{"path": args["target_path"]}},
        format="{tgt.get('format', 'parquet')}",
        transformation_ctx="write_{self._safe_filename(tgt['name'])}_ctx"
    )''')

        # Determine output variable
        if sources:
            first_source_var = f"df_{self._safe_filename(sources[0]['name'])}"
        else:
            first_source_var = "spark.createDataFrame([], schema=None)"

        return f'''"""
AWS Glue Job: {job_name}
Generated by DataStage Migration Analyzer
Target: AWS Glue {self.glue_config.glue_version}
Pattern: {analysis.get('pattern', 'unknown')}
Generated at: {datetime.now().isoformat()}

Sources: {', '.join(s['name'] for s in sources) or 'None detected'}
Targets: {', '.join(t['name'] for t in targets) or 'None detected'}
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import *


def main():
    # Initialize Glue context
    args = getResolvedOptions(sys.argv, [
        "JOB_NAME",
        "source_database",
        "source_path",
        "target_database",
        "target_path",
    ])

    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args["JOB_NAME"], args)

    try:
        # ========================================
        # Source Reading
        # ========================================
{"".join(source_code) if source_code else "        # No sources detected - add source reading logic"}

        # ========================================
        # Transformations
        # ========================================
{"".join(transform_code) if transform_code else "        # No transformations detected"}

        # Placeholder: combine sources if multiple
        df_output = {first_source_var}

        # ========================================
        # Target Writing
        # ========================================
{"".join(target_code) if target_code else "        # No targets detected - add target writing logic"}

        job.commit()

    except Exception as e:
        print(f"Job failed with error: {{e}}")
        raise


if __name__ == "__main__":
    main()
'''

    def _build_terraform(
        self,
        job_name: str,
        safe_name: str,
        prediction: MigrationPrediction,
        analysis: Dict[str, Any],
        worker_config: Dict[str, Any],
    ) -> str:
        """Build Terraform configuration for the Glue job."""
        config = self.glue_config

        bookmark_option = "job-bookmark-enable" if config.enable_bookmarks else "job-bookmark-disable"

        return f'''# Terraform configuration for Glue job: {job_name}
# Generated by DataStage Migration Analyzer
# Target: AWS Glue
# Pattern: {analysis.get('pattern', 'unknown')}

resource "aws_glue_job" "{safe_name}" {{
  name     = "{job_name}"
  role_arn = var.glue_role_arn

  command {{
    name            = "glueetl"
    script_location = "s3://${{{config.scripts_bucket_var}}}/glue_jobs/{safe_name}.py"
    python_version  = "{config.python_version}"
  }}

  default_arguments = {{
    "--job-language"                     = "python"
    "--enable-metrics"                   = "{str(config.enable_metrics).lower()}"
    "--enable-spark-ui"                  = "{str(config.enable_spark_ui).lower()}"
    "--enable-job-insights"              = "true"
    "--enable-continuous-cloudwatch-log" = "{str(config.enable_continuous_logging).lower()}"
    "--job-bookmark-option"              = "{bookmark_option}"
    "--TempDir"                          = "s3://${{{config.temp_bucket_var}}}/glue-temp/"
    "--source_database"                  = var.source_database
    "--source_path"                      = var.source_path
    "--target_database"                  = var.target_database
    "--target_path"                      = var.target_path
  }}

  worker_type       = "{worker_config['worker_type']}"
  number_of_workers = {worker_config['number_of_workers']}
  glue_version      = "{config.glue_version}"
  timeout           = {config.default_timeout}

  execution_property {{
    max_concurrent_runs = 1
  }}

  tags = {{
    Source      = "DataStage"
    OriginalJob = "{job_name}"
    MigratedBy  = "datastage-migration-analyzer"
    Category    = "{prediction.category.value}"
  }}
}}

output "{safe_name}_job_name" {{
  value = aws_glue_job.{safe_name}.name
}}
'''

    def _build_unit_tests(self, job_name: str, safe_name: str) -> str:
        """Build pytest unit tests for the Glue job."""
        class_name = ''.join(word.title() for word in safe_name.split('_'))

        return f'''"""
Unit tests for Glue job: {job_name}
Generated by DataStage Migration Analyzer
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return SparkSession.builder \\
        .master("local[1]") \\
        .appName("test_{safe_name}") \\
        .getOrCreate()


@pytest.fixture
def sample_data(spark):
    """Create sample test data."""
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), True),
        StructField("value", IntegerType(), True),
    ])

    data = [
        (1, "test1", 100),
        (2, "test2", 200),
        (3, "test3", 300),
    ]

    return spark.createDataFrame(data, schema)


class Test{class_name}:
    """Test cases for {job_name}."""

    def test_data_not_empty(self, sample_data):
        """Verify test data is not empty."""
        assert sample_data.count() > 0

    def test_schema_matches(self, sample_data):
        """Verify schema has expected columns."""
        columns = sample_data.columns
        assert "id" in columns
        assert "name" in columns

    def test_no_null_ids(self, sample_data):
        """Verify no NULL values in ID column."""
        null_count = sample_data.filter(sample_data.id.isNull()).count()
        assert null_count == 0

    def test_transformation_logic(self, sample_data):
        """Test transformation logic."""
        # TODO: Add specific transformation tests
        result = sample_data.select("id", "name")
        assert result.count() == sample_data.count()

    @pytest.mark.skip(reason="Requires Glue context - run in Glue environment")
    def test_glue_job_integration(self):
        """Integration test for Glue job."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

    def _build_documentation(
        self,
        job_name: str,
        prediction: MigrationPrediction,
        analysis: Dict[str, Any],
    ) -> str:
        """Build markdown documentation for the migration."""
        sources = analysis.get('sources', [])
        targets = analysis.get('targets', [])
        transforms = analysis.get('transforms', [])

        # Build stage mapping table
        stage_rows = []
        for src in sources:
            stage_rows.append(f"| {src['name']} | {src['original_type']} | Source | Glue JDBC/S3 Read |")
        for t in transforms:
            stage_rows.append(f"| {t['name']} | {t['original_type']} | Transform | PySpark |")
        for tgt in targets:
            stage_rows.append(f"| {tgt['name']} | {tgt['original_type']} | Target | Glue JDBC/S3 Write |")

        stage_table = '\n'.join(stage_rows) if stage_rows else "| (No stages detected) | - | - | - |"

        # Build checklist
        checklist = [
            "- [ ] Review generated Glue script",
            "- [ ] Verify connection configurations",
        ]
        if analysis.get('has_jdbc'):
            checklist.extend([
                "- [ ] Set up JDBC connections in Glue",
                "- [ ] Verify database credentials in Secrets Manager",
            ])
        if analysis.get('has_s3'):
            checklist.extend([
                "- [ ] Verify S3 bucket permissions",
                "- [ ] Configure S3 paths in job parameters",
            ])
        if transforms:
            checklist.extend([
                "- [ ] Review transformation logic",
                "- [ ] Test with sample data",
            ])
        checklist.extend([
            "- [ ] Run unit tests",
            "- [ ] Deploy Terraform configuration",
            "- [ ] Validate output data",
        ])

        safe_name = self._safe_filename(job_name)

        return f'''# Migration Documentation: {job_name}

## Overview

| Property | Value |
|----------|-------|
| **Original Job** | {job_name} |
| **Target Platform** | AWS Glue {self.glue_config.glue_version} |
| **Migration Category** | {prediction.category.value} |
| **Confidence** | {prediction.confidence:.0%} |
| **Success Probability** | {prediction.success_probability:.0%} |
| **Estimated Effort** | {prediction.estimated_hours:.1f} hours |
| **Risk Level** | {prediction.risk_level.value} |
| **Pattern** | {analysis.get('pattern', 'unknown')} |
| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |

## Stage Mapping

| DataStage Stage | Type | Role | Glue Equivalent |
|-----------------|------|------|-----------------|
{stage_table}

## Risk Factors

{chr(10).join(f"- {rf}" for rf in prediction.risk_factors) if prediction.risk_factors else "- No significant risk factors identified"}

## Automation Blockers

{chr(10).join(f"- {ab}" for ab in prediction.automation_blockers) if prediction.automation_blockers else "- No automation blockers identified"}

## Glue Features Needed

{chr(10).join(f"- {gf}" for gf in prediction.glue_features_needed) if prediction.glue_features_needed else "- Standard Glue features"}

## Recommendations

{chr(10).join(f"- {rec}" for rec in prediction.recommendations) if prediction.recommendations else "- Follow standard migration procedure"}

## Migration Checklist

{chr(10).join(checklist)}

## Generated Files

- `glue/{safe_name}.py` - AWS Glue ETL script
- `infrastructure/{safe_name}.tf` - Terraform configuration
- `tests/test_{safe_name}.py` - Unit tests

## Notes

{f"**Unknown Stages**: {', '.join(analysis.get('unknown_stages', []))}" if analysis.get('unknown_stages') else "All stages successfully mapped to Glue equivalents."}

---
*Generated by DataStage Migration Analyzer - AWS Glue Target*
'''
