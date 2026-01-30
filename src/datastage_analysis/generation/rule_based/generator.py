"""
Rule-Based Generator Implementation

Uses Jinja2 templates to generate AWS Glue code from DataStage patterns.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..generator import GeneratedJob
from ...prediction.migration_predictor import MigrationPrediction, MigrationCategory

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


class RuleBasedGenerator:
    """
    Generates AWS Glue code using rule-based pattern matching.

    Templates are loaded from the templates directory and filled
    with job-specific information.
    """

    def __init__(self):
        """Initialize the rule-based generator."""
        self.template_dir = Path(__file__).parent / 'templates'

        # Try to load Jinja2
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self._has_jinja = True
        except ImportError:
            logger.warning("Jinja2 not installed, using basic templates")
            self._has_jinja = False

    def generate(
        self,
        prediction: MigrationPrediction,
        structure: Dict
    ) -> GeneratedJob:
        """
        Generate Glue code for a job.

        Args:
            prediction: Migration prediction for the job
            structure: Job structure from parser

        Returns:
            GeneratedJob with generated code
        """
        job_name = prediction.job_name
        warnings = []

        # Analyze job structure
        analysis = self._analyze_structure(structure)

        # Generate Glue script
        glue_script = self._generate_glue_script(job_name, structure, analysis)

        # Generate Terraform
        terraform = self._generate_terraform(job_name, prediction, analysis)

        # Generate unit tests
        unit_tests = self._generate_unit_tests(job_name, analysis)

        # Generate documentation
        documentation = self._generate_documentation(job_name, prediction, structure, analysis)

        # Add warnings for unsupported features
        if analysis.get('unknown_stages'):
            warnings.append(f"Unknown stages: {', '.join(analysis['unknown_stages'])}")
        if analysis.get('complex_transforms'):
            warnings.append("Contains complex transforms - manual review recommended")

        return GeneratedJob(
            job_name=job_name,
            category=prediction.category,
            success=True,
            glue_script=glue_script,
            terraform=terraform,
            unit_tests=unit_tests,
            documentation=documentation,
            generator_type="rule_based",
            warnings=warnings,
        )

    def _analyze_structure(self, structure: Dict) -> Dict[str, Any]:
        """
        Analyze job structure to determine generation strategy.

        Returns analysis including:
        - sources: List of source stages
        - targets: List of target stages
        - transforms: List of transformation stages
        - pattern: Overall pattern type
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

                # Categorize by role (simplified heuristic)
                if mapping['type'] in ('jdbc', 's3'):
                    # Determine if source or target based on naming convention
                    name_lower = stage_name.lower()
                    if any(x in name_lower for x in ['src', 'source', 'in', 'read']):
                        sources.append(stage_info)
                    elif any(x in name_lower for x in ['tgt', 'target', 'out', 'write', 'dest']):
                        targets.append(stage_info)
                    else:
                        # Default: first occurrence is source, rest are targets
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
                # Skip metadata stages
                if not stage_type.startswith('C') or stage_type in ('CTransformerStage',):
                    unknown_stages.append(stage_type)

        # Determine pattern
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
        has_transform = len(transforms) > 0
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
        elif has_transform:
            return 'transform_only'
        else:
            return 'simple_copy'

    def _generate_glue_script(
        self,
        job_name: str,
        structure: Dict,
        analysis: Dict
    ) -> str:
        """Generate AWS Glue Python script."""
        pattern = analysis['pattern']

        # Try to use Jinja2 template
        if self._has_jinja:
            try:
                template = self.jinja_env.get_template(f'glue_{pattern}.py.j2')
                return template.render(
                    job_name=job_name,
                    analysis=analysis,
                    structure=structure,
                    generated_at=datetime.now().isoformat(),
                )
            except Exception as e:
                logger.debug(f"Template not found for pattern {pattern}, using default: {e}")

        # Fallback to basic template
        return self._generate_basic_glue_script(job_name, analysis)

    def _generate_basic_glue_script(self, job_name: str, analysis: Dict) -> str:
        """Generate basic Glue script without Jinja2."""
        sources = analysis['sources']
        targets = analysis['targets']
        transforms = analysis['transforms']

        # Build source reading code
        source_code = []
        for i, src in enumerate(sources):
            var_name = f"df_{src['name'].lower().replace(' ', '_')}"
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
        transformation_ctx="write_{tgt['name'].lower()}_ctx"
    )''')
            else:
                target_code.append(f'''
    # Write to S3: {tgt['name']}
    glueContext.write_dynamic_frame.from_options(
        frame=DynamicFrame.fromDF(df_output, glueContext, "output"),
        connection_type="s3",
        connection_options={{"path": args["target_path"]}},
        format="{tgt.get('format', 'parquet')}",
        transformation_ctx="write_{tgt['name'].lower()}_ctx"
    )''')

        # Combine all parts
        script = f'''"""
AWS Glue Job: {job_name}
Generated by DataStage Migration Analyzer (rule-based)
Pattern: {analysis['pattern']}
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
        df_output = {f"df_{sources[0]['name'].lower().replace(' ', '_')}" if sources else "spark.createDataFrame([], schema=None)"}

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
        return script

    def _generate_terraform(
        self,
        job_name: str,
        prediction: MigrationPrediction,
        analysis: Dict
    ) -> str:
        """Generate Terraform configuration for Glue job."""
        safe_name = job_name.lower().replace(' ', '_').replace('-', '_')

        # Determine worker configuration based on complexity
        worker_type = "G.1X"
        num_workers = 2
        if analysis.get('complex_transforms') or len(analysis.get('sources', [])) > 2:
            worker_type = "G.2X"
            num_workers = 4

        return f'''# Terraform configuration for Glue job: {job_name}
# Generated by DataStage Migration Analyzer
# Pattern: {analysis['pattern']}

resource "aws_glue_job" "{safe_name}" {{
  name     = "{job_name}"
  role_arn = var.glue_role_arn

  command {{
    name            = "glueetl"
    script_location = "s3://${{var.scripts_bucket}}/glue_jobs/{safe_name}.py"
    python_version  = "3"
  }}

  default_arguments = {{
    "--job-language"                     = "python"
    "--enable-metrics"                   = "true"
    "--enable-spark-ui"                  = "true"
    "--enable-job-insights"              = "true"
    "--enable-continuous-cloudwatch-log" = "true"
    "--job-bookmark-option"              = "job-bookmark-enable"
    "--TempDir"                          = "s3://${{var.temp_bucket}}/glue-temp/"
    "--source_database"                  = var.source_database
    "--source_path"                      = var.source_path
    "--target_database"                  = var.target_database
    "--target_path"                      = var.target_path
  }}

  worker_type       = "{worker_type}"
  number_of_workers = {num_workers}
  glue_version      = "4.0"
  timeout           = 120

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

# Output the job name for reference
output "{safe_name}_job_name" {{
  value = aws_glue_job.{safe_name}.name
}}
'''

    def _generate_unit_tests(self, job_name: str, analysis: Dict) -> str:
        """Generate unit tests for the Glue job."""
        safe_name = job_name.lower().replace(' ', '_').replace('-', '_')

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


class Test{safe_name.title().replace("_", "")}:
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
        # Example: Test column derivations, filters, aggregations
        result = sample_data.select("id", "name")
        assert result.count() == sample_data.count()

    @pytest.mark.skip(reason="Requires Glue context - run in Glue environment")
    def test_glue_job_integration(self):
        """Integration test for Glue job."""
        # This test requires actual Glue context
        # Run with: pytest --run-integration
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

    def _generate_documentation(
        self,
        job_name: str,
        prediction: MigrationPrediction,
        structure: Dict,
        analysis: Dict
    ) -> str:
        """Generate migration documentation."""
        sources = analysis.get('sources', [])
        targets = analysis.get('targets', [])
        transforms = analysis.get('transforms', [])

        # Build stage mapping table
        stage_rows = []
        for src in sources:
            stage_rows.append(f"| {src['name']} | {src['original_type']} | Source | JDBC/S3 Read |")
        for t in transforms:
            stage_rows.append(f"| {t['name']} | {t['original_type']} | Transform | PySpark |")
        for tgt in targets:
            stage_rows.append(f"| {tgt['name']} | {tgt['original_type']} | Target | JDBC/S3 Write |")

        stage_table = '\n'.join(stage_rows) if stage_rows else "| (No stages detected) | - | - | - |"

        # Build checklist based on prediction
        checklist = []
        checklist.append("- [ ] Review generated Glue script")
        checklist.append("- [ ] Verify connection configurations")
        if analysis.get('has_jdbc'):
            checklist.append("- [ ] Set up JDBC connections in Glue")
            checklist.append("- [ ] Verify database credentials in Secrets Manager")
        if analysis.get('has_s3'):
            checklist.append("- [ ] Verify S3 bucket permissions")
            checklist.append("- [ ] Configure S3 paths in job parameters")
        if transforms:
            checklist.append("- [ ] Review transformation logic")
            checklist.append("- [ ] Test with sample data")
        checklist.append("- [ ] Run unit tests")
        checklist.append("- [ ] Deploy Terraform configuration")
        checklist.append("- [ ] Validate output data")

        return f'''# Migration Documentation: {job_name}

## Overview

| Property | Value |
|----------|-------|
| **Original Job** | {job_name} |
| **Migration Category** | {prediction.category.value} |
| **Confidence** | {prediction.confidence:.0%} |
| **Success Probability** | {prediction.success_probability:.0%} |
| **Estimated Effort** | {prediction.estimated_hours:.1f} hours |
| **Risk Level** | {prediction.risk_level.value} |
| **Pattern** | {analysis['pattern']} |
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

- `glue_jobs/{job_name.lower().replace(' ', '_')}.py` - AWS Glue ETL script
- `terraform/{job_name.lower().replace(' ', '_')}.tf` - Terraform configuration
- `tests/test_{job_name.lower().replace(' ', '_')}.py` - Unit tests

## Notes

{f"**Unknown Stages**: {', '.join(analysis.get('unknown_stages', []))}" if analysis.get('unknown_stages') else "All stages successfully mapped."}

---
*Generated by DataStage Migration Analyzer*
'''
