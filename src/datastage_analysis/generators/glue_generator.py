"""
AWS Glue Job Generator

Generates AWS Glue ETL scripts from analyzed DataStage job patterns.
Supports automatic code generation for common ETL patterns.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class GlueJobType(Enum):
    """Types of AWS Glue jobs."""
    SPARK = "glueetl"
    PYTHON_SHELL = "pythonshell"
    RAY = "glueray"


class GlueWorkerType(Enum):
    """AWS Glue worker types."""
    STANDARD = "Standard"
    G_1X = "G.1X"
    G_2X = "G.2X"
    G_4X = "G.4X"
    G_8X = "G.8X"
    Z_2X = "Z.2X"


@dataclass
class GlueConnectionConfig:
    """Configuration for a Glue Connection."""
    name: str
    connection_type: str  # JDBC, NETWORK, KAFKA, etc.
    jdbc_url: Optional[str] = None
    username: Optional[str] = None
    password_secret_arn: Optional[str] = None
    subnet_id: Optional[str] = None
    security_groups: List[str] = field(default_factory=list)
    additional_options: Dict[str, str] = field(default_factory=dict)


@dataclass
class GlueJobConfig:
    """Configuration for a generated Glue job."""
    job_name: str
    description: str
    script_location: str
    job_type: GlueJobType = GlueJobType.SPARK
    glue_version: str = "4.0"
    worker_type: GlueWorkerType = GlueWorkerType.G_1X
    number_of_workers: int = 2
    timeout_minutes: int = 60
    max_retries: int = 1
    connections: List[str] = field(default_factory=list)
    default_arguments: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    enable_bookmarks: bool = True
    enable_metrics: bool = True
    enable_spark_ui: bool = True


@dataclass
class GeneratedGlueJob:
    """Result of Glue job generation."""
    job_name: str
    script_content: str
    job_config: GlueJobConfig
    terraform_config: str
    connections_needed: List[GlueConnectionConfig]
    estimated_dpu_hours: float
    migration_notes: List[str]


class GlueScriptGenerator:
    """
    Generates AWS Glue ETL scripts from DataStage job patterns.

    Supports:
    - S3 to S3 transformations
    - JDBC source/target operations
    - Common transformations (Join, Filter, Aggregate, etc.)
    - CDC patterns with Glue Bookmarks
    """

    # Template fragments for common operations
    IMPORTS_TEMPLATE = '''import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
'''

    INIT_TEMPLATE = '''
# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME'{additional_args}])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger = glueContext.get_logger()
logger.info(f"Starting job: {{args['JOB_NAME']}}")
'''

    COMMIT_TEMPLATE = '''
# Commit job (enables bookmarks for incremental processing)
job.commit()
logger.info("Job completed successfully")
'''

    def __init__(self):
        self.source_generators = {
            's3_source': self._generate_s3_source,
            's3_partitioned': self._generate_s3_partitioned_source,
            'catalog_table': self._generate_catalog_source,
            'jdbc_oracle': self._generate_jdbc_source,
            'jdbc_db2': self._generate_jdbc_source,
            'jdbc_sqlserver': self._generate_jdbc_source,
            'jdbc_teradata': self._generate_jdbc_source,
            'jdbc_generic': self._generate_jdbc_source,
            'redshift_native': self._generate_redshift_source,
        }

        self.transform_generators = {
            'apply_mapping': self._generate_apply_mapping,
            'filter_transform': self._generate_filter,
            'join_transform': self._generate_join,
            'join_broadcast': self._generate_broadcast_join,
            'group_by': self._generate_aggregate,
            'order_by': self._generate_sort,
            'union_all': self._generate_union,
            'pivot_transform': self._generate_pivot,
            'surrogate_key': self._generate_surrogate_key,
            'cdc_bookmarks': self._generate_cdc_logic,
        }

        self.target_generators = {
            's3_target': self._generate_s3_target,
            'catalog_target': self._generate_catalog_target,
            'jdbc_target': self._generate_jdbc_target,
            'redshift_target': self._generate_redshift_target,
        }

    def generate_from_pattern(self, job_pattern: Any, job_structure: Dict) -> GeneratedGlueJob:
        """
        Generate a complete Glue job from a DataStage job pattern.

        Args:
            job_pattern: JobPattern from PatternAnalyzer
            job_structure: Raw job structure with stages and links

        Returns:
            GeneratedGlueJob with script, config, and metadata
        """
        job_name = self._sanitize_job_name(job_pattern.job_name)
        migration_notes = []
        connections_needed = []

        # Analyze stages and determine generation strategy
        stages = job_structure.get('stages', [])
        links = job_structure.get('links', [])

        # Build execution graph
        execution_order = self._build_execution_order(stages, links)

        # Generate script sections
        script_parts = [self.IMPORTS_TEMPLATE]

        # Determine additional arguments needed
        additional_args = self._determine_additional_args(stages)
        script_parts.append(self.INIT_TEMPLATE.format(
            additional_args=additional_args
        ))

        # Generate source reads
        source_code, source_notes, source_connections = self._generate_sources(
            stages, job_pattern.source_types
        )
        script_parts.append(source_code)
        migration_notes.extend(source_notes)
        connections_needed.extend(source_connections)

        # Generate transformations
        transform_code, transform_notes = self._generate_transforms(
            stages, execution_order, job_pattern.transformation_types
        )
        script_parts.append(transform_code)
        migration_notes.extend(transform_notes)

        # Generate target writes
        target_code, target_notes, target_connections = self._generate_targets(
            stages, job_pattern.target_types
        )
        script_parts.append(target_code)
        migration_notes.extend(target_notes)
        connections_needed.extend(target_connections)

        # Add commit
        script_parts.append(self.COMMIT_TEMPLATE)

        # Combine script
        script_content = '\n'.join(script_parts)

        # Generate job configuration
        job_config = self._create_job_config(
            job_name, job_pattern, connections_needed
        )

        # Generate Terraform configuration
        terraform_config = self._generate_terraform(job_config, connections_needed)

        # Estimate DPU hours
        estimated_dpu = self._estimate_dpu_hours(job_pattern)

        return GeneratedGlueJob(
            job_name=job_name,
            script_content=script_content,
            job_config=job_config,
            terraform_config=terraform_config,
            connections_needed=connections_needed,
            estimated_dpu_hours=estimated_dpu,
            migration_notes=migration_notes
        )

    def _sanitize_job_name(self, name: str) -> str:
        """Convert DataStage job name to valid Glue job name."""
        # Replace invalid characters
        sanitized = name.replace(' ', '_').replace('-', '_')
        # Ensure it starts with a letter
        if sanitized[0].isdigit():
            sanitized = 'job_' + sanitized
        # Lowercase for consistency
        return sanitized.lower()[:255]

    def _build_execution_order(self, stages: List[Dict], links: List[Dict]) -> List[str]:
        """Build topological order of stage execution."""
        # Build adjacency list
        graph = {s.get('name', ''): [] for s in stages}
        in_degree = {s.get('name', ''): 0 for s in stages}

        for link in links:
            from_stage = link.get('from', '')
            to_stage = link.get('to', '')
            if from_stage in graph and to_stage in graph:
                graph[from_stage].append(to_stage)
                in_degree[to_stage] += 1

        # Kahn's algorithm for topological sort
        queue = [s for s, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbor in graph.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def _determine_additional_args(self, stages: List[Dict]) -> str:
        """Determine additional job arguments needed."""
        args = []

        # Check for JDBC connections
        for stage in stages:
            stage_type = stage.get('type', '')
            if 'Connector' in stage_type and 'File' not in stage_type:
                args.append('DATABASE_CONNECTION')
                break

        # Check for S3 paths
        for stage in stages:
            stage_type = stage.get('type', '')
            if 'File' in stage_type or 'Dataset' in stage_type:
                args.append('S3_INPUT_PATH')
                args.append('S3_OUTPUT_PATH')
                break

        if args:
            return ", '" + "', '".join(set(args)) + "'"
        return ""

    def _generate_sources(self, stages: List[Dict], source_types: List[str]) -> tuple:
        """Generate source reading code."""
        code_parts = ["\n# ========== SOURCE READS =========="]
        notes = []
        connections = []

        source_stages = [s for s in stages if self._is_source_stage(s)]

        for i, stage in enumerate(source_stages):
            stage_name = stage.get('name', f'source_{i}')
            stage_type = stage.get('type', 'Unknown')
            var_name = self._to_variable_name(stage_name)

            if 'File' in stage_type or 'Dataset' in stage_type:
                code, note = self._generate_s3_source(stage, var_name)
            elif 'Oracle' in stage_type:
                code, note, conn = self._generate_jdbc_source(stage, var_name, 'oracle')
                connections.append(conn)
            elif 'DB2' in stage_type:
                code, note, conn = self._generate_jdbc_source(stage, var_name, 'db2')
                connections.append(conn)
            elif 'SQLServer' in stage_type:
                code, note, conn = self._generate_jdbc_source(stage, var_name, 'sqlserver')
                connections.append(conn)
            elif 'Teradata' in stage_type:
                code, note, conn = self._generate_jdbc_source(stage, var_name, 'teradata')
                connections.append(conn)
                notes.append(f"⚠️ {stage_name}: Teradata requires custom JDBC driver in Glue")
            elif 'Redshift' in stage_type:
                code, note = self._generate_redshift_source(stage, var_name)
            else:
                code = f"\n# TODO: Implement source for {stage_type}\n{var_name}_df = None"
                note = f"⚠️ {stage_name}: Unknown source type {stage_type} - manual implementation needed"

            code_parts.append(code)
            if note:
                notes.append(note)

        return '\n'.join(code_parts), notes, connections

    def _generate_transforms(self, stages: List[Dict], execution_order: List[str],
                            transform_types: List[str]) -> tuple:
        """Generate transformation code."""
        code_parts = ["\n# ========== TRANSFORMATIONS =========="]
        notes = []

        transform_stages = [s for s in stages if self._is_transform_stage(s)]

        for stage in transform_stages:
            stage_name = stage.get('name', 'transform')
            stage_type = stage.get('type', 'Unknown')
            var_name = self._to_variable_name(stage_name)

            if stage_type == 'Transformer':
                code, note = self._generate_apply_mapping(stage, var_name)
            elif stage_type == 'Filter':
                code, note = self._generate_filter(stage, var_name)
            elif stage_type == 'Join':
                code, note = self._generate_join(stage, var_name)
            elif stage_type == 'Lookup':
                code, note = self._generate_broadcast_join(stage, var_name)
            elif stage_type == 'Aggregator':
                code, note = self._generate_aggregate(stage, var_name)
            elif stage_type == 'Sort':
                code, note = self._generate_sort(stage, var_name)
            elif stage_type == 'Funnel':
                code, note = self._generate_union(stage, var_name)
            elif stage_type == 'Pivot':
                code, note = self._generate_pivot(stage, var_name)
            elif 'Change' in stage_type:
                code, note = self._generate_cdc_logic(stage, var_name)
                notes.append(f"⚠️ {stage_name}: CDC logic requires manual review")
            elif stage_type == 'SurrogateKeyGenerator':
                code, note = self._generate_surrogate_key(stage, var_name)
            elif stage_type in ('Copy', 'Modify', 'ColumnGenerator'):
                code, note = self._generate_apply_mapping(stage, var_name)
            else:
                code = f"\n# TODO: Implement transform for {stage_type}\n{var_name}_df = input_df"
                note = f"⚠️ {stage_name}: Transform type {stage_type} needs manual implementation"

            code_parts.append(code)
            if note:
                notes.append(note)

        return '\n'.join(code_parts), notes

    def _generate_targets(self, stages: List[Dict], target_types: List[str]) -> tuple:
        """Generate target writing code."""
        code_parts = ["\n# ========== TARGET WRITES =========="]
        notes = []
        connections = []

        target_stages = [s for s in stages if self._is_target_stage(s)]

        for i, stage in enumerate(target_stages):
            stage_name = stage.get('name', f'target_{i}')
            stage_type = stage.get('type', 'Unknown')

            if 'File' in stage_type or 'Dataset' in stage_type:
                code, note = self._generate_s3_target(stage, stage_name)
            elif 'Connector' in stage_type:
                code, note = self._generate_jdbc_target(stage, stage_name)
            elif 'Redshift' in stage_type:
                code, note = self._generate_redshift_target(stage, stage_name)
            else:
                code = f"\n# TODO: Implement target for {stage_type}"
                note = f"⚠️ {stage_name}: Unknown target type {stage_type}"

            code_parts.append(code)
            if note:
                notes.append(note)

        return '\n'.join(code_parts), notes, connections

    # ========== Source Generators ==========

    def _generate_s3_source(self, stage: Dict, var_name: str) -> tuple:
        """Generate S3 source reading code."""
        stage_name = stage.get('name', 'source')
        code = f'''
# Read from S3: {stage_name}
{var_name}_dyf = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={{
        "paths": [args.get('S3_INPUT_PATH', 's3://bucket/input/')],
        "recurse": True
    }},
    format="parquet",  # TODO: Adjust format (csv, json, parquet, etc.)
    transformation_ctx="{var_name}_ctx"
)
{var_name}_df = {var_name}_dyf.toDF()
logger.info(f"{stage_name}: Read {{{{var_name}_df.count()}} records from S3")
'''
        return code, None

    def _generate_s3_partitioned_source(self, stage: Dict, var_name: str) -> tuple:
        """Generate partitioned S3 source code."""
        stage_name = stage.get('name', 'source')
        code = f'''
# Read partitioned data from S3: {stage_name}
{var_name}_dyf = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={{
        "paths": [args.get('S3_INPUT_PATH', 's3://bucket/input/')],
        "recurse": True,
        "groupFiles": "inPartition",
        "groupSize": "1048576"  # 1MB groups
    }},
    format="parquet",
    transformation_ctx="{var_name}_ctx"
)
{var_name}_df = {var_name}_dyf.toDF()
'''
        return code, None

    def _generate_catalog_source(self, stage: Dict, var_name: str) -> tuple:
        """Generate Glue Data Catalog source code."""
        stage_name = stage.get('name', 'source')
        code = f'''
# Read from Glue Data Catalog: {stage_name}
{var_name}_dyf = glueContext.create_dynamic_frame.from_catalog(
    database="your_database",  # TODO: Set database name
    table_name="your_table",   # TODO: Set table name
    transformation_ctx="{var_name}_ctx",
    push_down_predicate="(partition_date >= '2024-01-01')"  # TODO: Adjust predicate
)
{var_name}_df = {var_name}_dyf.toDF()
'''
        return code, f"📝 {stage_name}: Configure Glue Data Catalog database/table"

    def _generate_jdbc_source(self, stage: Dict, var_name: str, db_type: str = 'generic') -> tuple:
        """Generate JDBC source reading code."""
        stage_name = stage.get('name', 'source')
        connection_name = f"{db_type}_connection"

        code = f'''
# Read from {db_type.upper()} via JDBC: {stage_name}
{var_name}_dyf = glueContext.create_dynamic_frame.from_catalog(
    database="your_database",
    table_name="your_table",
    transformation_ctx="{var_name}_ctx",
    additional_options={{
        "jobBookmarkKeys": ["id"],  # TODO: Set bookmark key for incremental
        "jobBookmarkKeysSortOrder": "asc"
    }}
)
# Alternative: Direct JDBC connection
# {var_name}_dyf = glueContext.create_dynamic_frame.from_options(
#     connection_type="jdbc",
#     connection_options={{
#         "useConnectionProperties": "true",
#         "connectionName": "{connection_name}",
#         "dbtable": "schema.table_name",
#         "hashexpression": "id",  # Parallel read column
#         "hashpartitions": "10"
#     }},
#     transformation_ctx="{var_name}_ctx"
# )
{var_name}_df = {var_name}_dyf.toDF()
logger.info(f"{stage_name}: Read {{{{var_name}_df.count()}} records from {db_type.upper()}")
'''
        connection = GlueConnectionConfig(
            name=connection_name,
            connection_type="JDBC",
            jdbc_url=f"jdbc:{db_type}://host:port/database",
            username="username",
            password_secret_arn="arn:aws:secretsmanager:region:account:secret:name"
        )
        return code, f"🔌 {stage_name}: Create Glue Connection '{connection_name}'", connection

    def _generate_redshift_source(self, stage: Dict, var_name: str) -> tuple:
        """Generate Redshift source reading code."""
        stage_name = stage.get('name', 'source')
        code = f'''
# Read from Redshift: {stage_name}
{var_name}_dyf = glueContext.create_dynamic_frame.from_catalog(
    database="your_glue_database",
    table_name="your_redshift_table",
    redshift_tmp_dir=args.get("TempDir", "s3://bucket/temp/"),
    transformation_ctx="{var_name}_ctx"
)
{var_name}_df = {var_name}_dyf.toDF()
'''
        return code, f"📝 {stage_name}: Configure Redshift connection in Glue Catalog"

    # ========== Transform Generators ==========

    def _generate_apply_mapping(self, stage: Dict, var_name: str) -> tuple:
        """Generate ApplyMapping transformation."""
        stage_name = stage.get('name', 'transform')
        code = f'''
# Transform/Mapping: {stage_name}
{var_name}_mapped = ApplyMapping.apply(
    frame=input_dyf,  # TODO: Set correct input DynamicFrame
    mappings=[
        # ("source_col", "source_type", "target_col", "target_type"),
        ("id", "long", "id", "long"),
        ("name", "string", "customer_name", "string"),
        # TODO: Add column mappings from DataStage Transformer
    ],
    transformation_ctx="{var_name}_mapping_ctx"
)
{var_name}_df = {var_name}_mapped.toDF()
'''
        return code, f"📝 {stage_name}: Review and complete column mappings"

    def _generate_filter(self, stage: Dict, var_name: str) -> tuple:
        """Generate Filter transformation."""
        stage_name = stage.get('name', 'filter')
        code = f'''
# Filter: {stage_name}
{var_name}_filtered = Filter.apply(
    frame=input_dyf,  # TODO: Set correct input
    f=lambda x: x["status"] == "ACTIVE",  # TODO: Set filter condition
    transformation_ctx="{var_name}_filter_ctx"
)
# Alternative with SQL-like syntax:
# {var_name}_df = input_df.filter(F.col("status") == "ACTIVE")
'''
        return code, f"📝 {stage_name}: Configure filter condition from DataStage"

    def _generate_join(self, stage: Dict, var_name: str) -> tuple:
        """Generate Join transformation."""
        stage_name = stage.get('name', 'join')
        code = f'''
# Join: {stage_name}
{var_name}_joined = Join.apply(
    frame1=left_dyf,   # TODO: Set left input
    frame2=right_dyf,  # TODO: Set right input
    keys1=["id"],      # TODO: Set join keys
    keys2=["id"],
    transformation_ctx="{var_name}_join_ctx"
)
{var_name}_df = {var_name}_joined.toDF()
# For outer joins, use DataFrame API:
# {var_name}_df = left_df.join(right_df, on="id", how="left_outer")
'''
        return code, f"📝 {stage_name}: Configure join keys and type"

    def _generate_broadcast_join(self, stage: Dict, var_name: str) -> tuple:
        """Generate broadcast join for Lookup operations."""
        stage_name = stage.get('name', 'lookup')
        code = f'''
# Lookup (Broadcast Join): {stage_name}
# Broadcast the smaller lookup table for performance
from pyspark.sql.functions import broadcast

lookup_df = lookup_dyf.toDF()
{var_name}_df = main_df.join(
    broadcast(lookup_df),
    on=main_df["lookup_key"] == lookup_df["key"],  # TODO: Set lookup keys
    how="left"
).drop(lookup_df["key"])

# Handle lookup failures (no match)
{var_name}_df = {var_name}_df.withColumn(
    "lookup_value",
    F.coalesce(F.col("lookup_value"), F.lit("DEFAULT"))  # TODO: Set default
)
'''
        return code, f"📝 {stage_name}: Configure lookup keys and default values"

    def _generate_aggregate(self, stage: Dict, var_name: str) -> tuple:
        """Generate Aggregation transformation."""
        stage_name = stage.get('name', 'aggregate')
        code = f'''
# Aggregation: {stage_name}
{var_name}_df = input_df.groupBy(
    "group_col1", "group_col2"  # TODO: Set grouping columns
).agg(
    F.sum("amount").alias("total_amount"),
    F.count("*").alias("record_count"),
    F.avg("value").alias("avg_value"),
    F.max("date").alias("max_date")
    # TODO: Add aggregations from DataStage Aggregator
)
'''
        return code, f"📝 {stage_name}: Configure grouping columns and aggregations"

    def _generate_sort(self, stage: Dict, var_name: str) -> tuple:
        """Generate Sort transformation."""
        stage_name = stage.get('name', 'sort')
        code = f'''
# Sort: {stage_name}
{var_name}_df = input_df.orderBy(
    F.col("sort_col1").asc(),
    F.col("sort_col2").desc()
    # TODO: Set sort columns and order
)
'''
        return code, None

    def _generate_union(self, stage: Dict, var_name: str) -> tuple:
        """Generate Union/Funnel transformation."""
        stage_name = stage.get('name', 'union')
        code = f'''
# Union (Funnel): {stage_name}
{var_name}_df = input1_df.unionByName(
    input2_df,
    allowMissingColumns=True  # Handle schema differences
)
# For multiple inputs:
# {var_name}_df = input1_df.unionByName(input2_df).unionByName(input3_df)
'''
        return code, None

    def _generate_pivot(self, stage: Dict, var_name: str) -> tuple:
        """Generate Pivot transformation."""
        stage_name = stage.get('name', 'pivot')
        code = f'''
# Pivot: {stage_name}
# Note: Pivot requires DataFrame API (not DynamicFrame)
{var_name}_df = input_df.groupBy("row_key").pivot(
    "pivot_column",
    ["value1", "value2", "value3"]  # TODO: Set pivot values (optional but recommended)
).agg(
    F.sum("measure")  # TODO: Set aggregation
)
'''
        return code, f"📝 {stage_name}: Configure pivot column and values"

    def _generate_surrogate_key(self, stage: Dict, var_name: str) -> tuple:
        """Generate Surrogate Key generation."""
        stage_name = stage.get('name', 'surrogate_key')
        code = f'''
# Surrogate Key Generation: {stage_name}
from pyspark.sql.functions import monotonically_increasing_id, row_number

# Option 1: Simple monotonic ID (not guaranteed sequential)
{var_name}_df = input_df.withColumn("sk_id", monotonically_increasing_id())

# Option 2: Sequential within partition (better for dimension tables)
# window_spec = Window.orderBy("natural_key")
# {var_name}_df = input_df.withColumn("sk_id", row_number().over(window_spec))

# Option 3: Start from max existing value
# max_id = spark.sql("SELECT MAX(sk_id) FROM target_table").collect()[0][0] or 0
# {var_name}_df = input_df.withColumn("sk_id", monotonically_increasing_id() + max_id + 1)
'''
        return code, f"📝 {stage_name}: Choose appropriate surrogate key strategy"

    def _generate_cdc_logic(self, stage: Dict, var_name: str) -> tuple:
        """Generate CDC/Change Capture logic."""
        stage_name = stage.get('name', 'cdc')
        code = f'''
# CDC/Change Capture: {stage_name}
# ⚠️ IMPORTANT: This requires careful review of DataStage CDC logic

# Option 1: Use Glue Bookmarks for incremental processing
# (Configured at job level with --job-bookmark-option)

# Option 2: Implement SCD Type 2 with Delta Lake
# from delta.tables import DeltaTable
#
# if DeltaTable.isDeltaTable(spark, target_path):
#     delta_table = DeltaTable.forPath(spark, target_path)
#     delta_table.alias("target").merge(
#         source_df.alias("source"),
#         "target.id = source.id"
#     ).whenMatchedUpdate(
#         condition="target.hash != source.hash",
#         set={{
#             "effective_end_date": F.current_date(),
#             "is_current": F.lit(False)
#         }}
#     ).whenNotMatchedInsert(
#         values={{
#             "id": "source.id",
#             "effective_start_date": F.current_date(),
#             "effective_end_date": F.lit("9999-12-31"),
#             "is_current": F.lit(True)
#         }}
#     ).execute()

# Option 3: Manual change detection
{var_name}_df = input_df.withColumn(
    "record_hash",
    F.sha2(F.concat_ws("|", *[F.col(c) for c in input_df.columns]), 256)
).withColumn(
    "load_timestamp",
    F.current_timestamp()
)
'''
        return code, f"⚠️ {stage_name}: CDC logic requires manual review and testing"

    # ========== Target Generators ==========

    def _generate_s3_target(self, stage: Dict, stage_name: str) -> tuple:
        """Generate S3 target writing code."""
        var_name = self._to_variable_name(stage_name)
        code = f'''
# Write to S3: {stage_name}
output_dyf = DynamicFrame.fromDF(output_df, glueContext, "{var_name}_output")
glueContext.write_dynamic_frame.from_options(
    frame=output_dyf,
    connection_type="s3",
    connection_options={{
        "path": args.get('S3_OUTPUT_PATH', 's3://bucket/output/'),
        "partitionKeys": ["year", "month", "day"]  # TODO: Set partition keys
    }},
    format="parquet",
    format_options={{
        "compression": "snappy"
    }},
    transformation_ctx="{var_name}_sink_ctx"
)
logger.info(f"{stage_name}: Wrote {{output_df.count()}} records to S3")
'''
        return code, None

    def _generate_catalog_target(self, stage: Dict, stage_name: str) -> tuple:
        """Generate Glue Catalog target writing code."""
        var_name = self._to_variable_name(stage_name)
        code = f'''
# Write to Glue Data Catalog: {stage_name}
output_dyf = DynamicFrame.fromDF(output_df, glueContext, "{var_name}_output")
glueContext.write_dynamic_frame.from_catalog(
    frame=output_dyf,
    database="your_database",
    table_name="your_table",
    transformation_ctx="{var_name}_sink_ctx"
)
'''
        return code, f"📝 {stage_name}: Configure target Glue Catalog table"

    def _generate_jdbc_target(self, stage: Dict, stage_name: str) -> tuple:
        """Generate JDBC target writing code."""
        var_name = self._to_variable_name(stage_name)
        code = f'''
# Write to JDBC target: {stage_name}
output_dyf = DynamicFrame.fromDF(output_df, glueContext, "{var_name}_output")
glueContext.write_dynamic_frame.from_options(
    frame=output_dyf,
    connection_type="jdbc",
    connection_options={{
        "useConnectionProperties": "true",
        "connectionName": "target_connection",  # TODO: Set connection name
        "dbtable": "schema.table_name"  # TODO: Set target table
    }},
    transformation_ctx="{var_name}_sink_ctx"
)
'''
        return code, f"🔌 {stage_name}: Configure JDBC target connection"

    def _generate_redshift_target(self, stage: Dict, stage_name: str) -> tuple:
        """Generate Redshift target writing code."""
        var_name = self._to_variable_name(stage_name)
        code = f'''
# Write to Redshift: {stage_name}
output_dyf = DynamicFrame.fromDF(output_df, glueContext, "{var_name}_output")
glueContext.write_dynamic_frame.from_catalog(
    frame=output_dyf,
    database="your_glue_database",
    table_name="your_redshift_table",
    redshift_tmp_dir=args.get("TempDir", "s3://bucket/temp/"),
    transformation_ctx="{var_name}_sink_ctx"
)
'''
        return code, f"📝 {stage_name}: Configure Redshift target table"

    # ========== Helper Methods ==========

    def _is_source_stage(self, stage: Dict) -> bool:
        """Check if stage is a source."""
        stage_type = stage.get('type', '')
        source_indicators = ['File', 'Dataset', 'Connector', 'Redshift']
        # Check if it's in the typical source position (no incoming links)
        return any(ind in stage_type for ind in source_indicators)

    def _is_transform_stage(self, stage: Dict) -> bool:
        """Check if stage is a transformation."""
        stage_type = stage.get('type', '')
        transforms = ['Transformer', 'Filter', 'Join', 'Lookup', 'Aggregator',
                     'Sort', 'Funnel', 'Pivot', 'Merge', 'Change', 'Surrogate',
                     'Copy', 'Modify', 'Column', 'Row', 'Switch', 'Remove']
        return any(t in stage_type for t in transforms)

    def _is_target_stage(self, stage: Dict) -> bool:
        """Check if stage is a target."""
        # Targets are usually connectors/files at the end of the flow
        return self._is_source_stage(stage)  # Same types, different position

    def _to_variable_name(self, stage_name: str) -> str:
        """Convert stage name to valid Python variable name."""
        var = stage_name.lower().replace(' ', '_').replace('-', '_')
        var = ''.join(c if c.isalnum() or c == '_' else '_' for c in var)
        if var[0].isdigit():
            var = 'stage_' + var
        return var

    def _create_job_config(self, job_name: str, pattern: Any,
                          connections: List[GlueConnectionConfig]) -> GlueJobConfig:
        """Create Glue job configuration based on pattern analysis."""
        # Determine worker configuration based on complexity
        if pattern.complexity_score < 30:
            worker_type = GlueWorkerType.G_1X
            num_workers = 2
        elif pattern.complexity_score < 60:
            worker_type = GlueWorkerType.G_1X
            num_workers = 5
        else:
            worker_type = GlueWorkerType.G_2X
            num_workers = 10

        return GlueJobConfig(
            job_name=job_name,
            description=f"Migrated from DataStage: {pattern.job_name}",
            script_location=f"s3://glue-scripts/{job_name}.py",
            worker_type=worker_type,
            number_of_workers=num_workers,
            connections=[c.name for c in connections],
            default_arguments={
                "--job-language": "python",
                "--job-bookmark-option": "job-bookmark-enable",
                "--enable-metrics": "true",
                "--enable-spark-ui": "true",
                "--spark-event-logs-path": f"s3://spark-logs/{job_name}/",
                "--TempDir": f"s3://glue-temp/{job_name}/",
            },
            tags={
                "migrated_from": "datastage",
                "original_job": pattern.job_name,
                "complexity": str(int(pattern.complexity_score)),
            }
        )

    def _generate_terraform(self, config: GlueJobConfig,
                           connections: List[GlueConnectionConfig]) -> str:
        """Generate Terraform configuration for the Glue job."""
        terraform = f'''# Terraform configuration for Glue job: {config.job_name}
# Generated from DataStage migration

resource "aws_glue_job" "{config.job_name}" {{
  name     = "{config.job_name}"
  role_arn = var.glue_role_arn

  command {{
    name            = "{config.job_type.value}"
    script_location = "{config.script_location}"
    python_version  = "3"
  }}

  default_arguments = {{
{self._format_terraform_map(config.default_arguments, 4)}
  }}

  glue_version      = "{config.glue_version}"
  worker_type       = "{config.worker_type.value}"
  number_of_workers = {config.number_of_workers}
  timeout           = {config.timeout_minutes}
  max_retries       = {config.max_retries}

  connections = [{", ".join(f'"{c}"' for c in config.connections)}]

  tags = {{
{self._format_terraform_map(config.tags, 4)}
  }}
}}
'''
        # Add connection resources
        for conn in connections:
            terraform += f'''
resource "aws_glue_connection" "{conn.name}" {{
  name            = "{conn.name}"
  connection_type = "{conn.connection_type}"

  connection_properties = {{
    JDBC_CONNECTION_URL = "{conn.jdbc_url or 'jdbc:database://host:port/db'}"
    USERNAME            = "{conn.username or 'username'}"
    PASSWORD            = "{{{{var.{conn.name}_password}}}}"  # Use Secrets Manager in production
  }}

  physical_connection_requirements {{
    availability_zone      = var.availability_zone
    security_group_id_list = var.security_group_ids
    subnet_id              = var.subnet_id
  }}
}}
'''
        return terraform

    def _format_terraform_map(self, data: Dict[str, str], indent: int) -> str:
        """Format a dictionary as Terraform map entries."""
        spaces = " " * indent
        lines = [f'{spaces}"{k}" = "{v}"' for k, v in data.items()]
        return "\n".join(lines)

    def _estimate_dpu_hours(self, pattern: Any) -> float:
        """Estimate DPU hours based on job complexity."""
        # Base estimate: 0.5 DPU-hours for simple jobs
        base = 0.5

        # Add based on complexity
        complexity_factor = pattern.complexity_score / 100

        # Add based on stage count
        stage_factor = pattern.stage_count * 0.05

        return round(base + complexity_factor + stage_factor, 2)


class GlueBatchGenerator:
    """Generate multiple Glue jobs from a batch of DataStage patterns."""

    def __init__(self):
        self.generator = GlueScriptGenerator()

    def generate_batch(self, patterns: List[Any], structures: Dict[str, Dict]) -> Dict[str, GeneratedGlueJob]:
        """
        Generate Glue jobs for multiple DataStage patterns.

        Args:
            patterns: List of JobPattern objects
            structures: Dict mapping job names to their structures

        Returns:
            Dict mapping job names to GeneratedGlueJob objects
        """
        results = {}

        for pattern in patterns:
            job_name = pattern.job_name
            structure = structures.get(job_name, {'stages': [], 'links': []})

            try:
                generated = self.generator.generate_from_pattern(pattern, structure)
                results[job_name] = generated
                logger.info(f"Generated Glue job for: {job_name}")
            except Exception as e:
                logger.error(f"Failed to generate Glue job for {job_name}: {e}")
                results[job_name] = None

        return results

    def generate_summary_report(self, generated_jobs: Dict[str, GeneratedGlueJob]) -> Dict[str, Any]:
        """Generate a summary report of all generated jobs."""
        successful = [j for j in generated_jobs.values() if j is not None]

        total_dpu_hours = sum(j.estimated_dpu_hours for j in successful)
        all_notes = [note for j in successful for note in j.migration_notes]
        all_connections = []
        for j in successful:
            all_connections.extend(j.connections_needed)

        # Unique connections
        unique_connections = {c.name: c for c in all_connections}

        return {
            'total_jobs': len(generated_jobs),
            'successful_generations': len(successful),
            'failed_generations': len(generated_jobs) - len(successful),
            'estimated_total_dpu_hours': round(total_dpu_hours, 2),
            'estimated_monthly_cost_usd': round(total_dpu_hours * 0.44 * 30, 2),  # $0.44/DPU-hour
            'unique_connections_needed': len(unique_connections),
            'connections': list(unique_connections.keys()),
            'migration_notes_count': len(all_notes),
            'critical_notes': [n for n in all_notes if '⚠️' in n],
        }
