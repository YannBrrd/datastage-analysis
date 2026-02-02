"""
SQL Target Generator

Implements the BaseTargetGenerator interface for SQL databases.
Generates SQL scripts from DataStage jobs with support for multiple dialects.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ..base import BaseTargetGenerator, GeneratedOutput, TargetConfig
from .config import SQLTargetConfig, SQLDialect
from .dialects import BaseDialect, TeradataDialect, GenericDialect, get_dialect, TableDefinition, ColumnDefinition
from ....prediction.migration_predictor import MigrationPrediction, MigrationCategory

logger = logging.getLogger(__name__)


# Stage type mappings to SQL operations
STAGE_MAPPINGS = {
    # Database connectors (sources/targets)
    'OracleConnectorPX': {'type': 'table', 'driver': 'oracle'},
    'OracleODBC': {'type': 'table', 'driver': 'oracle'},
    'DB2ConnectorPX': {'type': 'table', 'driver': 'db2'},
    'TeradataConnectorPX': {'type': 'table', 'driver': 'teradata'},
    'SQLServerConnectorPX': {'type': 'table', 'driver': 'sqlserver'},
    'PostgreSQLConnectorPX': {'type': 'table', 'driver': 'postgresql'},
    'ODBCConnectorPX': {'type': 'table', 'driver': 'generic'},
    'DSDB2PX': {'type': 'table', 'driver': 'db2'},

    # File stages (external tables or staging)
    'PxSequentialFile': {'type': 'staging', 'format': 'csv'},
    'SequentialFile': {'type': 'staging', 'format': 'csv'},
    'PxDataSet': {'type': 'staging', 'format': 'binary'},
    'FileConnectorPX': {'type': 'staging', 'format': 'csv'},

    # Transform stages (SQL expressions)
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


class SQLTargetGenerator(BaseTargetGenerator):
    """
    SQL target generator.

    Generates SQL scripts from DataStage jobs.
    Supports multiple SQL dialects (Teradata, PostgreSQL, etc.).
    """

    TARGET_NAME = "sql"
    TARGET_DISPLAY_NAME = "SQL Database"
    MAIN_SCRIPT_EXTENSION = ".sql"
    INFRASTRUCTURE_EXTENSION = ".sql"
    INFRASTRUCTURE_TYPE = "ddl"

    def __init__(self, config: Optional[TargetConfig] = None):
        """
        Initialize the SQL target generator.

        Args:
            config: Optional SQLTargetConfig (or TargetConfig)
        """
        if config is None:
            config = SQLTargetConfig()
        elif not isinstance(config, SQLTargetConfig):
            config = SQLTargetConfig.from_dict(config.__dict__)

        super().__init__(config)
        self.sql_config: SQLTargetConfig = config

        # Get dialect instance
        self.dialect: BaseDialect = get_dialect(self.sql_config.dialect.value)

    def generate(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> GeneratedOutput:
        """
        Generate SQL code for a DataStage job.

        Args:
            prediction: Migration prediction with category and analysis
            structure: Parsed job structure from DSX parser
            **kwargs: Additional arguments

        Returns:
            GeneratedOutput with SQL script, DDL, tests, and docs
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

        # Generate infrastructure (DDL)
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
                'dialect': self.sql_config.dialect.value,
                'dialect_name': self.sql_config.get_dialect_name(),
            },
        )

    def generate_main_script(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> Tuple[str, List[str]]:
        """
        Generate the main SQL script.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            **kwargs: May contain 'analysis'

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

        # Build SQL statements
        statements = self._build_sql_statements(job_name, analysis, structure)

        # Wrap in batch script if enabled
        if self.sql_config.generate_batch_scripts:
            script = self.dialect.batch_script_wrapper(
                statements=statements,
                script_name=job_name,
                options={
                    'database': self.sql_config.teradata_options.get('default_database', 'DBC'),
                    'error_handling': self.sql_config.include_error_handling,
                }
            )
        else:
            script = self._build_simple_script(job_name, statements, analysis)

        return script, warnings

    def generate_infrastructure(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        **kwargs
    ) -> Optional[str]:
        """
        Generate DDL (CREATE TABLE, GRANT, etc.).

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            **kwargs: May contain 'analysis'

        Returns:
            DDL SQL script
        """
        analysis = kwargs.get('analysis') or self._analyze_structure(structure)
        job_name = prediction.job_name

        ddl_statements = []

        # Generate schema/database if needed
        if self.sql_config.generate_schemas:
            schema_ddl = self._generate_schema_ddl(analysis)
            if schema_ddl:
                ddl_statements.append(schema_ddl)

        # Generate table DDL for targets
        if self.sql_config.generate_ddl:
            for target in analysis.get('targets', []):
                table_ddl = self._generate_table_ddl(target, structure)
                if table_ddl:
                    ddl_statements.append(table_ddl)

        # Generate staging tables for file sources
        for source in analysis.get('sources', []):
            if source.get('type') == 'staging':
                staging_ddl = self._generate_staging_table_ddl(source, structure)
                if staging_ddl:
                    ddl_statements.append(staging_ddl)

        # Generate grants if enabled
        if self.sql_config.generate_grants:
            grants = self._generate_grants(analysis)
            if grants:
                ddl_statements.append(grants)

        if not ddl_statements:
            return None

        header = f"""/*******************************************************************************
 * DDL Script: {job_name}
 * Dialect: {self.sql_config.get_dialect_name()}
 * Generated by DataStage Migration Analyzer
 ******************************************************************************/

"""
        return header + '\n\n'.join(ddl_statements)

    def generate_tests(
        self,
        prediction: MigrationPrediction,
        structure: Dict[str, Any],
        main_script: str,
        **kwargs
    ) -> Optional[str]:
        """
        Generate SQL-based validation tests.

        Args:
            prediction: Migration prediction
            structure: Parsed job structure
            main_script: The generated SQL script
            **kwargs: Additional arguments

        Returns:
            SQL test script
        """
        job_name = prediction.job_name
        analysis = kwargs.get('analysis') or self._analyze_structure(structure)
        safe_name = self._safe_filename(job_name)

        targets = analysis.get('targets', [])
        if not targets:
            return None

        test_statements = []

        for target in targets:
            table_name = target.get('name', 'target_table')

            test_statements.append(f"""
-- Test: Row count validation for {table_name}
SELECT
    CASE
        WHEN COUNT(*) > 0 THEN 'PASS: {table_name} has data'
        ELSE 'FAIL: {table_name} is empty'
    END AS test_result,
    COUNT(*) AS row_count
FROM {table_name};
""")

            test_statements.append(f"""
-- Test: Null check for key columns in {table_name}
SELECT
    CASE
        WHEN null_count = 0 THEN 'PASS: No null keys in {table_name}'
        ELSE 'FAIL: ' || CAST(null_count AS VARCHAR(20)) || ' null keys found'
    END AS test_result
FROM (
    SELECT COUNT(*) AS null_count
    FROM {table_name}
    WHERE 1=0  -- TODO: Add actual key column NULL checks
) t;
""")

        return f"""/*******************************************************************************
 * Validation Tests: {job_name}
 * Dialect: {self.sql_config.get_dialect_name()}
 * Generated by DataStage Migration Analyzer
 ******************************************************************************/

{''.join(test_statements)}
"""

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
            main_script: The generated SQL script
            **kwargs: May contain 'analysis'

        Returns:
            Markdown documentation
        """
        analysis = kwargs.get('analysis') or self._analyze_structure(structure)
        job_name = prediction.job_name

        return self._build_documentation(job_name, prediction, analysis)

    def supports_batch_generation(self) -> bool:
        """SQL target supports batch generation."""
        return True

    def _analyze_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze job structure for SQL generation."""
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
                    'columns': self._extract_columns(stage),
                }

                # Categorize by role
                if mapping['type'] in ('table', 'staging'):
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
            'has_lookup': len(lookups) > 0,
            'has_aggregate': any(t.get('type') == 'aggregate' for t in transforms),
            'complex_transforms': len(transforms) > 2,
        }

    def _extract_columns(self, stage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract column definitions from a stage."""
        columns = []
        for col in stage.get('columns', []):
            columns.append({
                'name': col.get('name', 'unknown'),
                'type': col.get('type', 'VARCHAR'),
                'length': col.get('length'),
                'precision': col.get('precision'),
                'scale': col.get('scale'),
                'nullable': col.get('nullable', True),
            })
        return columns

    def _determine_pattern(
        self,
        sources: List[Dict],
        targets: List[Dict],
        transforms: List[Dict],
        lookups: List[Dict]
    ) -> str:
        """Determine the job pattern type."""
        has_staging_source = any(s.get('type') == 'staging' for s in sources)
        has_staging_target = any(t.get('type') == 'staging' for t in targets)
        has_table_source = any(s.get('type') == 'table' for s in sources)
        has_table_target = any(t.get('type') == 'table' for t in targets)
        has_lookup = len(lookups) > 0

        if has_staging_source and has_table_target:
            return 'staging_to_table'
        elif has_table_source and has_table_target:
            return 'table_to_table'
        elif has_table_source and has_staging_target:
            return 'table_to_staging'
        elif has_lookup:
            return 'lookup_transform'
        elif transforms:
            return 'transform'
        else:
            return 'simple_copy'

    def _build_sql_statements(
        self,
        job_name: str,
        analysis: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> List[str]:
        """Build the main SQL statements for the job."""
        statements = []
        sources = analysis.get('sources', [])
        targets = analysis.get('targets', [])
        lookups = analysis.get('lookups', [])

        # Comment header
        statements.append(f"/* Job: {job_name} - Pattern: {analysis.get('pattern', 'unknown')} */")

        # Build source query
        source_query = self._build_source_query(sources, lookups, analysis)

        # Generate INSERT or MERGE for each target
        for target in targets:
            target_name = target.get('name', 'target_table')

            # Check if this is an upsert pattern
            if self._is_upsert_pattern(target, structure):
                key_columns = self._extract_key_columns(target, structure)
                update_columns = self._extract_update_columns(target, structure)

                stmt = self.dialect.merge_statement(
                    target_table=target_name,
                    source_query=source_query,
                    key_columns=key_columns,
                    update_columns=update_columns,
                )
            else:
                # Simple insert
                columns = [c.get('name') for c in target.get('columns', [])]
                stmt = self.dialect.insert_select(
                    target_table=target_name,
                    source_query=source_query,
                    columns=columns if columns else None,
                )

            statements.append(stmt)

        if not statements or len(statements) == 1:
            statements.append(f"""
/* TODO: Complete SQL migration for {job_name}
   Sources: {', '.join(s['name'] for s in sources) or 'None detected'}
   Targets: {', '.join(t['name'] for t in targets) or 'None detected'}
*/
SELECT 'Migration placeholder' AS status;
""")

        return statements

    def _build_source_query(
        self,
        sources: List[Dict],
        lookups: List[Dict],
        analysis: Dict[str, Any]
    ) -> str:
        """Build the source SELECT query."""
        if not sources:
            return "SELECT * FROM dual  -- TODO: Add source table"

        main_source = sources[0]
        source_name = main_source.get('name', 'source_table')

        # Start with main source
        query = f"SELECT\n    *  -- TODO: Specify columns\nFROM {source_name} src"

        # Add lookups as JOINs
        for i, lookup in enumerate(lookups):
            lookup_name = lookup.get('name', f'lookup_{i}')
            query += f"\nLEFT JOIN {lookup_name} lkp{i}\n    ON src.key = lkp{i}.key  -- TODO: Specify join keys"

        # Add additional sources as JOINs
        for i, source in enumerate(sources[1:], 1):
            query += f"\nJOIN {source.get('name', f'source_{i}')} src{i}\n    ON src.key = src{i}.key  -- TODO: Specify join keys"

        return query

    def _is_upsert_pattern(self, target: Dict, structure: Dict) -> bool:
        """Check if target uses upsert/merge pattern."""
        props = target.get('properties', {})
        return props.get('write_mode') in ('merge', 'upsert', 'update_insert')

    def _extract_key_columns(self, target: Dict, structure: Dict) -> List[str]:
        """Extract key columns for merge operations."""
        columns = target.get('columns', [])
        keys = [c['name'] for c in columns if c.get('is_key') or c.get('primary_key')]
        return keys if keys else ['id']  # Default to 'id' if no keys found

    def _extract_update_columns(self, target: Dict, structure: Dict) -> List[str]:
        """Extract update columns for merge operations."""
        columns = target.get('columns', [])
        keys = set(self._extract_key_columns(target, structure))
        return [c['name'] for c in columns if c['name'] not in keys]

    def _build_simple_script(
        self,
        job_name: str,
        statements: List[str],
        analysis: Dict[str, Any]
    ) -> str:
        """Build a simple SQL script without batch wrapper."""
        header = f"""/*******************************************************************************
 * SQL Script: {job_name}
 * Dialect: {self.sql_config.get_dialect_name()}
 * Pattern: {analysis.get('pattern', 'unknown')}
 * Generated by DataStage Migration Analyzer
 * Generated at: {datetime.now().isoformat()}
 ******************************************************************************/

"""
        return header + '\n\n'.join(statements)

    def _generate_schema_ddl(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Generate schema/database DDL."""
        database = self.sql_config.teradata_options.get('default_database', 'DBC')

        if self.sql_config.dialect == SQLDialect.TERADATA:
            return f"""-- Create database if not exists (Teradata)
-- DATABASE {database};
"""
        return None

    def _generate_table_ddl(self, target: Dict, structure: Dict) -> str:
        """Generate CREATE TABLE DDL for a target."""
        columns = target.get('columns', [])

        col_defs = []
        for col in columns:
            sql_type = self.dialect.map_type(
                col.get('type', 'VARCHAR'),
                length=col.get('length', 255),
                precision=col.get('precision'),
                scale=col.get('scale'),
            )
            col_defs.append(ColumnDefinition(
                name=col.get('name', 'unknown'),
                data_type=sql_type,
                nullable=col.get('nullable', True),
            ))

        if not col_defs:
            col_defs = [
                ColumnDefinition(name='id', data_type='INTEGER', nullable=False),
                ColumnDefinition(name='data', data_type='VARCHAR(255)', nullable=True),
            ]

        # Extract primary key
        pk_cols = [c.get('name') for c in columns if c.get('is_key') or c.get('primary_key')]

        table_def = TableDefinition(
            name=target.get('name', 'target_table'),
            columns=col_defs,
            primary_key=pk_cols if pk_cols else None,
        )

        return self.dialect.create_table(
            table_def,
            options=self.sql_config.teradata_options
        )

    def _generate_staging_table_ddl(self, source: Dict, structure: Dict) -> str:
        """Generate staging table DDL for file sources."""
        staging_name = f"STG_{source.get('name', 'source')}"

        columns = source.get('columns', [])
        col_defs = []
        for col in columns:
            sql_type = self.dialect.map_type(
                col.get('type', 'VARCHAR'),
                length=col.get('length', 255),
            )
            col_defs.append(ColumnDefinition(
                name=col.get('name', 'unknown'),
                data_type=sql_type,
                nullable=True,  # Staging tables allow nulls
            ))

        if not col_defs:
            col_defs = [ColumnDefinition(name='raw_data', data_type='VARCHAR(4000)', nullable=True)]

        table_def = TableDefinition(
            name=staging_name,
            columns=col_defs,
        )

        options = dict(self.sql_config.teradata_options)
        options['use_primary_index'] = False  # No PI for staging

        return f"-- Staging table for {source.get('name', 'source')}\n" + \
               self.dialect.create_table(table_def, options=options)

    def _generate_grants(self, analysis: Dict[str, Any]) -> str:
        """Generate GRANT statements."""
        targets = analysis.get('targets', [])

        grants = []
        for target in targets:
            table_name = target.get('name', 'target_table')
            grants.append(f"-- GRANT SELECT, INSERT, UPDATE ON {table_name} TO etl_user;")

        return '\n'.join(grants) if grants else ""

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
            stage_rows.append(f"| {src['name']} | {src['original_type']} | Source | SELECT |")
        for t in transforms:
            stage_rows.append(f"| {t['name']} | {t['original_type']} | Transform | SQL Expression |")
        for tgt in targets:
            stage_rows.append(f"| {tgt['name']} | {tgt['original_type']} | Target | INSERT/MERGE |")

        stage_table = '\n'.join(stage_rows) if stage_rows else "| (No stages detected) | - | - | - |"

        # Build checklist
        checklist = [
            "- [ ] Review generated SQL script",
            "- [ ] Verify column mappings",
            "- [ ] Test with sample data",
        ]
        if analysis.get('has_lookup'):
            checklist.append("- [ ] Verify JOIN conditions")
        if analysis.get('has_aggregate'):
            checklist.append("- [ ] Verify GROUP BY logic")
        checklist.extend([
            "- [ ] Run DDL scripts",
            "- [ ] Execute validation tests",
            "- [ ] Compare row counts with source",
        ])

        safe_name = self._safe_filename(job_name)

        return f'''# Migration Documentation: {job_name}

## Overview

| Property | Value |
|----------|-------|
| **Original Job** | {job_name} |
| **Target Platform** | {self.sql_config.get_dialect_name()} |
| **Migration Category** | {prediction.category.value} |
| **Confidence** | {prediction.confidence:.0%} |
| **Success Probability** | {prediction.success_probability:.0%} |
| **Estimated Effort** | {prediction.estimated_hours:.1f} hours |
| **Risk Level** | {prediction.risk_level.value} |
| **Pattern** | {analysis.get('pattern', 'unknown')} |
| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |

## Stage Mapping

| DataStage Stage | Type | Role | SQL Equivalent |
|-----------------|------|------|----------------|
{stage_table}

## Risk Factors

{chr(10).join(f"- {rf}" for rf in prediction.risk_factors) if prediction.risk_factors else "- No significant risk factors identified"}

## Automation Blockers

{chr(10).join(f"- {ab}" for ab in prediction.automation_blockers) if prediction.automation_blockers else "- No automation blockers identified"}

## {self.sql_config.get_dialect_name()} Considerations

{self._get_dialect_considerations()}

## Recommendations

{chr(10).join(f"- {rec}" for rec in prediction.recommendations) if prediction.recommendations else "- Follow standard migration procedure"}

## Migration Checklist

{chr(10).join(checklist)}

## Generated Files

- `sql/{safe_name}.sql` - Main SQL script
- `infrastructure/{safe_name}.sql` - DDL (CREATE TABLE, GRANT)
- `tests/test_{safe_name}.sql` - Validation tests

## Notes

{f"**Unknown Stages**: {', '.join(analysis.get('unknown_stages', []))}" if analysis.get('unknown_stages') else "All stages successfully mapped to SQL."}

---
*Generated by DataStage Migration Analyzer - SQL Target ({self.sql_config.get_dialect_name()})*
'''

    def _get_dialect_considerations(self) -> str:
        """Get dialect-specific considerations for documentation."""
        if self.sql_config.dialect == SQLDialect.TERADATA:
            return """- Using MULTISET tables (allows duplicates)
- PRIMARY INDEX should be chosen based on access patterns
- Consider using BTEQ for batch execution
- Use TPT for high-volume data loads
- MERGE statement used for upsert patterns"""
        elif self.sql_config.dialect == SQLDialect.POSTGRESQL:
            return """- Use INSERT ... ON CONFLICT for upserts
- Consider partitioning for large tables
- Use COPY for bulk data loads"""
        else:
            return "- Standard SQL syntax used"
