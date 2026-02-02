"""
SQL Dialect Implementations

Provides dialect-specific SQL generation logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ColumnDefinition:
    """Column definition for DDL generation."""
    name: str
    data_type: str
    nullable: bool = True
    default: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class TableDefinition:
    """Table definition for DDL generation."""
    name: str
    schema: Optional[str] = None
    columns: List[ColumnDefinition] = None
    primary_key: List[str] = None
    indexes: Dict[str, List[str]] = None


class BaseDialect(ABC):
    """Abstract base class for SQL dialects."""

    DIALECT_NAME: str = "generic"

    # Type mappings from DataStage to this dialect
    TYPE_MAPPINGS: Dict[str, str] = {
        'VarChar': 'VARCHAR',
        'Char': 'CHAR',
        'Integer': 'INTEGER',
        'SmallInt': 'SMALLINT',
        'BigInt': 'BIGINT',
        'Decimal': 'DECIMAL',
        'Float': 'FLOAT',
        'Double': 'DOUBLE PRECISION',
        'Date': 'DATE',
        'Time': 'TIME',
        'Timestamp': 'TIMESTAMP',
        'Binary': 'BINARY',
        'VarBinary': 'VARBINARY',
        'LongVarChar': 'CLOB',
        'LongVarBinary': 'BLOB',
    }

    @abstractmethod
    def create_table(
        self,
        table: TableDefinition,
        options: Dict[str, Any] = None
    ) -> str:
        """Generate CREATE TABLE statement."""
        pass

    @abstractmethod
    def insert_select(
        self,
        target_table: str,
        source_query: str,
        columns: List[str] = None
    ) -> str:
        """Generate INSERT ... SELECT statement."""
        pass

    @abstractmethod
    def merge_statement(
        self,
        target_table: str,
        source_query: str,
        key_columns: List[str],
        update_columns: List[str],
        insert_columns: List[str] = None
    ) -> str:
        """Generate MERGE/UPSERT statement."""
        pass

    @abstractmethod
    def batch_script_wrapper(
        self,
        statements: List[str],
        script_name: str,
        options: Dict[str, Any] = None
    ) -> str:
        """Wrap SQL statements in a batch script (BTEQ, psql, etc.)."""
        pass

    def map_type(self, datastage_type: str, length: int = None, precision: int = None, scale: int = None) -> str:
        """Map DataStage type to dialect-specific type."""
        base_type = self.TYPE_MAPPINGS.get(datastage_type, datastage_type)

        if length and base_type in ('VARCHAR', 'CHAR', 'BINARY', 'VARBINARY'):
            return f"{base_type}({length})"
        elif precision is not None:
            if scale is not None:
                return f"{base_type}({precision},{scale})"
            return f"{base_type}({precision})"
        return base_type

    def quote_identifier(self, name: str) -> str:
        """Quote an identifier (table, column name)."""
        return f'"{name}"'

    def qualify_table(self, table: str, schema: str = None, database: str = None) -> str:
        """Fully qualify a table name."""
        parts = []
        if database:
            parts.append(database)
        if schema:
            parts.append(schema)
        parts.append(table)
        return '.'.join(parts)


class TeradataDialect(BaseDialect):
    """
    Teradata SQL dialect.

    Supports Teradata-specific features:
    - MULTISET vs SET tables
    - PRIMARY INDEX
    - BTEQ/TPT batch processing
    - MERGE for upserts
    """

    DIALECT_NAME = "teradata"

    TYPE_MAPPINGS = {
        'VarChar': 'VARCHAR',
        'Char': 'CHAR',
        'Integer': 'INTEGER',
        'SmallInt': 'SMALLINT',
        'BigInt': 'BIGINT',
        'Decimal': 'DECIMAL',
        'Float': 'FLOAT',
        'Double': 'FLOAT',  # Teradata uses FLOAT for both
        'Date': 'DATE',
        'Time': 'TIME',
        'Timestamp': 'TIMESTAMP',
        'Binary': 'BYTE',
        'VarBinary': 'VARBYTE',
        'LongVarChar': 'CLOB',
        'LongVarBinary': 'BLOB',
        # Teradata-specific
        'Number': 'NUMBER',
        'ByteInt': 'BYTEINT',
    }

    def create_table(
        self,
        table: TableDefinition,
        options: Dict[str, Any] = None
    ) -> str:
        """Generate Teradata CREATE TABLE statement."""
        options = options or {}

        # Table type
        table_type = "MULTISET" if options.get('use_multiset', True) else "SET"

        # Table options
        table_opts = []
        if options.get('fallback'):
            table_opts.append("FALLBACK")
        if options.get('journal'):
            table_opts.append("JOURNAL")

        # Qualified name
        qualified_name = self.qualify_table(
            table.name,
            schema=table.schema,
            database=options.get('database')
        )

        # Column definitions
        col_defs = []
        for col in (table.columns or []):
            col_def = f"    {col.name} {col.data_type}"
            if not col.nullable:
                col_def += " NOT NULL"
            if col.default:
                col_def += f" DEFAULT {col.default}"
            col_defs.append(col_def)

        columns_sql = ',\n'.join(col_defs) if col_defs else "    -- Add columns here"

        # Primary index
        primary_index = ""
        if options.get('use_primary_index', True) and table.primary_key:
            pi_cols = ', '.join(table.primary_key)
            primary_index = f"\nPRIMARY INDEX ({pi_cols})"

        # Build statement
        opts_str = f", {', '.join(table_opts)}" if table_opts else ""

        return f"""CREATE {table_type} TABLE {qualified_name}{opts_str}
(
{columns_sql}
){primary_index};"""

    def insert_select(
        self,
        target_table: str,
        source_query: str,
        columns: List[str] = None
    ) -> str:
        """Generate Teradata INSERT ... SELECT statement."""
        cols_clause = ""
        if columns:
            cols_clause = f" ({', '.join(columns)})"

        return f"""INSERT INTO {target_table}{cols_clause}
{source_query};"""

    def merge_statement(
        self,
        target_table: str,
        source_query: str,
        key_columns: List[str],
        update_columns: List[str],
        insert_columns: List[str] = None
    ) -> str:
        """Generate Teradata MERGE statement."""
        # Build ON clause
        on_conditions = ' AND '.join(
            f"tgt.{col} = src.{col}" for col in key_columns
        )

        # Build UPDATE SET clause
        update_sets = ', '.join(
            f"tgt.{col} = src.{col}" for col in update_columns
        )

        # Build INSERT clause
        insert_cols = insert_columns or (key_columns + update_columns)
        insert_col_list = ', '.join(insert_cols)
        insert_val_list = ', '.join(f"src.{col}" for col in insert_cols)

        return f"""MERGE INTO {target_table} AS tgt
USING (
{source_query}
) AS src
ON {on_conditions}
WHEN MATCHED THEN
    UPDATE SET {update_sets}
WHEN NOT MATCHED THEN
    INSERT ({insert_col_list})
    VALUES ({insert_val_list});"""

    def batch_script_wrapper(
        self,
        statements: List[str],
        script_name: str,
        options: Dict[str, Any] = None
    ) -> str:
        """Generate BTEQ script for Teradata."""
        options = options or {}
        database = options.get('database', 'DBC')
        logon = options.get('logon', '${TDPID}/${TDUSER},${TDPASSWORD}')

        # Error handling
        error_handling = ""
        if options.get('error_handling', True):
            error_handling = """
.SET ERROROUT STDOUT
.SET ERRORLEVEL (3807) SEVERITY 0  -- Table does not exist (for DROP IF EXISTS)

/* Exit on error */
.IF ERRORCODE <> 0 THEN .GOTO ERROR_EXIT
"""

        # Join statements
        statements_sql = '\n\n'.join(statements)

        return f"""/*******************************************************************************
 * BTEQ Script: {script_name}
 * Database: {database}
 * Generated by DataStage Migration Analyzer
 ******************************************************************************/

.LOGON {logon}

.SET WIDTH 254
.SET SEPARATOR '|'
{error_handling}
DATABASE {database};

/* ============================================================================
   Main SQL Statements
   ============================================================================ */

{statements_sql}

/* ============================================================================
   Successful completion
   ============================================================================ */
.IF ERRORCODE = 0 THEN .GOTO SUCCESS_EXIT

.LABEL ERROR_EXIT
.QUIT 8

.LABEL SUCCESS_EXIT
.LOGOFF
.QUIT 0
"""

    def quote_identifier(self, name: str) -> str:
        """Quote identifier for Teradata (double quotes)."""
        return f'"{name}"'


class GenericDialect(BaseDialect):
    """Generic ANSI SQL dialect."""

    DIALECT_NAME = "generic"

    def create_table(
        self,
        table: TableDefinition,
        options: Dict[str, Any] = None
    ) -> str:
        """Generate standard CREATE TABLE statement."""
        options = options or {}

        qualified_name = self.qualify_table(table.name, schema=table.schema)

        col_defs = []
        for col in (table.columns or []):
            col_def = f"    {col.name} {col.data_type}"
            if not col.nullable:
                col_def += " NOT NULL"
            if col.default:
                col_def += f" DEFAULT {col.default}"
            col_defs.append(col_def)

        # Primary key constraint
        if table.primary_key:
            pk_cols = ', '.join(table.primary_key)
            col_defs.append(f"    PRIMARY KEY ({pk_cols})")

        columns_sql = ',\n'.join(col_defs) if col_defs else "    -- Add columns here"

        return f"""CREATE TABLE {qualified_name}
(
{columns_sql}
);"""

    def insert_select(
        self,
        target_table: str,
        source_query: str,
        columns: List[str] = None
    ) -> str:
        """Generate standard INSERT ... SELECT statement."""
        cols_clause = ""
        if columns:
            cols_clause = f" ({', '.join(columns)})"

        return f"""INSERT INTO {target_table}{cols_clause}
{source_query};"""

    def merge_statement(
        self,
        target_table: str,
        source_query: str,
        key_columns: List[str],
        update_columns: List[str],
        insert_columns: List[str] = None
    ) -> str:
        """Generate standard MERGE statement."""
        on_conditions = ' AND '.join(
            f"tgt.{col} = src.{col}" for col in key_columns
        )

        update_sets = ', '.join(
            f"tgt.{col} = src.{col}" for col in update_columns
        )

        insert_cols = insert_columns or (key_columns + update_columns)
        insert_col_list = ', '.join(insert_cols)
        insert_val_list = ', '.join(f"src.{col}" for col in insert_cols)

        return f"""MERGE INTO {target_table} AS tgt
USING (
{source_query}
) AS src
ON {on_conditions}
WHEN MATCHED THEN
    UPDATE SET {update_sets}
WHEN NOT MATCHED THEN
    INSERT ({insert_col_list})
    VALUES ({insert_val_list});"""

    def batch_script_wrapper(
        self,
        statements: List[str],
        script_name: str,
        options: Dict[str, Any] = None
    ) -> str:
        """Generate simple SQL script."""
        statements_sql = '\n\n'.join(statements)

        return f"""-- SQL Script: {script_name}
-- Generated by DataStage Migration Analyzer

{statements_sql}
"""


def get_dialect(dialect_name: str) -> BaseDialect:
    """
    Get a dialect instance by name.

    Args:
        dialect_name: Name of the dialect ('teradata', 'postgresql', etc.)

    Returns:
        Dialect instance
    """
    dialects = {
        'teradata': TeradataDialect(),
        'generic': GenericDialect(),
        # Add more dialects as needed
    }

    return dialects.get(dialect_name.lower(), GenericDialect())
