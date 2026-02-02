"""
SQL Target Configuration

SQL-specific configuration options with dialect support.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from ..base import TargetConfig


class SQLDialect(Enum):
    """Supported SQL dialects."""
    TERADATA = "teradata"
    POSTGRESQL = "postgresql"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    GENERIC = "generic"


@dataclass
class SQLTargetConfig(TargetConfig):
    """
    Configuration for SQL target generator.

    Extends TargetConfig with SQL-specific options.
    """
    target_name: str = "sql"

    # SQL dialect
    dialect: SQLDialect = SQLDialect.TERADATA

    # Output mode
    generate_stored_procedures: bool = False  # Simple scripts by default
    generate_batch_scripts: bool = True       # BTEQ/TPT for batch processing

    # Infrastructure generation
    generate_ddl: bool = True                 # CREATE TABLE, etc.
    generate_grants: bool = True              # User/role grants
    generate_schemas: bool = True             # CREATE SCHEMA/DATABASE

    # Teradata-specific options
    teradata_options: Dict[str, Any] = field(default_factory=lambda: {
        'use_multiset': True,          # MULTISET TABLE (vs SET)
        'use_primary_index': True,     # Generate PRIMARY INDEX
        'use_bteq': True,              # Use BTEQ for batch scripts
        'use_tpt': False,              # Use TPT for high-volume loads
        'fallback': False,             # FALLBACK option
        'journal': False,              # JOURNAL option
        'default_database': 'DBC',     # Default database
    })

    # Script options
    include_comments: bool = True
    include_error_handling: bool = True
    script_header_template: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SQLTargetConfig':
        """Create config from dictionary."""
        dialect_str = data.get('dialect', 'teradata')
        try:
            dialect = SQLDialect(dialect_str)
        except ValueError:
            dialect = SQLDialect.GENERIC

        return cls(
            target_name=data.get('target_name', 'sql'),
            enabled=data.get('enabled', True),
            generate_infrastructure=data.get('generate_infrastructure', True),
            generate_tests=data.get('generate_tests', True),
            generate_docs=data.get('generate_docs', True),
            options=data.get('options', {}),
            dialect=dialect,
            generate_stored_procedures=data.get('generate_stored_procedures', False),
            generate_batch_scripts=data.get('generate_batch_scripts', True),
            generate_ddl=data.get('generate_ddl', True),
            generate_grants=data.get('generate_grants', True),
            generate_schemas=data.get('generate_schemas', True),
            teradata_options=data.get('teradata_options', cls.teradata_options),
            include_comments=data.get('include_comments', True),
            include_error_handling=data.get('include_error_handling', True),
            script_header_template=data.get('script_header_template'),
        )

    def get_dialect_name(self) -> str:
        """Get the human-readable dialect name."""
        names = {
            SQLDialect.TERADATA: "Teradata",
            SQLDialect.POSTGRESQL: "PostgreSQL",
            SQLDialect.ORACLE: "Oracle",
            SQLDialect.SQLSERVER: "SQL Server",
            SQLDialect.GENERIC: "Generic SQL",
        }
        return names.get(self.dialect, "Unknown")
