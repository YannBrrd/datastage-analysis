"""
Cost Tracker

Tracks LLM API usage and costs across providers.
Provides reporting and budget management.
"""

import json
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# Cost per 1K tokens by provider and model
COST_TABLE = {
    # Anthropic
    'anthropic': {
        'claude-sonnet-4-20250514': {'input': 0.003, 'output': 0.015},
        'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
        'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
        'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
        'default': {'input': 0.003, 'output': 0.015},
    },
    # Azure OpenAI
    'azure': {
        'gpt-4o': {'input': 0.005, 'output': 0.015},
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'default': {'input': 0.005, 'output': 0.015},
    },
    # Azure Foundry
    'azure_foundry': {
        'meta-llama-3.1-70b-instruct': {'input': 0.00268, 'output': 0.00354},
        'meta-llama-3.1-8b-instruct': {'input': 0.0003, 'output': 0.00061},
        'mistral-large': {'input': 0.004, 'output': 0.012},
        'cohere-command-r-plus': {'input': 0.003, 'output': 0.015},
        'default': {'input': 0.003, 'output': 0.01},
    },
    # AWS Bedrock
    'aws': {
        'anthropic.claude-3-sonnet-20240229-v1:0': {'input': 0.003, 'output': 0.015},
        'anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.00025, 'output': 0.00125},
        'meta.llama3-1-70b-instruct-v1:0': {'input': 0.00265, 'output': 0.0035},
        'mistral.mistral-large-2402-v1:0': {'input': 0.004, 'output': 0.012},
        'amazon.titan-text-premier-v1:0': {'input': 0.0005, 'output': 0.0015},
        'default': {'input': 0.003, 'output': 0.015},
    },
    # GCP Vertex AI
    'gcp': {
        'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005},
        'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
        'gemini-1.0-pro': {'input': 0.0005, 'output': 0.0015},
        'claude-3-5-sonnet@20240620': {'input': 0.003, 'output': 0.015},
        'default': {'input': 0.00125, 'output': 0.005},
    },
    # OpenRouter
    'openrouter': {
        'anthropic/claude-sonnet-4': {'input': 0.003, 'output': 0.015},
        'anthropic/claude-3.5-sonnet': {'input': 0.003, 'output': 0.015},
        'openai/gpt-4o': {'input': 0.005, 'output': 0.015},
        'meta-llama/llama-3.1-70b-instruct': {'input': 0.0008, 'output': 0.0008},
        'google/gemini-pro-1.5': {'input': 0.00125, 'output': 0.005},
        'default': {'input': 0.003, 'output': 0.015},
    },
}


@dataclass
class UsageRecord:
    """Single usage record."""
    timestamp: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    cached: bool
    job_name: Optional[str]
    operation: str  # 'migration', 'validation', 'documentation', etc.
    latency_ms: float


class CostTracker:
    """
    Tracks LLM API costs and usage.

    Features:
    - Per-request cost tracking
    - Aggregated reporting by provider, model, time period
    - Budget alerts
    - Savings from caching
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        budget_limit: Optional[float] = None,
        alert_threshold: float = 0.8
    ):
        """
        Initialize cost tracker.

        Args:
            db_path: Path to SQLite database for persistence
            budget_limit: Optional budget limit in dollars
            alert_threshold: Percentage of budget to trigger alert (0.0-1.0)
        """
        if db_path is None:
            cache_dir = Path('.cache')
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / 'cost_tracking.db')

        self.db_path = db_path
        self.budget_limit = budget_limit
        self.alert_threshold = alert_threshold
        self._init_db()

        # In-memory counters for current session
        self.session_start = time.time()
        self.session_requests = 0
        self.session_tokens = {'input': 0, 'output': 0}
        self.session_cost = 0.0
        self.session_cached = 0
        self.session_cache_savings = 0.0

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    cached INTEGER NOT NULL,
                    job_name TEXT,
                    operation TEXT NOT NULL,
                    latency_ms REAL NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp
                ON usage_records(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_provider
                ON usage_records(provider)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            provider: Provider name
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in dollars
        """
        provider_costs = COST_TABLE.get(provider, COST_TABLE.get('anthropic', {}))
        model_costs = provider_costs.get(model, provider_costs.get('default', {
            'input': 0.003, 'output': 0.015
        }))

        input_cost = (input_tokens / 1000) * model_costs['input']
        output_cost = (output_tokens / 1000) * model_costs['output']

        return input_cost + output_cost

    def record_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
        job_name: Optional[str] = None,
        operation: str = 'migration',
        latency_ms: float = 0.0
    ) -> UsageRecord:
        """
        Record a usage event.

        Args:
            provider: Provider name
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached: Whether response was from cache
            job_name: Associated job name
            operation: Type of operation
            latency_ms: Request latency

        Returns:
            UsageRecord object
        """
        cost = 0.0 if cached else self.calculate_cost(
            provider, model, input_tokens, output_tokens
        )

        record = UsageRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            cached=cached,
            job_name=job_name,
            operation=operation,
            latency_ms=latency_ms
        )

        # Update session counters
        self.session_requests += 1
        if cached:
            self.session_cached += 1
            self.session_cache_savings += self.calculate_cost(
                provider, model, input_tokens, output_tokens
            )
        else:
            self.session_tokens['input'] += input_tokens
            self.session_tokens['output'] += output_tokens
            self.session_cost += cost

        # Persist to database
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO usage_records
                (timestamp, provider, model, input_tokens, output_tokens, cost,
                 cached, job_name, operation, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp,
                    record.provider,
                    record.model,
                    record.input_tokens,
                    record.output_tokens,
                    record.cost,
                    1 if record.cached else 0,
                    record.job_name,
                    record.operation,
                    record.latency_ms
                )
            )
            conn.commit()

        # Check budget
        if self.budget_limit:
            self._check_budget()

        return record

    def _check_budget(self):
        """Check if approaching or exceeding budget."""
        total_cost = self.get_total_cost()

        if total_cost >= self.budget_limit:
            logger.warning(f"Budget exceeded: ${total_cost:.2f} / ${self.budget_limit:.2f}")
        elif total_cost >= self.budget_limit * self.alert_threshold:
            logger.warning(
                f"Approaching budget limit: ${total_cost:.2f} / ${self.budget_limit:.2f} "
                f"({total_cost/self.budget_limit*100:.1f}%)"
            )

    def get_total_cost(self, days: Optional[int] = None) -> float:
        """
        Get total cost.

        Args:
            days: Optional number of days to look back

        Returns:
            Total cost in dollars
        """
        with self._get_connection() as conn:
            if days:
                cutoff = time.time() - (days * 86400)
                cursor = conn.execute(
                    "SELECT SUM(cost) as total FROM usage_records WHERE timestamp > ?",
                    (cutoff,)
                )
            else:
                cursor = conn.execute("SELECT SUM(cost) as total FROM usage_records")

            row = cursor.fetchone()
            return row['total'] or 0.0

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        duration = time.time() - self.session_start

        return {
            'duration_seconds': round(duration, 1),
            'total_requests': self.session_requests,
            'cached_requests': self.session_cached,
            'cache_hit_rate': round(
                self.session_cached / self.session_requests * 100, 1
            ) if self.session_requests > 0 else 0,
            'input_tokens': self.session_tokens['input'],
            'output_tokens': self.session_tokens['output'],
            'total_tokens': self.session_tokens['input'] + self.session_tokens['output'],
            'total_cost': round(self.session_cost, 4),
            'cache_savings': round(self.session_cache_savings, 4),
            'effective_cost': round(self.session_cost, 4),
        }

    def get_report(
        self,
        days: int = 30,
        group_by: str = 'provider'
    ) -> Dict[str, Any]:
        """
        Generate usage report.

        Args:
            days: Number of days to include
            group_by: Grouping ('provider', 'model', 'operation', 'day')

        Returns:
            Report dictionary
        """
        cutoff = time.time() - (days * 86400)

        with self._get_connection() as conn:
            # Overall stats
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_requests,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(cost) as total_cost,
                    SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END) as cached_requests,
                    AVG(latency_ms) as avg_latency
                FROM usage_records
                WHERE timestamp > ?
                """,
                (cutoff,)
            )
            overall = dict(cursor.fetchone())

            # Grouped stats
            if group_by == 'day':
                group_col = "date(timestamp, 'unixepoch')"
            else:
                group_col = group_by

            cursor = conn.execute(
                f"""
                SELECT
                    {group_col} as group_key,
                    COUNT(*) as requests,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(cost) as cost,
                    SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END) as cached
                FROM usage_records
                WHERE timestamp > ?
                GROUP BY {group_col}
                ORDER BY cost DESC
                """,
                (cutoff,)
            )

            grouped = [dict(row) for row in cursor.fetchall()]

        # Calculate cache savings
        cache_savings = 0.0
        if overall['cached_requests']:
            # Estimate what cached requests would have cost
            avg_cost_per_request = (
                overall['total_cost'] /
                (overall['total_requests'] - overall['cached_requests'])
            ) if overall['total_requests'] > overall['cached_requests'] else 0
            cache_savings = overall['cached_requests'] * avg_cost_per_request

        return {
            'period_days': days,
            'overall': {
                'total_requests': overall['total_requests'] or 0,
                'total_input_tokens': overall['total_input_tokens'] or 0,
                'total_output_tokens': overall['total_output_tokens'] or 0,
                'total_cost': round(overall['total_cost'] or 0, 4),
                'cached_requests': overall['cached_requests'] or 0,
                'cache_hit_rate': round(
                    (overall['cached_requests'] or 0) /
                    (overall['total_requests'] or 1) * 100, 1
                ),
                'estimated_cache_savings': round(cache_savings, 4),
                'avg_latency_ms': round(overall['avg_latency'] or 0, 1),
            },
            f'by_{group_by}': grouped,
            'budget': {
                'limit': self.budget_limit,
                'used': round(overall['total_cost'] or 0, 4),
                'remaining': round(
                    (self.budget_limit or 0) - (overall['total_cost'] or 0), 4
                ) if self.budget_limit else None,
                'percentage_used': round(
                    (overall['total_cost'] or 0) / self.budget_limit * 100, 1
                ) if self.budget_limit else None,
            }
        }

    def format_report(self, report: Dict[str, Any]) -> str:
        """Format report as readable string."""
        lines = [
            "=" * 60,
            "LLM USAGE REPORT",
            "=" * 60,
            f"Period: Last {report['period_days']} days",
            "",
            "OVERALL STATISTICS",
            "-" * 40,
            f"Total Requests:      {report['overall']['total_requests']:,}",
            f"Total Input Tokens:  {report['overall']['total_input_tokens']:,}",
            f"Total Output Tokens: {report['overall']['total_output_tokens']:,}",
            f"Total Cost:          ${report['overall']['total_cost']:.4f}",
            f"Cached Requests:     {report['overall']['cached_requests']:,}",
            f"Cache Hit Rate:      {report['overall']['cache_hit_rate']:.1f}%",
            f"Est. Cache Savings:  ${report['overall']['estimated_cache_savings']:.4f}",
            f"Avg Latency:         {report['overall']['avg_latency_ms']:.1f}ms",
            "",
        ]

        if report.get('budget', {}).get('limit'):
            budget = report['budget']
            lines.extend([
                "BUDGET",
                "-" * 40,
                f"Limit:      ${budget['limit']:.2f}",
                f"Used:       ${budget['used']:.4f}",
                f"Remaining:  ${budget['remaining']:.4f}",
                f"Percentage: {budget['percentage_used']:.1f}%",
                "",
            ])

        # Find the grouped data
        for key, value in report.items():
            if key.startswith('by_') and value:
                group_name = key[3:].title()
                lines.extend([
                    f"BY {group_name.upper()}",
                    "-" * 40,
                ])
                for item in value[:10]:  # Top 10
                    lines.append(
                        f"  {item['group_key']}: "
                        f"{item['requests']:,} requests, "
                        f"${item['cost']:.4f}"
                    )
                lines.append("")

        lines.append("=" * 60)

        return '\n'.join(lines)

    def export_csv(self, output_path: str, days: int = 30):
        """Export usage records to CSV."""
        import csv

        cutoff = time.time() - (days * 86400)

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM usage_records
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                """,
                (cutoff,)
            )

            rows = cursor.fetchall()

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'datetime', 'provider', 'model',
                'input_tokens', 'output_tokens', 'cost',
                'cached', 'job_name', 'operation', 'latency_ms'
            ])

            for row in rows:
                writer.writerow([
                    row['timestamp'],
                    datetime.fromtimestamp(row['timestamp']).isoformat(),
                    row['provider'],
                    row['model'],
                    row['input_tokens'],
                    row['output_tokens'],
                    row['cost'],
                    'Yes' if row['cached'] else 'No',
                    row['job_name'] or '',
                    row['operation'],
                    row['latency_ms']
                ])

        logger.info(f"Exported {len(rows)} records to {output_path}")
