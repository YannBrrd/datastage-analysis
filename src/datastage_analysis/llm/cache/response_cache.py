"""
Response Cache Implementation

SQLite-based caching for LLM responses with exact-match lookup.
"""

import os
import json
import time
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from ..client import LLMClient, LLMResponse, ModelInfo

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    SQLite-based cache for LLM responses.

    Stores responses keyed by a hash of (prompt, system, model, temperature).
    Supports TTL (time-to-live) for cache entries.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        ttl_hours: int = 168,  # 7 days default
    ):
        """
        Initialize the response cache.

        Args:
            cache_path: Path to SQLite database file
                        Default: .cache/llm_cache.db
            ttl_hours: Time-to-live for cache entries in hours
        """
        if cache_path is None:
            cache_dir = Path('.cache')
            cache_dir.mkdir(exist_ok=True)
            cache_path = str(cache_dir / 'llm_cache.db')

        self.cache_path = cache_path
        self.ttl_seconds = ttl_hours * 3600
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    response_content TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    finish_reason TEXT,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_hash
                ON llm_cache(prompt_hash)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON llm_cache(created_at)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.cache_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _generate_cache_key(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        temperature: float
    ) -> str:
        """
        Generate a deterministic cache key.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model identifier
            temperature: Temperature setting

        Returns:
            SHA-256 hash as cache key
        """
        content = json.dumps({
            'prompt': prompt,
            'system': system or '',
            'model': model,
            'temperature': round(temperature, 2),
        }, sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        temperature: float
    ) -> Optional[LLMResponse]:
        """
        Retrieve cached response if available and not expired.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model identifier
            temperature: Temperature setting

        Returns:
            LLMResponse if cached and valid, None otherwise
        """
        cache_key = self._generate_cache_key(prompt, system, model, temperature)
        current_time = time.time()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM llm_cache
                WHERE cache_key = ? AND created_at > ?
                """,
                (cache_key, current_time - self.ttl_seconds)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # Update access stats
            conn.execute(
                """
                UPDATE llm_cache
                SET last_accessed = ?, access_count = access_count + 1
                WHERE cache_key = ?
                """,
                (current_time, cache_key)
            )
            conn.commit()

            # Parse metadata
            metadata = {}
            if row['metadata']:
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    pass

            logger.debug(f"Cache hit for key {cache_key[:16]}...")

            return LLMResponse(
                content=row['response_content'],
                model=row['model'],
                input_tokens=row['input_tokens'],
                output_tokens=row['output_tokens'],
                cached=True,
                latency_ms=0.0,
                finish_reason=row['finish_reason'] or 'stop',
                metadata=metadata,
            )

    def set(
        self,
        prompt: str,
        system: Optional[str],
        model: str,
        temperature: float,
        response: LLMResponse
    ):
        """
        Store a response in the cache.

        Args:
            prompt: User prompt
            system: System prompt
            model: Model identifier
            temperature: Temperature setting
            response: LLM response to cache
        """
        cache_key = self._generate_cache_key(prompt, system, model, temperature)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:32]
        current_time = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (cache_key, prompt_hash, model, temperature, response_content,
                 input_tokens, output_tokens, finish_reason, metadata,
                 created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    cache_key,
                    prompt_hash,
                    model,
                    temperature,
                    response.content,
                    response.input_tokens,
                    response.output_tokens,
                    response.finish_reason,
                    json.dumps(response.metadata) if response.metadata else None,
                    current_time,
                    current_time,
                )
            )
            conn.commit()

        logger.debug(f"Cached response for key {cache_key[:16]}...")

    def invalidate(self, cache_key: Optional[str] = None, older_than_hours: Optional[int] = None):
        """
        Invalidate cache entries.

        Args:
            cache_key: Specific key to invalidate (if None, use older_than_hours)
            older_than_hours: Invalidate entries older than this (if cache_key is None)
        """
        with self._get_connection() as conn:
            if cache_key:
                conn.execute("DELETE FROM llm_cache WHERE cache_key = ?", (cache_key,))
            elif older_than_hours is not None:
                cutoff = time.time() - (older_than_hours * 3600)
                conn.execute("DELETE FROM llm_cache WHERE created_at < ?", (cutoff,))
            conn.commit()

    def clear(self):
        """Clear all cache entries."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM llm_cache")
            conn.commit()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(access_count) as total_accesses,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry
                FROM llm_cache
            """)
            row = cursor.fetchone()

            # Get model breakdown
            cursor = conn.execute("""
                SELECT model, COUNT(*) as count
                FROM llm_cache
                GROUP BY model
            """)
            models = {r['model']: r['count'] for r in cursor.fetchall()}

            return {
                'total_entries': row['total_entries'] or 0,
                'total_accesses': row['total_accesses'] or 0,
                'total_input_tokens': row['total_input_tokens'] or 0,
                'total_output_tokens': row['total_output_tokens'] or 0,
                'oldest_entry': row['oldest_entry'],
                'newest_entry': row['newest_entry'],
                'models': models,
                'cache_file': self.cache_path,
                'cache_size_mb': os.path.getsize(self.cache_path) / (1024 * 1024) if os.path.exists(self.cache_path) else 0,
            }


class CachedLLMClient(LLMClient):
    """
    LLM client wrapper that adds caching.

    Wraps any LLMClient and caches responses to reduce API calls.
    """

    def __init__(
        self,
        client: LLMClient,
        cache: Optional[ResponseCache] = None,
    ):
        """
        Initialize cached client wrapper.

        Args:
            client: The underlying LLM client
            cache: ResponseCache instance (creates default if None)
        """
        super().__init__(model=client.model)
        self._client = client
        self._cache = cache or ResponseCache()

        # Track statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
        }

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
        skip_cache: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion with caching.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            stop_sequences: Stop sequences
            skip_cache: If True, bypass cache for this request
            **kwargs: Additional parameters

        Returns:
            LLMResponse (may be cached)
        """
        self._stats['total_requests'] += 1

        # Check cache first (unless skipped)
        if not skip_cache:
            cached = self._cache.get(
                prompt=prompt,
                system=system,
                model=self._client.model,
                temperature=temperature
            )
            if cached:
                self._stats['hits'] += 1
                return cached

        # Cache miss - call the actual API
        self._stats['misses'] += 1

        response = self._client.complete(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs
        )

        # Store in cache
        self._cache.set(
            prompt=prompt,
            system=system,
            model=self._client.model,
            temperature=temperature,
            response=response
        )

        return response

    def estimate_tokens(self, text: str) -> int:
        """Delegate to underlying client."""
        return self._client.estimate_tokens(text)

    def get_model_info(self) -> ModelInfo:
        """Delegate to underlying client."""
        return self._client.get_model_info()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get caching statistics.

        Returns:
            Dict with hit/miss stats and cache storage stats
        """
        hit_rate = 0.0
        if self._stats['total_requests'] > 0:
            hit_rate = self._stats['hits'] / self._stats['total_requests'] * 100

        return {
            **self._stats,
            'hit_rate_percent': round(hit_rate, 1),
            'cache_storage': self._cache.get_stats(),
        }

    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        self._stats = {'hits': 0, 'misses': 0, 'total_requests': 0}
