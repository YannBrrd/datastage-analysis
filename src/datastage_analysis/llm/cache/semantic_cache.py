"""
Semantic Cache Implementation

Uses embeddings to find similar prompts and return cached responses.
This reduces API calls for near-duplicate job migrations.
"""

import json
import time
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from ..client import LLMResponse

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticCache:
    """
    Semantic similarity-based cache for LLM responses.

    Uses embeddings to find similar prompts and return cached responses
    when similarity exceeds threshold. This is especially useful for
    DataStage jobs that have similar structures.
    """

    def __init__(
        self,
        cache_path: Optional[str] = None,
        similarity_threshold: float = 0.95,
        ttl_hours: int = 168,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize semantic cache.

        Args:
            cache_path: Path to SQLite database
            similarity_threshold: Minimum similarity for cache hit (0.0-1.0)
            ttl_hours: Time-to-live for cache entries
            embedding_model: Model to use for embeddings
        """
        if cache_path is None:
            cache_dir = Path('.cache')
            cache_dir.mkdir(exist_ok=True)
            cache_path = str(cache_dir / 'semantic_cache.db')

        self.cache_path = cache_path
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_hours * 3600
        self.embedding_model = embedding_model
        self._embedding_client = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    response_content TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    finish_reason TEXT,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    similarity_hits INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_model
                ON semantic_cache(model)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_created
                ON semantic_cache(created_at)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.cache_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.

        Uses a simple hash-based embedding if no embedding API is available,
        or calls the embedding API if configured.
        """
        # Try to use OpenAI embeddings if available
        try:
            return self._get_openai_embedding(text)
        except Exception:
            pass

        # Fallback: simple hash-based pseudo-embedding
        return self._get_hash_embedding(text)

    def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI API."""
        import os

        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("No OpenAI API key")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            response = client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000],  # Limit text length
            )

            return response.data[0].embedding

        except ImportError:
            raise ValueError("OpenAI package not installed")

    def _get_hash_embedding(self, text: str, dimensions: int = 256) -> List[float]:
        """
        Generate pseudo-embedding using hash functions.

        This provides a deterministic embedding without requiring an API,
        useful for exact and near-exact matches based on content structure.
        """
        import struct

        # Normalize text
        normalized = ' '.join(text.lower().split())

        # Generate multiple hashes for different "dimensions"
        embedding = []
        for i in range(dimensions):
            # Create varied hash by adding dimension index
            hash_input = f"{normalized}:{i}".encode()
            hash_bytes = hashlib.sha256(hash_input).digest()

            # Convert first 4 bytes to float in [-1, 1]
            value = struct.unpack('I', hash_bytes[:4])[0]
            normalized_value = (value / (2**32)) * 2 - 1
            embedding.append(normalized_value)

        # Normalize to unit vector
        norm = sum(v**2 for v in embedding) ** 0.5
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    def get(
        self,
        prompt: str,
        model: str,
    ) -> Optional[Tuple[LLMResponse, float]]:
        """
        Find cached response for semantically similar prompt.

        Args:
            prompt: The prompt to search for
            model: Model identifier

        Returns:
            Tuple of (LLMResponse, similarity_score) if found, None otherwise
        """
        current_time = time.time()
        query_embedding = self._get_embedding(prompt)

        with self._get_connection() as conn:
            # Get all non-expired entries for this model
            cursor = conn.execute(
                """
                SELECT id, embedding, response_content, input_tokens, output_tokens,
                       finish_reason, metadata
                FROM semantic_cache
                WHERE model = ? AND created_at > ?
                """,
                (model, current_time - self.ttl_seconds)
            )

            best_match = None
            best_similarity = 0.0
            best_id = None

            for row in cursor:
                # Deserialize embedding
                cached_embedding = json.loads(row['embedding'])

                # Compute similarity
                similarity = cosine_similarity(query_embedding, cached_embedding)

                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = row
                    best_id = row['id']

            if best_match is None:
                return None

            # Update access stats
            conn.execute(
                """
                UPDATE semantic_cache
                SET last_accessed = ?, access_count = access_count + 1,
                    similarity_hits = similarity_hits + 1
                WHERE id = ?
                """,
                (current_time, best_id)
            )
            conn.commit()

            # Parse metadata
            metadata = {}
            if best_match['metadata']:
                try:
                    metadata = json.loads(best_match['metadata'])
                except json.JSONDecodeError:
                    pass

            metadata['semantic_similarity'] = best_similarity
            metadata['semantic_cache_hit'] = True

            logger.debug(f"Semantic cache hit with similarity {best_similarity:.3f}")

            response = LLMResponse(
                content=best_match['response_content'],
                model=model,
                input_tokens=best_match['input_tokens'],
                output_tokens=best_match['output_tokens'],
                cached=True,
                latency_ms=0.0,
                finish_reason=best_match['finish_reason'] or 'stop',
                metadata=metadata,
            )

            return response, best_similarity

    def set(
        self,
        prompt: str,
        model: str,
        response: LLMResponse
    ):
        """
        Store response with embedding for semantic search.

        Args:
            prompt: The prompt
            model: Model identifier
            response: LLM response to cache
        """
        embedding = self._get_embedding(prompt)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:32]
        current_time = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO semantic_cache
                (prompt_hash, model, embedding, response_content, input_tokens,
                 output_tokens, finish_reason, metadata, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt_hash,
                    model,
                    json.dumps(embedding),
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

        logger.debug(f"Stored response in semantic cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(access_count) as total_accesses,
                    SUM(similarity_hits) as total_similarity_hits,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens
                FROM semantic_cache
            """)
            row = cursor.fetchone()

            return {
                'total_entries': row['total_entries'] or 0,
                'total_accesses': row['total_accesses'] or 0,
                'total_similarity_hits': row['total_similarity_hits'] or 0,
                'total_input_tokens': row['total_input_tokens'] or 0,
                'total_output_tokens': row['total_output_tokens'] or 0,
                'similarity_threshold': self.similarity_threshold,
                'embedding_model': self.embedding_model,
            }

    def clear(self):
        """Clear all cache entries."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM semantic_cache")
            conn.commit()
