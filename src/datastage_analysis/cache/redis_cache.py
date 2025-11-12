"""
Redis Cache Module

Provides caching functionality using Redis to avoid recomputation.
"""

import json
import logging
from typing import Any, Optional
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisCache:
    """Async Redis cache for storing computation results."""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 86400):
        self.redis_url = redis_url
        self.ttl = ttl  # Time to live in seconds (24 hours default)
        self.client: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis."""
        if self.client is None:
            self.client = redis.from_url(self.redis_url)
            try:
                await self.client.ping()
                logger.info("Connected to Redis")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.client = None

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            self.client = None

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            await self.connect()
            if not self.client:
                return None

        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.client:
            await self.connect()
            if not self.client:
                return False

        try:
            ttl_value = ttl or self.ttl
            json_value = json.dumps(value)
            await self.client.setex(key, ttl_value, json_value)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.client:
            await self.connect()
            if not self.client:
                return False

        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        if not self.client:
            await self.connect()
            if not self.client:
                return False

        try:
            await self.client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False