"""Query caching for ast-grep MCP server using cachetools."""

import hashlib
import json
from typing import Any, Dict, List, Optional

from cachetools import TTLCache

from ast_grep_mcp.constants import CacheDefaults, FormattingDefaults


class QueryCache:
  """LRU cache with TTL for ast-grep query results using cachetools.

  Caches query results to avoid redundant ast-grep executions for identical queries.
  Uses cachetools.TTLCache for automatic LRU eviction and TTL expiration.
  """

  def __init__(
      self, max_size: int = CacheDefaults.DEFAULT_CACHE_SIZE, ttl_seconds: int = CacheDefaults.CLEANUP_INTERVAL_SECONDS
  ) -> None:
    """Initialize the cache.

    Args:
        max_size: Maximum number of entries to cache (default: CacheDefaults.DEFAULT_CACHE_SIZE)
        ttl_seconds: Time-to-live for cache entries in seconds (default: CacheDefaults.CLEANUP_INTERVAL_SECONDS)
    """
    self.max_size = max_size
    self.ttl_seconds = ttl_seconds
    self.cache: TTLCache[str, List[Dict[str, Any]]] = TTLCache(maxsize=max_size, ttl=ttl_seconds)
    self.hits = 0
    self.misses = 0

  def _make_key(self, command: str, args: List[str], project_folder: str) -> str:
    """Create a cache key from query parameters.

    Args:
        command: ast-grep command (run/scan)
        args: Command arguments
        project_folder: Project folder path

    Returns:
        Hash-based cache key
    """
    # Order matters (flag/value pairs) and values may contain any separator, so serialize as JSON.
    key_str = json.dumps([command, project_folder, *args])
    return hashlib.sha256(key_str.encode()).hexdigest()[: CacheDefaults.CACHE_KEY_LENGTH]

  def get(self, command: str, args: List[str], project_folder: str) -> Optional[List[Dict[str, Any]]]:
    """Get cached results if available and not expired.

    Args:
        command: ast-grep command (run/scan)
        args: Command arguments
        project_folder: Project folder path

    Returns:
        Cached results if found and valid, None otherwise
    """
    key = self._make_key(command, args, project_folder)

    if key not in self.cache:
      self.misses += 1
      return None

    results = self.cache[key]
    self.hits += 1
    return results

  def put(self, command: str, args: List[str], project_folder: str, results: List[Dict[str, Any]]) -> None:
    """Store results in cache.

    Args:
        command: ast-grep command (run/scan)
        args: Command arguments
        project_folder: Project folder path
        results: Query results to cache
    """
    key = self._make_key(command, args, project_folder)
    self.cache[key] = results

  def clear(self) -> None:
    """Clear all cache entries."""
    self.cache.clear()
    self.hits = 0
    self.misses = 0

  def get_stats(self) -> Dict[str, Any]:
    """Get cache statistics.

    Returns:
        Dictionary with cache stats
    """
    total = self.hits + self.misses
    hit_rate = self.hits / total if total > 0 else 0
    return {
        "size": len(self.cache),
        "max_size": self.max_size,
        "hits": self.hits,
        "misses": self.misses,
        "hit_rate": round(hit_rate, FormattingDefaults.ROUNDING_PRECISION),
        "ttl_seconds": self.ttl_seconds,
    }


_query_cache: Optional[QueryCache] = None


def get_query_cache() -> Optional[QueryCache]:
  """Get the global query cache instance if caching is enabled."""
  from ast_grep_mcp.core.config import CACHE_ENABLED

  return _query_cache if CACHE_ENABLED else None


def init_query_cache(max_size: int, ttl_seconds: int) -> None:
  """Initialize the global query cache.

  Args:
      max_size: Maximum number of entries to cache
      ttl_seconds: Time-to-live for cache entries in seconds
  """
  global _query_cache
  _query_cache = QueryCache(max_size=max_size, ttl_seconds=ttl_seconds)
