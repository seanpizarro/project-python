"""
Simple in-memory cache with TTL for market data
Avoids repeated API calls for data that doesn't change frequently
"""

import time
from typing import Any, Optional, Callable
from functools import wraps
import threading


class TTLCache:
    """Thread-safe cache with time-to-live expiration"""
    
    def __init__(self, default_ttl: int = 60):
        """
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self._cache = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    self.hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self._cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in cache with TTL"""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        with self._lock:
            self._cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        """Remove key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """Clear all cached data"""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> dict:
        """Return cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'size': len(self._cache)
        }


# Global cache instance
_cache = TTLCache(default_ttl=60)


def cached(ttl: int = 60, key_prefix: str = ""):
    """
    Decorator to cache function results
    
    Usage:
        @cached(ttl=30, key_prefix="vix")
        def get_vix():
            return fetch_vix_from_api()
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key from function name and arguments
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Check cache
            result = _cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            if result is not None:
                _cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def get_cache() -> TTLCache:
    """Get the global cache instance"""
    return _cache


def clear_cache() -> None:
    """Clear the global cache"""
    _cache.clear()

