"""Helper utilities"""

import time
import functools
from typing import Callable, Any


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
                        print(f"  🔄 Retrying in {delay}s...")
                        time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default