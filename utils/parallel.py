"""
Parallel execution utilities for faster API fetching
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any, Tuple
import time


def fetch_parallel(tasks: List[Tuple[str, Callable, tuple]], max_workers: int = 5, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute multiple tasks in parallel
    
    Args:
        tasks: List of (name, function, args) tuples
        max_workers: Maximum concurrent threads
        timeout: Timeout per task in seconds
    
    Returns:
        Dict mapping task names to results (or None if failed)
    
    Example:
        tasks = [
            ("alpaca", fetch_alpaca_balance, ()),
            ("tastytrade", fetch_tt_balance, ()),
            ("schwab", fetch_schwab_balance, ()),
        ]
        results = fetch_parallel(tasks)
        # results = {"alpaca": {...}, "tastytrade": {...}, "schwab": {...}}
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_name = {}
        for name, func, args in tasks:
            future = executor.submit(func, *args)
            future_to_name[future] = name
        
        # Collect results as they complete
        for future in as_completed(future_to_name, timeout=timeout):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {'error': str(e)}
    
    return results


def timed_fetch(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Execute function and return (result, elapsed_time)
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


class ParallelFetcher:
    """
    Reusable parallel fetcher for broker data
    """
    
    def __init__(self, max_workers: int = 6):
        self.max_workers = max_workers
        self.timings = {}
    
    def fetch_all_balances(self, brokers: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Fetch balances from all brokers in parallel
        
        Args:
            brokers: Dict of broker_name -> client_instance
        
        Returns:
            Dict of broker_name -> balance_dict
        """
        def get_balance(name, client):
            try:
                return client.get_account_balance()
            except Exception as e:
                return {'error': str(e)}
        
        tasks = [(name, lambda n=name, c=client: get_balance(n, c), ()) 
                 for name, client in brokers.items()]
        
        start = time.time()
        results = fetch_parallel(tasks, max_workers=self.max_workers)
        self.timings['balances'] = time.time() - start
        
        return results
    
    def fetch_all_positions(self, brokers: Dict[str, Any]) -> Dict[str, List]:
        """
        Fetch positions from all brokers in parallel
        """
        def get_positions(name, client):
            try:
                if hasattr(client, 'get_all_positions'):
                    return client.get_all_positions()
                elif hasattr(client, 'get_positions'):
                    return client.get_positions()
                return []
            except Exception as e:
                return []
        
        tasks = [(name, lambda n=name, c=client: get_positions(n, c), ()) 
                 for name, client in brokers.items()]
        
        start = time.time()
        results = fetch_parallel(tasks, max_workers=self.max_workers)
        self.timings['positions'] = time.time() - start
        
        return results
    
    def get_timings(self) -> Dict[str, float]:
        """Return timing information for last fetch operations"""
        return self.timings

