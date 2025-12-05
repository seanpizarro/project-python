"""Market data fetcher"""

from typing import Dict


class MarketDataFetcher:
    """Fetch market data"""
    
    def fetch_all(self, symbol: str) -> Dict:
        """Get market data (defaults for now)"""
        
        return {
            'current_price': 450.0,  # Will be replaced by Alpaca
            'vix': 16.0,
            'iv_rank': 45,
            'iv_percentile': 48,
            'term_structure': 'normal_contango',
            'put_call_skew': -2.5,
            'vol_trend': 'stable',
            'earnings_in_dte': None
        }