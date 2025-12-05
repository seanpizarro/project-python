"""TastyTrade Market Data Client - Direct API (no SDK)"""

import requests
from typing import Dict, List, Optional
from utils.helpers import safe_float


class TastyTradeDataClient:
    """
    TastyTrade client for market data (IV rank, metrics)
    Uses direct API calls (SDK has auth issues)
    """
    
    BASE_URL = "https://api.tastyworks.com"
    
    def __init__(self, username: str = None, password: str = None):
        self.username = username
        self.password = password
        self.session_token = None
        self.headers = {}
        self._authenticated = False
        
        if username and password:
            self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with TastyTrade via direct API"""
        try:
            url = f"{self.BASE_URL}/sessions"
            payload = {
                "login": self.username,
                "password": self.password,
                "remember-me": True
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            self.session_token = data['data']['session-token']
            self.headers = {
                'Authorization': self.session_token,
                'Content-Type': 'application/json'
            }
            self._authenticated = True
            return True
            
        except Exception as e:
            print(f"  ⚠️  TastyTrade auth failed: {e}")
            self._authenticated = False
            return False
    
    def get_market_metrics(self, symbol: str) -> Dict:
        """Get IV rank, IV percentile, earnings from TastyTrade"""
        if not self._authenticated:
            return {}
        
        try:
            url = f"{self.BASE_URL}/market-metrics"
            params = {'symbols': symbol}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('data', {}).get('items', [])
            
            if items:
                m = items[0]
                # TastyTrade returns decimals (0.12 = 12%)
                iv_rank_raw = safe_float(m.get('implied-volatility-index-rank'))
                iv_pct_raw = safe_float(m.get('implied-volatility-percentile'))
                
                return {
                    'iv_rank': round(iv_rank_raw * 100, 1) if iv_rank_raw else 0,
                    'iv_percentile': round(iv_pct_raw * 100, 1) if iv_pct_raw else 0,
                    'iv': safe_float(m.get('implied-volatility-index')),
                    'hv_30': safe_float(m.get('hv-30-implied-volatility')),
                    'hv_60': safe_float(m.get('hv-60-implied-volatility')),
                    'hv_90': safe_float(m.get('hv-90-implied-volatility')),
                    'earnings_expected_move': safe_float(m.get('earnings-expected-move')),
                    'earnings_date': m.get('next-earnings-date'),
                    'liquidity_rating': m.get('liquidity-rating'),
                    'source': 'tastytrade_exchange'
                }
                
        except Exception as e:
            print(f"  ⚠️  TastyTrade metrics error: {e}")
        
        return {}
    
    def get_stock_quote(self, symbol: str) -> Dict:
        """Get current price for underlying"""
        if not self._authenticated:
            return {}
        
        try:
            # TastyTrade uses instruments endpoint for equities
            url = f"{self.BASE_URL}/instruments/equities/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {'symbol': symbol, 'data': data.get('data', {})}
                
        except Exception:
            pass
        
        return {}
    
    def enrich_positions_with_greeks(self, positions: List[Dict]) -> List[Dict]:
        """
        TastyTrade doesn't provide Greeks via REST API.
        Greeks require the DXFeed streaming WebSocket which isn't available.
        Return positions unchanged - let calculator handle Greeks.
        """
        enriched = []
        for pos in positions:
            pos_copy = pos.copy()
            pos_copy['iv_source'] = 'tastytrade_no_streaming_greeks'
            enriched.append(pos_copy)
        return enriched
    
    def test_connection(self) -> bool:
        """Test if TastyTrade connection works"""
        if not self._authenticated:
            return False
        
        try:
            metrics = self.get_market_metrics('SPY')
            return bool(metrics)
        except Exception:
            return False
    
    def close(self):
        """Close session"""
        if self.session_token:
            try:
                url = f"{self.BASE_URL}/sessions"
                requests.delete(url, headers=self.headers, timeout=5)
            except Exception:
                pass
