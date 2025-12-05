"""TastyTrade Market Data Client - Using official tastytrade SDK"""

from typing import Dict, List, Optional
import getpass


class TastyTradeDataClient:
    """
    TastyTrade client for market data (Greeks, IV, metrics)
    Uses official tastytrade SDK for reliable data
    """
    
    def __init__(self, username: str = None, password: str = None):
        self.username = username
        self.password = password
        self.session = None
        self._authenticated = False
        
        if username and password:
            self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with TastyTrade"""
        try:
            from tastytrade import Session
            self.session = Session(self.username, self.password)
            self._authenticated = True
            return True
        except Exception as e:
            print(f"  ⚠️  TastyTrade auth failed: {e}")
            self._authenticated = False
            return False
    
    def ensure_authenticated(self) -> bool:
        """Ensure we have a valid session"""
        if self._authenticated and self.session:
            return True
        
        if not self.username or not self.password:
            print("\n🔐 TastyTrade credentials needed for real Greeks")
            self.username = input("   Email: ").strip()
            self.password = getpass.getpass("   Password: ")
        
        return self._authenticate()
    
    def get_market_metrics(self, symbol: str) -> Dict:
        """Get IV rank, IV percentile, earnings from TastyTrade"""
        if not self.ensure_authenticated():
            return {}
        
        try:
            from tastytrade.instruments import get_option_chain
            from tastytrade.market_metrics import get_market_metrics
            
            metrics_list = get_market_metrics(self.session, [symbol])
            
            if metrics_list:
                m = metrics_list[0]
                # Convert decimals to percentages
                iv_rank = (m.implied_volatility_index_rank or 0) * 100
                iv_pct = (m.implied_volatility_percentile or 0) * 100
                
                return {
                    'iv_rank': round(iv_rank, 1),
                    'iv_percentile': round(iv_pct, 1),
                    'iv': m.implied_volatility_index,
                    'hv_30': m.hv_30_implied_volatility,
                    'earnings_expected_move': m.earnings_expected_move,
                    'earnings_date': str(m.next_earnings_date) if m.next_earnings_date else None,
                    'liquidity_rating': m.liquidity_rating,
                    'source': 'tastytrade_exchange'
                }
        except Exception as e:
            print(f"  ⚠️  TastyTrade metrics error: {e}")
        
        return {}
    
    def get_option_greeks(self, option_symbols: List[str]) -> Dict[str, Dict]:
        """
        Get real Greeks for option symbols from TastyTrade
        
        Args:
            option_symbols: List of OCC format symbols
        
        Returns:
            Dict mapping symbol -> {delta, gamma, theta, vega, iv}
        """
        if not self.ensure_authenticated():
            return {}
        
        try:
            from tastytrade.dxfeed import Greeks
            from tastytrade.instruments import EquityOption
            
            # Convert OCC symbols to streamer symbols
            streamer_symbols = []
            symbol_map = {}
            
            for sym in option_symbols:
                # OCC: SPY   251225C00700000 -> .SPY251225C700
                # Remove spaces and format
                clean = sym.replace(' ', '')
                # Extract parts
                underlying = ''
                for i, c in enumerate(clean):
                    if c.isdigit():
                        underlying = clean[:i]
                        rest = clean[i:]
                        break
                
                if underlying:
                    date = rest[:6]
                    opt_type = rest[6]
                    strike_raw = rest[7:]
                    strike = str(int(int(strike_raw) / 1000))
                    streamer = f'.{underlying}{date}{opt_type}{strike}'
                    streamer_symbols.append(streamer)
                    symbol_map[streamer] = sym
            
            # Get Greeks via DXFeed
            # Note: This requires the streaming API
            # For now, return empty and fall back to calculated
            return {}
            
        except Exception as e:
            print(f"  ⚠️  TastyTrade Greeks error: {e}")
        
        return {}
    
    def enrich_positions_with_greeks(self, positions: List[Dict]) -> List[Dict]:
        """
        Fetch real Greeks from TastyTrade for positions
        
        Note: The streaming API is needed for real-time Greeks.
        This method will attempt to get them, or fall back gracefully.
        """
        if not self.ensure_authenticated():
            # Return positions unchanged, let calculator handle it
            return positions
        
        enriched = []
        symbols = [p.get('symbol', '') for p in positions]
        
        # Try to get Greeks
        greeks_data = self.get_option_greeks(symbols)
        
        for pos in positions:
            pos_copy = pos.copy()
            sym = pos.get('symbol', '')
            
            if sym in greeks_data:
                g = greeks_data[sym]
                pos_copy['delta'] = g.get('delta')
                pos_copy['gamma'] = g.get('gamma')
                pos_copy['theta'] = g.get('theta')
                pos_copy['vega'] = g.get('vega')
                pos_copy['iv'] = g.get('iv')
                pos_copy['iv_source'] = 'tastytrade_exchange'
            else:
                pos_copy['iv_source'] = 'unavailable_from_tastytrade'
            
            enriched.append(pos_copy)
        
        return enriched
    
    def test_connection(self) -> bool:
        """Test if TastyTrade connection works"""
        if not self.ensure_authenticated():
            return False
        
        try:
            metrics = self.get_market_metrics('SPY')
            return bool(metrics)
        except Exception:
            return False
    
    def close(self):
        """Close session"""
        if self.session:
            try:
                self.session.destroy()
            except Exception:
                pass
