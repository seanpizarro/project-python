"""TastyTrade API client with real Greeks and IV"""

import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from utils.helpers import safe_float


class TastyTradeClient:
    """TastyTrade API client for options data with real Greeks"""
    
    # API endpoints
    PROD_URL = "https://api.tastyworks.com"
    SANDBOX_URL = "https://api.cert.tastyworks.com"
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
        account_number: str = None,
        sandbox: bool = False,
        session_token: str = None
    ):
        self.username = username
        self.password = password
        self.account_number = account_number
        self.sandbox = sandbox
        self.base_url = self.SANDBOX_URL if sandbox else self.PROD_URL
        self.session_token = session_token
        self.headers = {}
        
        if not session_token and username and password:
            self._authenticate()
    
    def _authenticate(self) -> None:
        """Authenticate and get session token"""
        url = f"{self.base_url}/sessions"
        
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
        
        # Get accounts if not specified
        if not self.account_number:
            self._get_default_account()
    
    def _get_default_account(self) -> None:
        """Get first available account"""
        url = f"{self.base_url}/customers/me/accounts"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        accounts = data.get('data', {}).get('items', [])
        
        if accounts:
            self.account_number = accounts[0]['account']['account-number']
    
    def get_account_balance(self) -> Dict:
        """Get account balance information"""
        url = f"{self.base_url}/accounts/{self.account_number}/balances"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()['data']
        
        return {
            'equity': safe_float(data.get('net-liquidating-value', 0)),
            'cash': safe_float(data.get('cash-balance', 0)),
            'buying_power': safe_float(data.get('derivative-buying-power', 0)),
            'portfolio_value': safe_float(data.get('net-liquidating-value', 0)),
            'maintenance_margin': safe_float(data.get('maintenance-requirement', 0)),
            'pending_cash': safe_float(data.get('pending-cash', 0)),
            'currency': 'USD'
        }
    
    def get_positions(self) -> List[Dict]:
        """Get all positions with Greeks from TastyTrade"""
        url = f"{self.base_url}/accounts/{self.account_number}/positions"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        positions = data.get('data', {}).get('items', [])
        
        # Filter and enrich option positions
        option_positions = []
        for pos in positions:
            if pos.get('instrument-type') == 'Equity Option':
                enriched = self._parse_position(pos)
                if enriched:
                    option_positions.append(enriched)
        
        return option_positions
    
    def _parse_position(self, position: Dict) -> Dict:
        """Parse TastyTrade position into standard format"""
        symbol = position.get('symbol', '')
        underlying = position.get('underlying-symbol', '')
        
        # Parse OCC symbol for strike/expiry/type
        try:
            # OCC format: SYMBOL YYMMDD C/P STRIKE
            # Find where the date starts
            date_start = len(underlying)
            exp_str = symbol[date_start:date_start+6]
            option_type = symbol[date_start+6]
            strike_str = symbol[date_start+7:]
            
            exp_year = 2000 + int(exp_str[:2])
            exp_month = int(exp_str[2:4])
            exp_day = int(exp_str[4:6])
            expiration = datetime(exp_year, exp_month, exp_day)
            
            dte = max(0, (expiration - datetime.now()).days)
            strike = float(strike_str) / 1000
            
        except Exception:
            expiration = datetime.now()
            dte = 0
            strike = 0
            option_type = 'U'
        
        qty = safe_float(position.get('quantity', 0))
        
        return {
            'symbol': symbol,
            'underlying_symbol': underlying,
            'strike': strike,
            'type': 'call' if option_type == 'C' else 'put',
            'position': 'long' if qty > 0 else 'short',
            'qty': abs(qty),
            'entry_premium': safe_float(position.get('average-open-price', 0)),
            'current_premium': safe_float(position.get('close-price', 0)),
            'market_value': safe_float(position.get('market-value', 0)),
            'unrealized_pl': safe_float(position.get('realized-day-gain', 0)),
            'expiration': expiration.strftime('%Y-%m-%d'),
            'dte': dte,
            # Greeks from TastyTrade (will be filled by get_option_chain)
            'delta': None,
            'gamma': None,
            'theta': None,
            'vega': None,
            'iv': None
        }
    
    def get_option_chain(self, symbol: str) -> Dict:
        """Get option chain with Greeks and IV"""
        url = f"{self.base_url}/option-chains/{symbol}/nested"
        
        response = requests.get(url, headers=self.headers, timeout=15)
        response.raise_for_status()
        
        return response.json()
    
    def get_option_quote(self, option_symbol: str) -> Dict:
        """Get quote with Greeks for specific option"""
        url = f"{self.base_url}/market-data"
        
        params = {'symbols': option_symbol}
        response = requests.get(url, headers=self.headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get('data', {}).get('items', [])
        
        if items:
            quote = items[0]
            return {
                'bid': safe_float(quote.get('bid')),
                'ask': safe_float(quote.get('ask')),
                'last': safe_float(quote.get('last')),
                'delta': safe_float(quote.get('delta')),
                'gamma': safe_float(quote.get('gamma')),
                'theta': safe_float(quote.get('theta')),
                'vega': safe_float(quote.get('vega')),
                'iv': safe_float(quote.get('implied-volatility')),
                'volume': int(quote.get('volume', 0)),
                'open_interest': int(quote.get('open-interest', 0))
            }
        
        return {}
    
    def get_market_metrics(self, symbol: str) -> Dict:
        """Get IV rank, IV percentile, and other metrics"""
        url = f"{self.base_url}/market-metrics"
        
        params = {'symbols': symbol}
        response = requests.get(url, headers=self.headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get('data', {}).get('items', [])
        
        if items:
            metrics = items[0]
            return {
                'iv_rank': safe_float(metrics.get('implied-volatility-index-rank')),
                'iv_percentile': safe_float(metrics.get('implied-volatility-percentile')),
                'iv': safe_float(metrics.get('implied-volatility-index')),
                'hv_30': safe_float(metrics.get('hv-30-implied-volatility')),
                'hv_60': safe_float(metrics.get('hv-60-implied-volatility')),
                'hv_90': safe_float(metrics.get('hv-90-implied-volatility')),
                'earnings_expected_move': safe_float(metrics.get('earnings-expected-move')),
                'earnings_date': metrics.get('next-earnings-date'),
                'liquidity_rating': metrics.get('liquidity-rating'),
                'option_expiration_iv': safe_float(metrics.get('option-expiration-implied-volatility'))
            }
        
        return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for underlying"""
        url = f"{self.base_url}/market-data"
        
        params = {'symbols': symbol}
        response = requests.get(url, headers=self.headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get('data', {}).get('items', [])
        
        if items:
            quote = items[0]
            bid = safe_float(quote.get('bid'))
            ask = safe_float(quote.get('ask'))
            return (bid + ask) / 2 if bid and ask else safe_float(quote.get('last'))
        
        raise Exception(f"No quote for {symbol}")
    
    def enrich_positions_with_greeks(self, positions: List[Dict]) -> List[Dict]:
        """Fetch real Greeks from TastyTrade for each position"""
        enriched = []
        
        for pos in positions:
            try:
                quote = self.get_option_quote(pos['symbol'])
                pos_copy = pos.copy()
                
                if quote:
                    pos_copy['delta'] = quote.get('delta')
                    pos_copy['gamma'] = quote.get('gamma')
                    pos_copy['theta'] = quote.get('theta')
                    pos_copy['vega'] = quote.get('vega')
                    pos_copy['iv'] = quote.get('iv')
                    pos_copy['current_premium'] = quote.get('last') or quote.get('ask')
                
                enriched.append(pos_copy)
                
            except Exception as e:
                print(f"  ⚠️  Could not get Greeks for {pos['symbol']}: {e}")
                enriched.append(pos)
        
        return enriched
    
    def get_earnings_date(self, symbol: str) -> Optional[str]:
        """Get next earnings date for symbol"""
        metrics = self.get_market_metrics(symbol)
        return metrics.get('earnings_date')
    
    def close(self) -> None:
        """Close session"""
        if self.session_token:
            try:
                url = f"{self.base_url}/sessions"
                requests.delete(url, headers=self.headers, timeout=5)
            except Exception:
                pass

