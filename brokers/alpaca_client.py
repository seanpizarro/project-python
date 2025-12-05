"""Alpaca API client"""

import requests
from typing import List, Dict
from datetime import datetime
from utils.helpers import retry_on_failure, safe_float


class AlpacaClient:
    """Alpaca brokerage API client"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        
        self.headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key
        }
    
    @retry_on_failure(max_attempts=3, delay=2.0)
    def get_all_positions(self) -> List[Dict]:
        """Get all open positions"""
        url = f"{self.base_url}/v2/positions"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        positions = response.json()
        
        if not positions:
            return []
        
        # Filter options only
        option_positions = [p for p in positions if p.get('asset_class') == 'us_option']
        
        if not option_positions:
            print("\n  ℹ️  No option positions found")
            return []
        
        # Parse each position
        enriched = []
        for pos in option_positions:
            try:
                enriched.append(self._parse_option_position(pos))
            except Exception as e:
                print(f"  ⚠️  Skipping {pos.get('symbol')}: {e}")
        
        return enriched
    
    def get_positions_by_symbol(self, symbol: str) -> List[Dict]:
        """Get positions for specific underlying"""
        all_pos = self.get_all_positions()
        return [p for p in all_pos if p['underlying_symbol'] == symbol]
    
    def _parse_option_position(self, position: Dict) -> Dict:
        """Parse Alpaca option symbol (OCC format)"""
        symbol = position['symbol']
        
        try:
            # Find where date starts
            date_start = next(i for i, c in enumerate(symbol) if c.isdigit())
            
            underlying = symbol[:date_start]
            exp_date = symbol[date_start:date_start+6]  # YYMMDD
            option_type = symbol[date_start+6]          # C or P
            strike_raw = symbol[date_start+7:]          # Strike*1000
            
            # Parse expiration
            exp_year = 2000 + int(exp_date[:2])
            exp_month = int(exp_date[2:4])
            exp_day = int(exp_date[4:6])
            expiration = datetime(exp_year, exp_month, exp_day)
            
            # Calculate DTE
            dte = max(0, (expiration - datetime.now()).days)
            
            # Parse strike
            strike = float(strike_raw) / 1000
            
        except Exception as e:
            print(f"  ⚠️  Parse warning for {symbol}: {e}")
            underlying = symbol[:3]
            option_type = 'U'
            strike = 0
            dte = 0
            expiration = datetime.now()
        
        # Long or short
        qty = safe_float(position.get('qty', 0))
        pos_type = 'long' if qty > 0 else 'short'
        
        return {
            'symbol': symbol,
            'underlying_symbol': underlying,
            'strike': strike,
            'type': 'call' if option_type == 'C' else 'put',
            'position': pos_type,
            'qty': abs(qty),
            'entry_premium': safe_float(position.get('avg_entry_price')),
            'current_premium': safe_float(position.get('current_price')),
            'market_value': safe_float(position.get('market_value')),
            'unrealized_pl': safe_float(position.get('unrealized_pl')),
            'expiration': expiration.strftime('%Y-%m-%d'),
            'dte': dte,
            'delta': None,
            'gamma': None,
            'theta': None,
            'vega': None,
            'iv': None
        }
    
    @retry_on_failure(max_attempts=3, delay=1.0)
    def get_account(self) -> Dict:
        """Get account information including balance"""
        url = f"{self.base_url}/v2/account"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def get_account_balance(self) -> Dict:
        """Get account balance information"""
        account = self.get_account()
        
        return {
            'equity': safe_float(account.get('equity', 0)),
            'cash': safe_float(account.get('cash', 0)),
            'buying_power': safe_float(account.get('buying_power', 0)),
            'portfolio_value': safe_float(account.get('portfolio_value', 0)),
            'day_trading_buying_power': safe_float(account.get('daytrading_buying_power', 0)),
            'pattern_day_trader': account.get('pattern_day_trader', False),
            'trading_blocked': account.get('trading_blocked', False),
            'account_blocked': account.get('account_blocked', False),
            'currency': account.get('currency', 'USD')
        }
    
    @retry_on_failure(max_attempts=3, delay=1.0)
    def get_current_price(self, symbol: str) -> float:
        """Get current price for underlying"""
        url = f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"
        
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'quote' in data:
            bid = safe_float(data['quote'].get('bp'))
            ask = safe_float(data['quote'].get('ap'))
            return (bid + ask) / 2 if bid and ask else ask
        
        raise Exception(f"No quote for {symbol}")