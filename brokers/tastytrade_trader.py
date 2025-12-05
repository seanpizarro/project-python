"""TastyTrade Trading Client - For sandbox and live trading"""

import requests
from typing import Dict, List, Optional
from datetime import datetime
from utils.helpers import safe_float


class TastyTradeTrader:
    """
    TastyTrade trading client for placing orders
    Supports both sandbox (cert) and production environments
    """
    
    PROD_URL = "https://api.tastyworks.com"
    SANDBOX_URL = "https://api.cert.tastyworks.com"
    
    def __init__(
        self,
        username: str,
        password: str,
        sandbox: bool = True,
        account_number: str = None
    ):
        self.username = username
        self.password = password
        self.sandbox = sandbox
        self.base_url = self.SANDBOX_URL if sandbox else self.PROD_URL
        self.account_number = account_number
        self.session_token = None
        self.headers = {}
        self._authenticated = False
        
        self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with TastyTrade"""
        try:
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
            self._authenticated = True
            
            # Get account if not specified
            if not self.account_number:
                self._get_default_account()
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  TastyTrade auth failed: {e}")
            self._authenticated = False
            return False
    
    def _get_default_account(self) -> None:
        """Get first available account"""
        try:
            url = f"{self.base_url}/customers/me/accounts"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            accounts = data.get('data', {}).get('items', [])
            
            if accounts:
                self.account_number = accounts[0]['account']['account-number']
                print(f"      Using TastyTrade account: {self.account_number}")
        except Exception as e:
            print(f"  ⚠️  Could not get TastyTrade accounts: {e}")
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        if not self._authenticated:
            return {}
        
        try:
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
                'source': 'tastytrade_sandbox' if self.sandbox else 'tastytrade_live'
            }
        except Exception as e:
            print(f"  ⚠️  Balance error: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get all option positions"""
        if not self._authenticated:
            return []
        
        try:
            url = f"{self.base_url}/accounts/{self.account_number}/positions"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            positions = data.get('data', {}).get('items', [])
            
            # Filter and parse option positions
            option_positions = []
            for pos in positions:
                if pos.get('instrument-type') == 'Equity Option':
                    parsed = self._parse_position(pos)
                    if parsed:
                        option_positions.append(parsed)
            
            return option_positions
            
        except Exception as e:
            print(f"  ⚠️  Positions error: {e}")
            return []
    
    def _parse_position(self, pos: Dict) -> Dict:
        """Parse TastyTrade position to standard format"""
        symbol = pos.get('symbol', '').strip()
        underlying = pos.get('underlying-symbol', '')
        
        # Parse OCC symbol
        try:
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
        
        qty = safe_float(pos.get('quantity', 0))
        
        return {
            'symbol': symbol,
            'underlying_symbol': underlying,
            'strike': strike,
            'type': 'call' if option_type == 'C' else 'put',
            'position': 'long' if qty > 0 else 'short',
            'qty': abs(qty),
            'entry_premium': safe_float(pos.get('average-open-price', 0)),
            'current_premium': safe_float(pos.get('close-price', 0)),
            'market_value': safe_float(pos.get('market-value', 0)),
            'unrealized_pl': safe_float(pos.get('realized-day-gain', 0)),
            'expiration': expiration.strftime('%Y-%m-%d'),
            'dte': dte,
            'source': 'tastytrade'
        }
    
    def place_option_order(
        self,
        underlying: str,
        expiration: str,
        strike: float,
        option_type: str,
        action: str,
        quantity: int,
        order_type: str = 'Market',
        price: float = None
    ) -> Dict:
        """
        Place an option order
        
        Args:
            underlying: e.g., 'SPY'
            expiration: e.g., '2025-12-25'
            strike: e.g., 700.0
            option_type: 'call' or 'put'
            action: 'buy_to_open', 'sell_to_open', 'buy_to_close', 'sell_to_close'
            quantity: number of contracts
            order_type: 'Market' or 'Limit'
            price: limit price (required for Limit orders)
        
        Returns:
            Order response dict
        """
        if not self._authenticated:
            return {'error': 'Not authenticated'}
        
        try:
            # Build OCC symbol (TastyTrade format with padding)
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            exp_str = exp_date.strftime('%y%m%d')
            opt_char = 'C' if option_type.lower() == 'call' else 'P'
            strike_str = f"{int(strike * 1000):08d}"
            # TastyTrade requires underlying padded to 6 chars
            underlying_padded = underlying.ljust(6)
            occ_symbol = f"{underlying_padded}{exp_str}{opt_char}{strike_str}"
            
            # Determine price effect
            is_buy = 'buy' in action.lower()
            price_effect = 'Debit' if is_buy else 'Credit'
            
            # Format action: buy_to_open -> Buy to Open (lowercase 'to')
            action_map = {
                'buy_to_open': 'Buy to Open',
                'sell_to_open': 'Sell to Open',
                'buy_to_close': 'Buy to Close',
                'sell_to_close': 'Sell to Close'
            }
            action_formatted = action_map.get(action.lower(), action)
            
            # Build order
            order = {
                'time-in-force': 'Day',
                'order-type': order_type,
                'price-effect': price_effect,
                'legs': [{
                    'instrument-type': 'Equity Option',
                    'symbol': occ_symbol,
                    'action': action_formatted,
                    'quantity': quantity
                }]
            }
            
            if order_type == 'Limit' and price:
                order['price'] = str(price)
            
            url = f"{self.base_url}/accounts/{self.account_number}/orders"
            response = requests.post(url, headers=self.headers, json=order, timeout=15)
            
            if response.status_code in [200, 201]:
                return {
                    'success': True,
                    'order': response.json(),
                    'symbol': occ_symbol
                }
            else:
                return {
                    'success': False,
                    'error': response.text,
                    'status': response.status_code
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def place_spread_order(
        self,
        legs: List[Dict],
        order_type: str = 'Market',
        price: float = None
    ) -> Dict:
        """
        Place a multi-leg spread order
        
        Args:
            legs: List of leg dicts with keys:
                - underlying, expiration, strike, option_type, action, quantity
            order_type: 'Market' or 'Limit'
            price: net credit/debit for Limit orders
        
        Returns:
            Order response dict
        """
        if not self._authenticated:
            return {'error': 'Not authenticated'}
        
        try:
            order_legs = []
            
            for leg in legs:
                exp_date = datetime.strptime(leg['expiration'], '%Y-%m-%d')
                exp_str = exp_date.strftime('%y%m%d')
                opt_char = 'C' if leg['option_type'].lower() == 'call' else 'P'
                strike_str = f"{int(leg['strike'] * 1000):08d}"
                occ_symbol = f"{leg['underlying']}{exp_str}{opt_char}{strike_str}"
                
                order_legs.append({
                    'instrument-type': 'Equity Option',
                    'symbol': occ_symbol,
                    'action': leg['action'].replace('_', ' ').title().replace(' ', ' to '),
                    'quantity': leg['quantity']
                })
            
            order = {
                'time-in-force': 'Day',
                'order-type': order_type,
                'legs': order_legs
            }
            
            if order_type == 'Limit' and price is not None:
                order['price'] = str(price)
            
            url = f"{self.base_url}/accounts/{self.account_number}/orders"
            response = requests.post(url, headers=self.headers, json=order, timeout=15)
            
            if response.status_code in [200, 201]:
                return {
                    'success': True,
                    'order': response.json()
                }
            else:
                return {
                    'success': False,
                    'error': response.text,
                    'status': response.status_code
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_orders(self, status: str = None) -> List[Dict]:
        """Get orders, optionally filtered by status"""
        if not self._authenticated:
            return []
        
        try:
            url = f"{self.base_url}/accounts/{self.account_number}/orders"
            params = {}
            if status:
                params['status'] = status
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('data', {}).get('items', [])
            
        except Exception as e:
            print(f"  ⚠️  Orders error: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        if not self._authenticated:
            return {'error': 'Not authenticated'}
        
        try:
            url = f"{self.base_url}/accounts/{self.account_number}/orders/{order_id}"
            response = requests.delete(url, headers=self.headers, timeout=10)
            
            return {
                'success': response.status_code in [200, 204],
                'status': response.status_code
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Close session"""
        if self.session_token:
            try:
                url = f"{self.base_url}/sessions"
                requests.delete(url, headers=self.headers, timeout=5)
            except Exception:
                pass

