"""Detect strategy from legs"""

from typing import List, Dict


class StrategyDetector:
    """Identify options strategy"""
    
    def __init__(self, positions: List[Dict]):
        self.positions = positions
        self.underlying = positions[0]['underlying_symbol'] if positions else None
    
    def detect_strategy(self) -> Dict:
        """Main detection"""
        
        if not self.positions:
            raise ValueError("No positions")
        
        # Validate same underlying
        underlyings = set(p['underlying_symbol'] for p in self.positions)
        if len(underlyings) > 1:
            raise ValueError(f"Multiple underlyings: {underlyings}")
        
        # Analyze
        legs = self._analyze_legs()
        net_credit = self._calculate_net_credit()
        current_value = self._calculate_current_value()
        current_pnl = net_credit - current_value
        
        strategy_name = self._identify_strategy(legs)
        max_profit, max_loss = self._calculate_max_profit_loss()
        breakevens = self._calculate_breakevens(net_credit)
        position_greeks = self._aggregate_greeks()
        
        dte = min(p['dte'] for p in self.positions if p.get('dte') is not None)
        
        return {
            'strategy': strategy_name,
            'dte': dte,
            'entry_date': self._estimate_entry_date(),
            'net_credit': round(net_credit, 2),
            'current_value': round(current_value, 2),
            'current_pnl': round(current_pnl, 2),
            'max_profit': round(max_profit, 2),
            'max_loss': round(max_loss, 2),
            'breakeven_lower': breakevens.get('lower'),
            'breakeven_upper': breakevens.get('upper'),
            'position_greeks': position_greeks,
            'legs_count': len(self.positions)
        }
    
    def _analyze_legs(self) -> Dict:
        """Group legs"""
        legs = {
            'calls': {'long': [], 'short': []},
            'puts': {'long': [], 'short': []}
        }
        
        for p in self.positions:
            legs[f"{p['type']}s"][p['position']].append(p)
        
        return legs
    
    def _identify_strategy(self, legs: Dict) -> str:
        """Identify strategy pattern"""
        
        n_short_puts = len(legs['puts']['short'])
        n_long_puts = len(legs['puts']['long'])
        n_short_calls = len(legs['calls']['short'])
        n_long_calls = len(legs['calls']['long'])
        
        total = n_short_puts + n_long_puts + n_short_calls + n_long_calls
        
        # Iron Condor/Butterfly
        if total == 4 and n_short_puts == 1 and n_long_puts == 1 and \
           n_short_calls == 1 and n_long_calls == 1:
            short_put = legs['puts']['short'][0]['strike']
            short_call = legs['calls']['short'][0]['strike']
            if abs(short_put - short_call) < 2:
                return "Iron Butterfly"
            return "Iron Condor"
        
        # Spreads
        if total == 2:
            if n_short_puts == 1 and n_long_puts == 1:
                if legs['puts']['short'][0]['strike'] > legs['puts']['long'][0]['strike']:
                    return "Bull Put Spread"
                return "Bear Put Spread"
            
            if n_short_calls == 1 and n_long_calls == 1:
                if legs['calls']['long'][0]['strike'] < legs['calls']['short'][0]['strike']:
                    return "Bull Call Spread"
                return "Bear Call Spread"
        
        return f"Custom ({total} legs)"
    
    def _calculate_net_credit(self) -> float:
        """Initial credit"""
        credit = 0
        for p in self.positions:
            amt = p['entry_premium'] * p['qty'] * 100
            credit += amt if p['position'] == 'short' else -amt
        return credit
    
    def _calculate_current_value(self) -> float:
        """Current value"""
        value = 0
        for p in self.positions:
            amt = p['current_premium'] * p['qty'] * 100
            value += amt if p['position'] == 'short' else -amt
        return abs(value)
    
    def _calculate_max_profit_loss(self) -> tuple:
        """Max profit/loss"""
        net_credit = self._calculate_net_credit()
        strikes = [p['strike'] for p in self.positions]
        
        if len(strikes) >= 2:
            width = max(strikes) - min(strikes)
            max_loss = (width * 100) - net_credit
        else:
            max_loss = net_credit
        
        return max(net_credit, 0), max(max_loss, 0)
    
    def _calculate_breakevens(self, net_credit: float) -> Dict:
        """Breakeven points"""
        credit_per = net_credit / 100
        short_strikes = [p['strike'] for p in self.positions if p['position'] == 'short']
        
        if not short_strikes:
            return {}
        
        breakevens = {}
        
        # Put breakeven
        put_shorts = [s for s in short_strikes if any(
            p['strike'] == s and p['type'] == 'put' for p in self.positions
        )]
        if put_shorts:
            breakevens['lower'] = round(max(put_shorts) - credit_per, 2)
        
        # Call breakeven
        call_shorts = [s for s in short_strikes if any(
            p['strike'] == s and p['type'] == 'call' for p in self.positions
        )]
        if call_shorts:
            breakevens['upper'] = round(min(call_shorts) + credit_per, 2)
        
        return breakevens
    
    def _aggregate_greeks(self) -> Dict:
        """Sum Greeks"""
        delta = gamma = theta = vega = 0
        
        for p in self.positions:
            mult = 1 if p['position'] == 'long' else -1
            qty = p['qty']
            
            if p.get('delta'): delta += p['delta'] * qty * mult
            if p.get('gamma'): gamma += p['gamma'] * qty * mult
            if p.get('theta'): theta += p['theta'] * qty * mult
            if p.get('vega'): vega += p['vega'] * qty * mult
        
        return {
            'position_delta': round(delta, 3),
            'position_gamma': round(gamma, 4),
            'position_theta': round(theta, 3),
            'position_vega': round(vega, 3)
        }
    
    def _estimate_entry_date(self) -> str:
        """Return unknown - we don't have real entry date from broker"""
        # NOTE: Alpaca API doesn't provide entry date
        # TastyTrade API might - check there if available
        return None  # Don't fabricate dates