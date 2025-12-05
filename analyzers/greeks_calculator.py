"""Calculate Greeks - with implied volatility solver"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import List, Dict, Optional


class GreeksCalculator:
    """Black-Scholes Greeks with IV solver"""
    
    def __init__(self, broker_client):
        self.client = broker_client
    
    def enrich_positions(self, positions: List[Dict], market_data: Dict) -> List[Dict]:
        """Add Greeks to positions - calculates IV from option prices"""
        
        enriched = []
        for pos in positions:
            pos_copy = pos.copy()
            
            # Try to calculate implied volatility from option price
            iv = self._calculate_implied_vol(pos, market_data)
            
            if iv and iv > 0.01:
                # We have a real IV - calculate Greeks
                greeks = self._calculate_bs(pos, market_data, iv)
                pos_copy.update(greeks)
                pos_copy['iv_source'] = 'calculated_from_price'
            else:
                # No IV available - mark Greeks as unavailable
                pos_copy['delta'] = None
                pos_copy['gamma'] = None
                pos_copy['theta'] = None
                pos_copy['vega'] = None
                pos_copy['iv'] = None
                pos_copy['iv_source'] = 'unavailable'
            
            enriched.append(pos_copy)
        
        return enriched
    
    def _calculate_implied_vol(self, pos: Dict, market: Dict) -> Optional[float]:
        """
        Calculate implied volatility from option price using Brent's method
        Returns None if calculation fails
        """
        try:
            S = market.get('current_price')
            K = pos['strike']
            T = pos['dte'] / 365
            r = 0.05  # Risk-free rate assumption
            
            # Get option price (use mid of entry and current, or just current)
            option_price = pos.get('current_premium', 0)
            
            if not S or not option_price or T <= 0 or option_price <= 0:
                return None
            
            is_call = pos['type'] == 'call'
            
            # Define the objective function
            def objective(sigma):
                if sigma <= 0:
                    return float('inf')
                bs_price = self._bs_price(S, K, T, r, sigma, is_call)
                return bs_price - option_price
            
            # Check if solution exists in reasonable range
            try:
                # IV typically between 5% and 200%
                iv = brentq(objective, 0.01, 3.0, xtol=1e-6, maxiter=100)
                return round(iv, 4)
            except ValueError:
                # No solution in range - option might be deep ITM/OTM
                return None
                
        except Exception:
            return None
    
    def _bs_price(self, S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
        """Black-Scholes option price"""
        if T <= 0 or sigma <= 0:
            # At expiration, return intrinsic value
            if is_call:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if is_call:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def _calculate_bs(self, pos: Dict, market: Dict, sigma: float) -> Dict:
        """Black-Scholes Greeks calculation with given IV"""
        
        try:
            S = market['current_price']
            K = pos['strike']
            T = pos['dte'] / 365
            r = 0.05
            
            if T <= 0 or sigma <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'iv': sigma}
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if pos['type'] == 'call':
                delta = norm.cdf(d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                delta = -norm.cdf(-d1)
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            return {
                'delta': round(delta, 4),
                'gamma': round(gamma, 5),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'iv': round(sigma, 4)
            }
            
        except Exception:
            return {'delta': None, 'gamma': None, 'theta': None, 'vega': None, 'iv': None}
