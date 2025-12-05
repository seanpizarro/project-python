"""Calculate Greeks"""

import numpy as np
from scipy.stats import norm
from typing import List, Dict


class GreeksCalculator:
    """Black-Scholes Greeks"""
    
    def __init__(self, broker_client):
        self.client = broker_client
    
    def enrich_positions(self, positions: List[Dict], market_data: Dict) -> List[Dict]:
        """Add Greeks to positions"""
        
        enriched = []
        for pos in positions:
            greeks = self._calculate_bs(pos, market_data)
            pos_copy = pos.copy()
            pos_copy.update(greeks)
            enriched.append(pos_copy)
        
        return enriched
    
    def _calculate_bs(self, pos: Dict, market: Dict) -> Dict:
        """Black-Scholes calculation"""
        
        try:
            S = market['current_price']
            K = pos['strike']
            T = pos['dte'] / 365
            r = 0.05
            sigma = 0.25
            
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
                'iv': sigma
            }
            
        except Exception:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}