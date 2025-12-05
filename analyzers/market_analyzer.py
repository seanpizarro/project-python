"""Market Analysis - VIX Term Structure, Put/Call Skew, IV Analysis"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta


class MarketAnalyzer:
    """Advanced market analysis for options trading"""
    
    def __init__(self):
        self._yf = None
    
    @property
    def yf(self):
        """Lazy load yfinance"""
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf
    
    def get_vix_data(self) -> Dict:
        """Get VIX and related volatility indices"""
        try:
            vix = self.yf.Ticker("^VIX")
            vix_hist = vix.history(period="1y")
            
            if vix_hist.empty:
                return self._default_vix_data()
            
            current_vix = float(vix_hist['Close'].iloc[-1])
            vix_52w_high = float(vix_hist['High'].max())
            vix_52w_low = float(vix_hist['Low'].min())
            vix_mean = float(vix_hist['Close'].mean())
            
            vix_percentile = ((current_vix - vix_52w_low) / (vix_52w_high - vix_52w_low)) * 100
            
            try:
                vix3m = self.yf.Ticker("^VIX3M")
                vix3m_hist = vix3m.history(period="5d")
                current_vix3m = float(vix3m_hist['Close'].iloc[-1]) if not vix3m_hist.empty else current_vix
            except Exception:
                current_vix3m = current_vix * 1.05
            
            try:
                vix9d = self.yf.Ticker("^VIX9D")
                vix9d_hist = vix9d.history(period="5d")
                current_vix9d = float(vix9d_hist['Close'].iloc[-1]) if not vix9d_hist.empty else current_vix
            except Exception:
                current_vix9d = current_vix * 0.95
            
            return {
                'vix': round(current_vix, 2),
                'vix_9d': round(current_vix9d, 2),
                'vix_3m': round(current_vix3m, 2),
                'vix_52w_high': round(vix_52w_high, 2),
                'vix_52w_low': round(vix_52w_low, 2),
                'vix_mean': round(vix_mean, 2),
                'vix_percentile': round(vix_percentile, 1)
            }
            
        except Exception as e:
            print(f"  ⚠️  VIX fetch error: {e}")
            return self._default_vix_data()
    
    def _default_vix_data(self) -> Dict:
        return {
            'vix': 16.0,
            'vix_9d': 15.0,
            'vix_3m': 17.0,
            'vix_52w_high': 30.0,
            'vix_52w_low': 12.0,
            'vix_mean': 18.0,
            'vix_percentile': 25.0
        }
    
    def analyze_term_structure(self, vix_data: Dict = None) -> Dict:
        """Analyze VIX term structure"""
        if vix_data is None:
            vix_data = self.get_vix_data()
        
        vix = vix_data['vix']
        vix_9d = vix_data.get('vix_9d', vix)
        vix_3m = vix_data.get('vix_3m', vix)
        
        short_term_slope = (vix - vix_9d) / vix_9d * 100 if vix_9d > 0 else 0
        long_term_slope = (vix_3m - vix) / vix * 100 if vix > 0 else 0
        
        if vix_3m > vix > vix_9d:
            structure = "normal_contango"
            description = "Normal contango - longer-dated vol higher"
        elif vix_3m < vix < vix_9d:
            structure = "backwardation"
            description = "Backwardation - near-term fear elevated"
        elif vix > vix_3m and vix > vix_9d:
            structure = "inverted"
            description = "Inverted - current vol spike"
        else:
            structure = "flat"
            description = "Flat term structure"
        
        if structure == "normal_contango":
            implication = "Favorable for premium selling"
        elif structure == "backwardation":
            implication = "Caution for shorts, consider hedging"
        elif structure == "inverted":
            implication = "High fear, wait for normalization"
        else:
            implication = "Neutral environment"
        
        return {
            'structure': structure,
            'description': description,
            'implication': implication,
            'short_term_slope': round(short_term_slope, 2),
            'long_term_slope': round(long_term_slope, 2),
            'vix_9d': vix_9d,
            'vix': vix,
            'vix_3m': vix_3m
        }
    
    def calculate_iv_rank(self, symbol: str, current_iv: float = None) -> Dict:
        """Calculate IV Rank and Percentile from historical data"""
        try:
            ticker = self.yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return {'iv_rank': 50, 'iv_percentile': 50, 'hv_30': 0.20}
            
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            hv_30 = float(returns.tail(30).std() * np.sqrt(252))
            
            rolling_hv = returns.rolling(window=30).std() * np.sqrt(252)
            rolling_hv = rolling_hv.dropna()
            
            if len(rolling_hv) < 10:
                return {'iv_rank': 50, 'iv_percentile': 50, 'hv_30': hv_30}
            
            current = current_iv if current_iv else hv_30
            
            hv_high = float(rolling_hv.max())
            hv_low = float(rolling_hv.min())
            
            if hv_high > hv_low:
                iv_rank = ((current - hv_low) / (hv_high - hv_low)) * 100
            else:
                iv_rank = 50
            
            iv_percentile = (rolling_hv < current).mean() * 100
            
            return {
                'iv_rank': round(min(max(iv_rank, 0), 100), 1),
                'iv_percentile': round(min(max(iv_percentile, 0), 100), 1),
                'hv_30': round(hv_30, 4),
                'hv_52w_high': round(hv_high, 4),
                'hv_52w_low': round(hv_low, 4)
            }
            
        except Exception as e:
            print(f"  ⚠️  IV rank calculation error: {e}")
            return {'iv_rank': 50, 'iv_percentile': 50, 'hv_30': 0.20}
    
    def calculate_put_call_skew(
        self,
        options_chain: Dict = None,
        atm_strike: float = None,
        positions: List[Dict] = None
    ) -> Dict:
        """Calculate put/call skew"""
        if positions:
            return self._skew_from_positions(positions)
        return self._default_skew()
    
    def _skew_from_positions(self, positions: List[Dict]) -> Dict:
        put_ivs = []
        call_ivs = []
        
        for pos in positions:
            iv = pos.get('iv')
            if iv:
                if pos['type'] == 'put':
                    put_ivs.append(iv)
                else:
                    call_ivs.append(iv)
        
        if put_ivs and call_ivs:
            avg_put_iv = np.mean(put_ivs)
            avg_call_iv = np.mean(call_ivs)
            skew = (avg_put_iv - avg_call_iv) * 100
            
            if skew < -5:
                description = "Strong put skew (fear premium)"
            elif skew < -2:
                description = "Normal put skew"
            elif skew < 2:
                description = "Neutral skew"
            else:
                description = "Call skew (unusual)"
            
            return {
                'skew': round(skew, 2),
                'put_iv': round(avg_put_iv * 100, 1),
                'call_iv': round(avg_call_iv * 100, 1),
                'description': description
            }
        
        return self._default_skew()
    
    def _default_skew(self) -> Dict:
        return {
            'skew': -3.0,
            'put_iv': 25.0,
            'call_iv': 22.0,
            'description': "Estimated normal put skew"
        }
    
    def get_earnings_info(self, symbol: str) -> Dict:
        """Get earnings date - returns None for ETFs (they don't have earnings)"""
        try:
            # Suppress yfinance HTTP errors for ETFs
            import warnings
            import logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            
            ticker = self.yf.Ticker(symbol)
            
            # Check if it's an ETF (no earnings)
            info = ticker.info or {}
            if info.get('quoteType') == 'ETF':
                return {'earnings_date': None, 'days_to_earnings': None, 'is_etf': True}
            
            calendar = ticker.calendar
            
            if calendar is None:
                return {'earnings_date': None, 'days_to_earnings': None}
            
            if hasattr(calendar, 'empty') and calendar.empty:
                return {'earnings_date': None, 'days_to_earnings': None}
            
            if isinstance(calendar, dict) and not calendar:
                return {'earnings_date': None, 'days_to_earnings': None}
            
            if isinstance(calendar, dict):
                earnings_dates = calendar.get('Earnings Date') or calendar.get('earningsDate')
                if earnings_dates:
                    if isinstance(earnings_dates, (list, tuple)) and len(earnings_dates) > 0:
                        next_earnings = earnings_dates[0]
                    else:
                        next_earnings = earnings_dates
                    
                    if next_earnings:
                        if hasattr(next_earnings, 'date'):
                            next_earnings = next_earnings.date()
                        elif isinstance(next_earnings, str):
                            try:
                                next_earnings = datetime.strptime(next_earnings[:10], '%Y-%m-%d').date()
                            except ValueError:
                                return {'earnings_date': None, 'days_to_earnings': None}
                        
                        days_to = (next_earnings - datetime.now().date()).days
                        
                        return {
                            'earnings_date': str(next_earnings),
                            'days_to_earnings': days_to,
                            'earnings_before_expiry': days_to > 0
                        }
            
            elif hasattr(calendar, 'index') and 'Earnings Date' in calendar.index:
                earnings_dates = calendar.loc['Earnings Date']
                if isinstance(earnings_dates, (list, np.ndarray)) and len(earnings_dates) > 0:
                    next_earnings = earnings_dates[0]
                else:
                    next_earnings = earnings_dates
                
                if next_earnings:
                    if hasattr(next_earnings, 'date'):
                        next_earnings = next_earnings.date()
                    elif isinstance(next_earnings, str):
                        next_earnings = datetime.strptime(next_earnings, '%Y-%m-%d').date()
                    
                    days_to = (next_earnings - datetime.now().date()).days
                    
                    return {
                        'earnings_date': str(next_earnings),
                        'days_to_earnings': days_to,
                        'earnings_before_expiry': days_to > 0
                    }
            
            return {'earnings_date': None, 'days_to_earnings': None}
            
        except Exception as e:
            return {'earnings_date': None, 'days_to_earnings': None}
    
    def _determine_vol_trend(self, vix_data: Dict) -> str:
        vix = vix_data['vix']
        vix_mean = vix_data['vix_mean']
        
        if vix < vix_mean * 0.8:
            return "low"
        elif vix > vix_mean * 1.2:
            return "elevated"
        else:
            return "stable"

