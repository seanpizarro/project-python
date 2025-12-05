"""Format analysis output for Claude recommendations"""

from typing import Dict, List
from datetime import datetime


class ReportFormatter:
    """Format analysis data for display and Claude consumption"""
    
    @staticmethod
    def format_console_report(analysis: Dict) -> str:
        """
        Format analysis for console display
        Matches the mock-up format for Claude recommendations
        """
        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M EST")
        
        # Header
        lines.append("")
        lines.append("═" * 67)
        lines.append(f"RUN UPDATE – {timestamp}")
        lines.append("═" * 67)
        
        # Market Snapshot
        lines.append("")
        lines.append("MARKET SNAPSHOT")
        lines.append(f"Underlying: {analysis['underlying']} | Price: ${analysis['current_price']:.2f} | VIX: {analysis.get('vix', 'N/A')}")
        
        iv_rank = analysis.get('iv_rank', 0)
        term_struct = analysis.get('market_regime', {}).get('term_structure', 'unknown')
        skew = analysis.get('market_regime', {}).get('put_call_skew', 0)
        vol_trend = analysis.get('market_regime', {}).get('recent_volatility_trend', 'unknown')
        
        lines.append(f"IV Rank: {iv_rank:.0f}th percentile | Term Structure: {term_struct}")
        lines.append(f"Put/Call Skew: {skew:+.1f} | Vol Trend: {vol_trend.title()}")
        
        lines.append("─" * 67)
        
        # Position Info
        pos = analysis['position']
        lines.append("")
        lines.append(f"CURRENT POSITION: {pos['strategy']} {analysis['underlying']} ({pos['dte']} DTE)")
        entry_str = pos['entry_date'] if pos.get('entry_date') else "N/A (not from broker)"
        lines.append(f"Entry: {entry_str} | Net Credit: ${pos['net_credit']:.2f}")
        lines.append(f"Current Value: ${pos['current_value']:.2f} | P&L: ${pos['current_pnl']:+.2f}")
        
        # Max profit/loss
        lines.append(f"Max Profit: ${pos['max_profit']:.2f} | Max Loss: ${pos['max_loss']:.2f}")
        
        # Breakevens
        be_lower = pos.get('breakeven_lower')
        be_upper = pos.get('breakeven_upper')
        if be_lower and be_upper:
            lines.append(f"Breakevens: ${be_lower:.2f} (lower) | ${be_upper:.2f} (upper)")
        elif be_lower:
            lines.append(f"Breakeven: ${be_lower:.2f}")
        elif be_upper:
            lines.append(f"Breakeven: ${be_upper:.2f}")
        
        lines.append("")
        lines.append("Position Legs:")
        for leg in pos['legs']:
            delta_str = f"Δ{leg.get('delta', 0):+.2f}" if leg.get('delta') else ""
            iv_str = f"IV:{leg.get('iv', 0)*100:.0f}%" if leg.get('iv') else ""
            
            lines.append(
                f"  → {leg['position'].upper()} {leg['qty']:.0f}x {leg['type'].upper()} "
                f"${leg['strike']:.0f} @ ${leg['current_premium']:.2f} {delta_str} {iv_str}"
            )
        
        lines.append("")
        lines.append("Position Greeks:")
        greeks = analysis['greeks']
        lines.append(f"  → Delta: {greeks['position_delta']:+.3f} {'(neutral)' if abs(greeks['position_delta']) < 0.1 else ''}")
        lines.append(f"  → Gamma: {greeks['position_gamma']:+.4f}")
        lines.append(f"  → Theta: ${greeks['position_theta']*100:+.2f}/day")
        lines.append(f"  → Vega: {greeks['position_vega']:+.3f}")
        
        lines.append("─" * 67)
        
        # Monte Carlo (if available)
        if 'monte_carlo' in analysis:
            mc = analysis['monte_carlo']
            lines.append("")
            lines.append(f"MONTE CARLO ANALYSIS ({mc['paths']:,} paths, {mc['model']} model)")
            lines.append(f"Probability of Profit: {mc['pop']:.1f}%")
            lines.append(f"Probability of Touch: {mc['pot_lower']:.1f}% (lower) | {mc['pot_upper']:.1f}% (upper)")
            lines.append(f"Expected P&L: ${mc['expected_pl']:+.2f}")
            lines.append(f"Median Outcome: ${mc['median_pl']:+.2f}")
            lines.append("")
            lines.append("Risk Metrics:")
            lines.append(f"  → 95% VaR: ${mc['var_95']:.2f} (worst case in 95% of scenarios)")
            lines.append(f"  → 99% VaR: ${mc['var_99']:.2f} (extreme worst case)")
            lines.append(f"  → Expected Shortfall: ${mc['expected_shortfall_95']:.2f}")
            lines.append(f"Optimal Exit: {mc['optimal_exit_dte']} DTE")
            lines.append("─" * 67)
        
        # Market Regime Details
        regime = analysis.get('market_regime', {})
        if regime.get('earnings_date'):
            lines.append("")
            lines.append("⚠️  EARNINGS ALERT")
            lines.append(f"   Next earnings: {regime['earnings_date']}")
            days = regime.get('days_to_earnings')
            if days is not None:
                lines.append(f"   Days to earnings: {days}")
            lines.append("─" * 67)
        
        lines.append("")
        lines.append("═" * 67)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_json_for_claude(analysis: Dict) -> Dict:
        """
        Format analysis as clean JSON for Claude consumption
        Matches the mock-up JSON structure
        """
        pos = analysis['position']
        greeks = analysis['greeks']
        regime = analysis.get('market_regime', {})
        mc = analysis.get('monte_carlo', {})
        
        # Format legs without internal fields
        formatted_legs = []
        for leg in pos['legs']:
            formatted_legs.append({
                'strike': leg['strike'],
                'type': leg['type'],
                'position': leg['position'],
                'qty': int(leg['qty']),
                'entry_premium': leg['entry_premium'],
                'current_premium': leg['current_premium'],
                'delta': leg.get('delta'),
                'gamma': leg.get('gamma'),
                'theta': leg.get('theta'),
                'vega': leg.get('vega'),
                'iv': leg.get('iv')
            })
        
        output = {
            'timestamp': analysis['timestamp'],
            'underlying': analysis['underlying'],
            'current_price': analysis['current_price'],
            'vix': analysis.get('vix', 0),
            'iv_rank': analysis.get('iv_rank', 0),
            'iv_percentile': analysis.get('iv_percentile', 0),
            
            'position': {
                'position_id': pos['position_id'],
                'strategy': pos['strategy'],
                'dte': pos['dte'],
                'entry_date': pos['entry_date'],
                'legs': formatted_legs,
                'net_credit': pos['net_credit'],
                'current_value': pos['current_value'],
                'current_pnl': pos['current_pnl'],
                'max_profit': pos['max_profit'],
                'max_loss': pos['max_loss'],
                'breakeven_lower': pos.get('breakeven_lower'),
                'breakeven_upper': pos.get('breakeven_upper')
            },
            
            'greeks': {
                'position_delta': greeks['position_delta'],
                'position_gamma': greeks['position_gamma'],
                'position_theta': greeks['position_theta'],
                'position_vega': greeks['position_vega']
            },
            
            'market_regime': {
                'term_structure': regime.get('term_structure', 'unknown'),
                'put_call_skew': regime.get('put_call_skew', 0),
                'recent_volatility_trend': regime.get('recent_volatility_trend', 'unknown'),
                'earnings_date': regime.get('earnings_date'),
                'earnings_in_dte': regime.get('earnings_in_dte', False)
            }
        }
        
        # Add Monte Carlo if available
        if mc:
            output['monte_carlo'] = {
                'paths': mc.get('paths', 50000),
                'model': mc.get('model', 'GBM'),
                'pop': mc.get('pop', 0),
                'pot_lower': mc.get('pot_lower', 0),
                'pot_upper': mc.get('pot_upper', 0),
                'expected_pl': mc.get('expected_pl', 0),
                'median_pl': mc.get('median_pl', 0),
                'var_95': mc.get('var_95', 0),
                'var_99': mc.get('var_99', 0),
                'expected_shortfall_95': mc.get('expected_shortfall_95', 0),
                'optimal_exit_dte': mc.get('optimal_exit_dte', 0)
            }
        
        return output
    
    @staticmethod
    def print_summary(analysis: Dict) -> None:
        """Print a quick summary to console"""
        pos = analysis['position']
        greeks = analysis['greeks']
        
        print(f"\n{'─'*50}")
        print(f"📊 {pos['strategy']} | {analysis['underlying']} | {pos['dte']} DTE")
        print(f"{'─'*50}")
        print(f"P&L: ${pos['current_pnl']:+.2f} ({pos['current_pnl']/pos['max_profit']*100:.0f}% of max)")
        print(f"Delta: {greeks['position_delta']:+.3f} | Theta: ${greeks['position_theta']*100:+.2f}/day")
        
        if 'monte_carlo' in analysis:
            mc = analysis['monte_carlo']
            print(f"POP: {mc['pop']:.1f}% | Expected: ${mc['expected_pl']:+.2f}")
        
        print(f"{'─'*50}\n")

