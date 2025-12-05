#!/usr/bin/env python3
"""
Project Python - Options Analyzer
- Positions from: Alpaca
- Real Greeks/IV from: TastyTrade (when available)
- Trading: Alpaca
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

from brokers.alpaca_client import AlpacaClient
from analyzers.strategy_detector import StrategyDetector
from analyzers.greeks_calculator import GreeksCalculator
from analyzers.monte_carlo import MonteCarloSimulator
from analyzers.market_analyzer import MarketAnalyzer
from analyzers.report_formatter import ReportFormatter
from config import load_config


def create_tastytrade_data_client(config: dict):
    """Create TastyTrade client for market data (IV rank, metrics)"""
    from brokers.tastytrade_data import TastyTradeDataClient
    
    username = config.get('tastytrade_username')
    password = config.get('tastytrade_password')
    
    if username and password:
        try:
            client = TastyTradeDataClient(username=username, password=password)
            if client.test_connection():
                return client
            else:
                print("  ⚠️  TastyTrade connection test failed")
        except Exception as e:
            print(f"  ⚠️  TastyTrade auth failed: {e}")
    else:
        # Will prompt for credentials when needed
        return TastyTradeDataClient()
    
    return None


def main():
    """Main program"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Options Analyzer - Alpaca + TastyTrade')
    parser.add_argument('--symbol', '-s', type=str, help='Symbol to analyze')
    parser.add_argument('--choice', '-c', type=int, help='Symbol choice number')
    parser.add_argument('--live', action='store_true', help='Use Alpaca live account')
    parser.add_argument('--monte-carlo', '-mc', type=int, default=50000,
                        help='Monte Carlo paths (default: 50000)')
    parser.add_argument('--heston', action='store_true', help='Use Heston model')
    parser.add_argument('--no-monte-carlo', action='store_true', help='Skip Monte Carlo')
    parser.add_argument('--no-tastytrade', action='store_true', help='Skip TastyTrade data')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    args = parser.parse_args()
    
    is_paper = not args.live
    
    print("\n" + "═"*60)
    print("PROJECT PYTHON - OPTIONS ANALYZER")
    print("═"*60)
    print(f"Positions: Alpaca {'Paper' if is_paper else '🔴 LIVE'}")
    print(f"Monte Carlo: {args.monte_carlo:,} paths {'(Heston)' if args.heston else '(GBM)'}")
    
    # Load config
    config = load_config()
    
    # Initialize Alpaca for positions
    try:
        if is_paper:
            alpaca = AlpacaClient(
                api_key=config['alpaca_paper_key'],
                secret_key=config['alpaca_paper_secret'],
                paper=True
            )
        else:
            alpaca = AlpacaClient(
                api_key=config['alpaca_live_key'],
                secret_key=config['alpaca_live_secret'],
                paper=False
            )
    except Exception as e:
        print(f"\n❌ Alpaca connection failed: {e}")
        sys.exit(1)
    
    # Initialize TastyTrade for market data (real Greeks/IV)
    tastytrade = None
    if not args.no_tastytrade:
        print("Market Data: ", end="")
        tastytrade = create_tastytrade_data_client(config)
        if tastytrade:
            print("TastyTrade (real Greeks/IV) ✓")
        else:
            print("Calculated (TastyTrade unavailable)")
    
    # Fetch account balance
    print(f"\n[1/7] Fetching Alpaca balance...")
    try:
        balance = alpaca.get_account_balance()
        print(f"      💰 Equity: ${balance['equity']:,.2f}")
        print(f"      💵 Cash: ${balance['cash']:,.2f}")
        print(f"      💳 Buying Power: ${balance['buying_power']:,.2f}")
    except Exception as e:
        print(f"      ⚠️  Balance error: {e}")
    
    # Fetch positions from Alpaca
    print(f"\n[2/7] Fetching positions from Alpaca...")
    positions = alpaca.get_all_positions()
    
    if not positions:
        print(f"\n❌ No option positions found in Alpaca")
        sys.exit(1)
    
    # Group by symbol
    symbols = {}
    for pos in positions:
        sym = pos['underlying_symbol']
        if sym not in symbols:
            symbols[sym] = []
        symbols[sym].append(pos)
    
    print(f"      ✓ Found {len(positions)} legs across {len(symbols)} symbols")
    
    symbol_list = list(symbols.keys())
    for i, sym in enumerate(symbol_list, 1):
        print(f"        {i}. {sym} ({len(symbols[sym])} legs)")
    
    # Select symbol
    if args.symbol and args.choice is not None:
        print("\n❌ Cannot specify both --symbol and --choice")
        sys.exit(1)
    
    if args.symbol:
        symbol = args.symbol.upper()
        if symbol not in symbol_list:
            print(f"\n❌ Symbol '{symbol}' not found")
            sys.exit(1)
    elif args.choice is not None:
        if args.choice < 1 or args.choice > len(symbol_list):
            print(f"\n❌ Invalid choice: {args.choice}")
            sys.exit(1)
        symbol = symbol_list[args.choice - 1]
    elif sys.stdin.isatty():
        try:
            user_input = input("\n      Select (number or name): ").strip()
            if user_input.isdigit():
                choice = int(user_input)
                if choice < 1 or choice > len(symbol_list):
                    print(f"\n❌ Invalid choice")
                    sys.exit(1)
                symbol = symbol_list[choice - 1]
            else:
                symbol = user_input.upper()
                if symbol not in symbol_list:
                    print(f"\n❌ Symbol not found")
                    sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Cancelled")
            sys.exit(1)
    else:
        symbol = symbol_list[0]
        print(f"      Using: {symbol}")
    
    symbol_positions = symbols[symbol]
    
    print(f"\n{'═'*60}")
    print(f"ANALYZING {symbol}")
    print(f"{'═'*60}")
    
    # Strategy detection
    print(f"\n[3/7] Detecting strategy...")
    detector = StrategyDetector(symbol_positions)
    strategy_info = detector.detect_strategy()
    print(f"      ✓ {strategy_info['strategy']}")
    print(f"      ✓ {strategy_info['dte']} DTE")
    print(f"      ✓ P&L: ${strategy_info['current_pnl']:.2f}")
    
    # Market data
    print(f"\n[4/7] Fetching market data...")
    market_analyzer = MarketAnalyzer()
    
    # Get price from Alpaca
    try:
        current_price = alpaca.get_current_price(symbol)
        print(f"      ✓ {symbol}: ${current_price:.2f} (Alpaca)")
    except Exception:
        current_price = 450.0
        print(f"      ⚠️  Using fallback price")
    
    # Get VIX and IV metrics
    vix_data = market_analyzer.get_vix_data()
    term_structure = market_analyzer.analyze_term_structure(vix_data)
    print(f"      ✓ VIX: {vix_data['vix']:.1f}")
    print(f"      ✓ Term Structure: {term_structure['structure']}")
    
    # Get IV rank from TastyTrade or calculate
    iv_rank = 0
    iv_percentile = 0
    earnings_date = None
    
    if tastytrade:
        try:
            metrics = tastytrade.get_market_metrics(symbol)
            if metrics:
                iv_rank = metrics.get('iv_rank', 0) or 0
                iv_percentile = metrics.get('iv_percentile', 0) or 0
                earnings_date = metrics.get('earnings_date')
                print(f"      ✓ IV Rank: {iv_rank:.0f} (TastyTrade)")
                print(f"      ✓ IV Percentile: {iv_percentile:.0f} (TastyTrade)")
                if earnings_date:
                    print(f"      ⚠️  Earnings: {earnings_date}")
        except Exception as e:
            print(f"      ⚠️  TastyTrade metrics error: {e}")
    
    if iv_rank == 0:
        iv_analysis = market_analyzer.calculate_iv_rank(symbol)
        iv_rank = iv_analysis['iv_rank']
        iv_percentile = iv_analysis['iv_percentile']
        print(f"      ✓ IV Rank: {iv_rank:.0f} (calculated from HV)")
    
    # Greeks - from TastyTrade or calculated
    print(f"\n[5/7] Fetching Greeks...")
    
    if tastytrade:
        try:
            enriched = tastytrade.enrich_positions_with_greeks(symbol_positions)
            has_real_greeks = any(p.get('iv_source') == 'tastytrade_exchange' for p in enriched)
            if has_real_greeks:
                print("      ✓ Real Greeks from TastyTrade exchange")
            else:
                raise Exception("No Greeks returned")
        except Exception as e:
            print(f"      ⚠️  TastyTrade Greeks unavailable: {e}")
            greeks_calc = GreeksCalculator(alpaca)
            enriched = greeks_calc.enrich_positions(symbol_positions, {'current_price': current_price})
            print("      ✓ Calculated Greeks (from option prices)")
    else:
        greeks_calc = GreeksCalculator(alpaca)
        enriched = greeks_calc.enrich_positions(symbol_positions, {'current_price': current_price})
        print("      ✓ Calculated Greeks (from option prices)")
    
    # Aggregate position Greeks
    position_delta = sum((p.get('delta') or 0) * p['qty'] * (1 if p['position'] == 'long' else -1) for p in enriched)
    position_gamma = sum((p.get('gamma') or 0) * p['qty'] * (1 if p['position'] == 'long' else -1) for p in enriched)
    position_theta = sum((p.get('theta') or 0) * p['qty'] * (1 if p['position'] == 'long' else -1) for p in enriched)
    position_vega = sum((p.get('vega') or 0) * p['qty'] * (1 if p['position'] == 'long' else -1) for p in enriched)
    
    greeks = {
        'position_delta': round(position_delta, 3),
        'position_gamma': round(position_gamma, 4),
        'position_theta': round(position_theta, 3),
        'position_vega': round(position_vega, 3)
    }
    
    print(f"      Δ Delta: {greeks['position_delta']:+.3f}")
    print(f"      Θ Theta: ${greeks['position_theta']*100:+.2f}/day")
    
    # Show IV from positions
    ivs = [p.get('iv') for p in enriched if p.get('iv')]
    if ivs:
        avg_iv = sum(ivs) / len(ivs)
        print(f"      IV: {avg_iv*100:.1f}%")
    
    # Put/Call Skew
    skew = market_analyzer.calculate_put_call_skew(positions=enriched)
    print(f"      Skew: {skew['skew']:+.1f}")
    
    # Monte Carlo
    monte_carlo_result = None
    if not args.no_monte_carlo:
        print(f"\n[6/7] Running Monte Carlo ({args.monte_carlo:,} paths)...")
        
        # Get IV for simulation
        ivs = [p.get('iv') for p in enriched if p.get('iv')]
        if ivs:
            avg_iv = sum(ivs) / len(ivs)
            iv_source = "from positions"
        else:
            avg_iv = vix_data['vix'] / 100 * 1.2
            iv_source = "VIX estimate"
        
        print(f"      Using IV: {avg_iv*100:.1f}% ({iv_source})")
        
        simulator = MonteCarloSimulator(n_paths=args.monte_carlo)
        
        try:
            mc_result = simulator.run_simulation(
                current_price=current_price,
                positions=enriched,
                dte=strategy_info['dte'],
                volatility=avg_iv,
                entry_credit=strategy_info['net_credit'],
                breakeven_lower=strategy_info.get('breakeven_lower'),
                breakeven_upper=strategy_info.get('breakeven_upper'),
                use_heston=args.heston
            )
            
            monte_carlo_result = mc_result.to_dict()
            
            print(f"      ✓ Probability of Profit: {mc_result.pop:.1f}%")
            print(f"      ✓ Expected P&L: ${mc_result.expected_pl:+.2f}")
            print(f"      ✓ 95% VaR: ${mc_result.var_95:.2f}")
            print(f"      ✓ Optimal Exit: {mc_result.optimal_exit_dte} DTE")
            
        except Exception as e:
            print(f"      ⚠️  Monte Carlo error: {e}")
    else:
        print(f"\n[6/7] Monte Carlo skipped")
    
    # Compile analysis
    print(f"\n[7/7] Compiling analysis...")
    
    has_real_iv = any(p.get('iv') and p.get('iv_source') == 'tastytrade_exchange' for p in enriched)
    has_calculated_iv = any(p.get('iv') and p.get('iv_source') == 'calculated_from_price' for p in enriched)
    
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'positions_source': f"Alpaca {'Paper' if is_paper else 'Live'}",
        'underlying': symbol,
        'current_price': current_price,
        'vix': vix_data['vix'],
        'iv_rank': iv_rank,
        'iv_percentile': iv_percentile,
        
        'data_sources': {
            'positions': 'alpaca',
            'prices': 'alpaca',
            'greeks': 'tastytrade_exchange' if has_real_iv else ('calculated_from_prices' if has_calculated_iv else 'unavailable'),
            'iv': 'tastytrade_exchange' if has_real_iv else ('calculated_from_prices' if has_calculated_iv else 'vix_estimate'),
            'iv_rank': 'tastytrade' if (tastytrade and iv_rank > 0) else 'calculated_52w_hv',
            'vix': 'yfinance'
        },
        
        'position': {
            'position_id': f"{strategy_info['strategy']}_{symbol}_{datetime.now().strftime('%Y%m%d')}",
            'symbol': symbol,
            'strategy': strategy_info['strategy'],
            'dte': strategy_info['dte'],
            'entry_date': None,  # Alpaca doesn't provide
            'legs': enriched,
            'net_credit': strategy_info['net_credit'],
            'current_value': strategy_info['current_value'],
            'current_pnl': strategy_info['current_pnl'],
            'max_profit': strategy_info['max_profit'],
            'max_loss': strategy_info['max_loss'],
            'breakeven_lower': strategy_info.get('breakeven_lower'),
            'breakeven_upper': strategy_info.get('breakeven_upper')
        },
        
        'greeks': greeks,
        
        'market_regime': {
            'term_structure': term_structure['structure'],
            'term_structure_description': term_structure['description'],
            'put_call_skew': skew['skew'],
            'recent_volatility_trend': market_analyzer._determine_vol_trend(vix_data),
            'earnings_date': earnings_date,
            'earnings_in_dte': bool(earnings_date)
        }
    }
    
    if monte_carlo_result:
        analysis_data['monte_carlo'] = monte_carlo_result
    
    # Save JSON
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    filename = f"analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_dir / filename
    
    clean_json = ReportFormatter.format_json_for_claude(analysis_data)
    
    with open(filepath, 'w') as f:
        json.dump(clean_json, f, indent=2)
    
    # Print formatted report
    if not args.quiet:
        report = ReportFormatter.format_console_report(analysis_data)
        print(report)
    
    print(f"\n✓ Saved to: {filepath}")
    print(f"\n📊 Send this JSON to Claude for recommendation!")
    print("═"*60 + "\n")


if __name__ == '__main__':
    main()
