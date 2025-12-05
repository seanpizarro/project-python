#!/usr/bin/env python3
"""Project Python - Options Analyzer with Monte Carlo, Real Greeks, and Market Analysis"""

import sys
import json
import argparse
import getpass
from datetime import datetime
from pathlib import Path

from brokers.alpaca_client import AlpacaClient
from analyzers.strategy_detector import StrategyDetector
from analyzers.greeks_calculator import GreeksCalculator
from analyzers.monte_carlo import MonteCarloSimulator
from analyzers.market_analyzer import MarketAnalyzer
from analyzers.report_formatter import ReportFormatter
from config import load_config


def create_alpaca_client(config: dict, paper: bool = True):
    """Create Alpaca client"""
    if paper:
        return AlpacaClient(
            api_key=config['alpaca_paper_key'],
            secret_key=config['alpaca_paper_secret'],
            paper=True
        )
    else:
        return AlpacaClient(
            api_key=config['alpaca_live_key'],
            secret_key=config['alpaca_live_secret'],
            paper=False
        )


def create_tastytrade_client(config: dict, sandbox: bool = False):
    """Create TastyTrade client"""
    from brokers.tastytrade_client import TastyTradeClient
    
    username = config.get('tastytrade_username')
    password = config.get('tastytrade_password')
    
    if not username or not password:
        print("\n🔐 TastyTrade Login Required")
        print("   (Add TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD to .env to skip)")
        username = input("   Username: ").strip()
        password = getpass.getpass("   Password: ")
    
    account = config.get('tastytrade_account') if not sandbox else None
    
    return TastyTradeClient(
        username=username,
        password=password,
        account_number=account,
        sandbox=sandbox
    )


def main():
    """Main program"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Options Analyzer with Monte Carlo')
    parser.add_argument('--symbol', '-s', type=str, help='Symbol to analyze')
    parser.add_argument('--choice', '-c', type=int, help='Symbol choice number')
    parser.add_argument('--broker', '-b', type=str, default='alpaca',
                        choices=['alpaca', 'tastytrade', 'tt'],
                        help='Broker (default: alpaca)')
    parser.add_argument('--live', action='store_true', help='Use live account')
    parser.add_argument('--monte-carlo', '-mc', type=int, default=50000,
                        help='Monte Carlo paths (default: 50000)')
    parser.add_argument('--heston', action='store_true', 
                        help='Use Heston model instead of GBM')
    parser.add_argument('--no-monte-carlo', action='store_true',
                        help='Skip Monte Carlo simulation')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    args = parser.parse_args()
    
    broker_name = 'tastytrade' if args.broker == 'tt' else args.broker
    is_paper = not args.live
    
    print("\n" + "═"*60)
    print("PROJECT PYTHON - OPTIONS ANALYZER")
    print("═"*60)
    print(f"Broker: {broker_name.upper()} ({'Paper' if is_paper else '🔴 LIVE'})")
    print(f"Monte Carlo: {args.monte_carlo:,} paths {'(Heston)' if args.heston else '(GBM)'}")
    
    # Load config
    config = load_config()
    
    # Initialize broker client
    try:
        if broker_name == 'tastytrade':
            client = create_tastytrade_client(config, sandbox=is_paper)
            broker_display = "TastyTrade" + (" Sandbox" if is_paper else " Live")
        else:
            client = create_alpaca_client(config, paper=is_paper)
            broker_display = "Alpaca" + (" Paper" if is_paper else " Live")
    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")
        sys.exit(1)
    
    # Fetch account balance
    print(f"\n[1/7] Fetching account balance...")
    try:
        balance = client.get_account_balance()
        print(f"      💰 Equity: ${balance['equity']:,.2f}")
        print(f"      💵 Cash: ${balance['cash']:,.2f}")
        print(f"      💳 Buying Power: ${balance['buying_power']:,.2f}")
    except Exception as e:
        print(f"      ⚠️  Balance error: {e}")
    
    # Fetch positions
    print(f"\n[2/7] Fetching positions...")
    positions = client.get_positions() if broker_name == 'tastytrade' else client.get_all_positions()
    
    if not positions:
        print(f"\n❌ No option positions found")
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
    
    # Market data and analysis
    print(f"\n[4/7] Analyzing market conditions...")
    market_analyzer = MarketAnalyzer()
    
    try:
        current_price = client.get_current_price(symbol)
        print(f"      ✓ {symbol}: ${current_price:.2f}")
    except Exception:
        current_price = 450.0
        print(f"      ⚠️  Using fallback price")
    
    # Get VIX and term structure
    vix_data = market_analyzer.get_vix_data()
    term_structure = market_analyzer.analyze_term_structure(vix_data)
    print(f"      ✓ VIX: {vix_data['vix']:.1f}")
    print(f"      ✓ Term Structure: {term_structure['structure']}")
    
    # Get IV rank (from broker or calculated)
    if broker_name == 'tastytrade':
        try:
            metrics = client.get_market_metrics(symbol)
            iv_rank = metrics.get('iv_rank', 0)
            iv_percentile = metrics.get('iv_percentile', 0)
            earnings_date = metrics.get('earnings_date')
            print(f"      ✓ IV Rank: {iv_rank:.0f} (from TastyTrade)")
        except Exception:
            iv_analysis = market_analyzer.calculate_iv_rank(symbol)
            iv_rank = iv_analysis['iv_rank']
            iv_percentile = iv_analysis['iv_percentile']
            earnings_date = None
    else:
        iv_analysis = market_analyzer.calculate_iv_rank(symbol)
        iv_rank = iv_analysis['iv_rank']
        iv_percentile = iv_analysis['iv_percentile']
        earnings_info = market_analyzer.get_earnings_info(symbol)
        earnings_date = earnings_info.get('earnings_date')
        print(f"      ✓ IV Rank: {iv_rank:.0f} (calculated)")
    
    if earnings_date:
        print(f"      ⚠️  Earnings: {earnings_date}")
    
    # Greeks
    print(f"\n[5/7] Fetching Greeks...")
    if broker_name == 'tastytrade':
        try:
            enriched = client.enrich_positions_with_greeks(symbol_positions)
            print("      ✓ Real Greeks from exchange")
        except Exception:
            greeks_calc = GreeksCalculator(client)
            enriched = greeks_calc.enrich_positions(symbol_positions, {'current_price': current_price})
            print("      ✓ Calculated Greeks (B-S)")
    else:
        greeks_calc = GreeksCalculator(client)
        enriched = greeks_calc.enrich_positions(symbol_positions, {'current_price': current_price})
        print("      ✓ Calculated Greeks (B-S)")
    
    # Aggregate Greeks
    position_delta = sum((p.get('delta', 0) or 0) * p['qty'] * (1 if p['position'] == 'long' else -1) for p in enriched)
    position_gamma = sum((p.get('gamma', 0) or 0) * p['qty'] * (1 if p['position'] == 'long' else -1) for p in enriched)
    position_theta = sum((p.get('theta', 0) or 0) * p['qty'] * (1 if p['position'] == 'long' else -1) for p in enriched)
    position_vega = sum((p.get('vega', 0) or 0) * p['qty'] * (1 if p['position'] == 'long' else -1) for p in enriched)
    
    greeks = {
        'position_delta': round(position_delta, 3),
        'position_gamma': round(position_gamma, 4),
        'position_theta': round(position_theta, 3),
        'position_vega': round(position_vega, 3)
    }
    
    print(f"      Δ Delta: {greeks['position_delta']:+.3f}")
    print(f"      Θ Theta: ${greeks['position_theta']*100:+.2f}/day")
    
    # Put/Call Skew
    skew = market_analyzer.calculate_put_call_skew(positions=enriched)
    print(f"      ✓ Put/Call Skew: {skew['skew']:+.1f}")
    
    # Monte Carlo
    monte_carlo_result = None
    if not args.no_monte_carlo:
        print(f"\n[6/7] Running Monte Carlo ({args.monte_carlo:,} paths)...")
        
        # Get average IV from positions
        ivs = [p.get('iv', 0) for p in enriched if p.get('iv')]
        avg_iv = sum(ivs) / len(ivs) if ivs else 0.25
        
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
    
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'account_type': broker_display,
        'underlying': symbol,
        'current_price': current_price,
        'vix': vix_data['vix'],
        'iv_rank': iv_rank,
        'iv_percentile': iv_percentile,
        
        'position': {
            'position_id': f"{strategy_info['strategy']}_{symbol}_{datetime.now().strftime('%Y%m%d')}",
            'symbol': symbol,
            'strategy': strategy_info['strategy'],
            'dte': strategy_info['dte'],
            'entry_date': strategy_info['entry_date'],
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
    
    # Use formatter for clean JSON
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
    
    # Cleanup
    if broker_name == 'tastytrade' and hasattr(client, 'close'):
        client.close()


if __name__ == '__main__':
    main()
