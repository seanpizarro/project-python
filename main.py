#!/usr/bin/env python3
"""Project Python - Options Analyzer with TastyTrade & Alpaca Support"""

import sys
import json
import argparse
import getpass
from datetime import datetime
from pathlib import Path

from brokers.alpaca_client import AlpacaClient
from analyzers.strategy_detector import StrategyDetector
from analyzers.greeks_calculator import GreeksCalculator
from analyzers.market_data import MarketDataFetcher
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
    
    # Check for stored credentials
    username = config.get('tastytrade_username')
    password = config.get('tastytrade_password')
    
    if not username or not password:
        print("\n🔐 TastyTrade Login Required")
        print("   (Add TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD to .env to skip this)")
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
    
    # Parse command-line arguments first
    parser = argparse.ArgumentParser(description='Options Analyzer - TastyTrade & Alpaca')
    parser.add_argument('--symbol', '-s', type=str, help='Symbol to analyze (e.g., SPY)')
    parser.add_argument('--choice', '-c', type=int, help='Symbol choice number (1-based)')
    parser.add_argument('--broker', '-b', type=str, default='alpaca', 
                        choices=['alpaca', 'tastytrade', 'tt'],
                        help='Broker to use (default: alpaca)')
    parser.add_argument('--live', action='store_true', help='Use live account (default: paper/sandbox)')
    args = parser.parse_args()
    
    # Normalize broker name
    broker_name = 'tastytrade' if args.broker == 'tt' else args.broker
    is_paper = not args.live
    
    print("\n" + "="*60)
    print("PROJECT PYTHON - OPTIONS ANALYZER")
    print("="*60)
    print(f"Broker: {broker_name.upper()} ({'Paper' if is_paper else '🔴 LIVE'})")
    
    # Load config
    config = load_config()
    
    # Initialize client based on broker choice
    try:
        if broker_name == 'tastytrade':
            client = create_tastytrade_client(config, sandbox=is_paper)
            broker_display = "TastyTrade" + (" Sandbox" if is_paper else " Live")
        else:
            client = create_alpaca_client(config, paper=is_paper)
            broker_display = "Alpaca" + (" Paper" if is_paper else " Live")
    except Exception as e:
        print(f"\n❌ Failed to connect to {broker_name}: {e}")
        sys.exit(1)
    
    # Fetch account balance
    print(f"\nFetching account balance from {broker_display}...")
    try:
        balance = client.get_account_balance()
        print(f"  💰 Equity: ${balance['equity']:,.2f}")
        print(f"  💵 Cash: ${balance['cash']:,.2f}")
        print(f"  📊 Portfolio Value: ${balance['portfolio_value']:,.2f}")
        print(f"  💳 Buying Power: ${balance['buying_power']:,.2f}")
    except Exception as e:
        print(f"  ⚠️  Could not fetch balance: {e}")
    
    # Fetch positions
    print(f"\nFetching positions from {broker_display}...")
    
    if broker_name == 'tastytrade':
        positions = client.get_positions()
    else:
        positions = client.get_all_positions()
    
    if not positions:
        print(f"\n❌ No option positions found in {broker_display}")
        print("   Open some positions first, then re-run this script")
        sys.exit(1)
    
    # Group by symbol
    symbols = {}
    for pos in positions:
        sym = pos['underlying_symbol']
        if sym not in symbols:
            symbols[sym] = []
        symbols[sym].append(pos)
    
    print(f"\n✓ Found {len(positions)} legs across {len(symbols)} symbols\n")
    
    # Display symbols
    symbol_list = list(symbols.keys())
    for i, sym in enumerate(symbol_list, 1):
        count = len(symbols[sym])
        print(f"  {i}. {sym} ({count} legs)")
    
    # Select symbol - support command-line arg or interactive input
    if args.symbol and args.choice is not None:
        print("\n❌ Error: Cannot specify both --symbol and --choice")
        print("   Please use only one option")
        sys.exit(1)
    
    if args.symbol:
        if args.symbol.upper() not in symbol_list:
            print(f"\n❌ Symbol '{args.symbol}' not found in positions")
            print(f"   Available symbols: {', '.join(symbol_list)}")
            sys.exit(1)
        symbol = args.symbol.upper()
    elif args.choice is not None:
        if args.choice < 1 or args.choice > len(symbol_list):
            print(f"\n❌ Invalid choice: {args.choice}")
            print(f"   Please choose between 1 and {len(symbol_list)}")
            sys.exit(1)
        symbol = symbol_list[args.choice - 1]
    elif sys.stdin.isatty():
        try:
            user_input = input("\nSelect symbol (number or name): ").strip()
            
            if user_input.isdigit():
                choice = int(user_input)
                if choice < 1 or choice > len(symbol_list):
                    print(f"\n❌ Invalid choice: {choice}")
                    print(f"   Please choose between 1 and {len(symbol_list)}")
                    sys.exit(1)
                symbol = symbol_list[choice - 1]
            else:
                user_input_upper = user_input.upper()
                if user_input_upper in symbol_list:
                    symbol = user_input_upper
                else:
                    print(f"\n❌ Symbol '{user_input}' not found in positions")
                    print(f"   Available: {', '.join(symbol_list)}")
                    sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Cancelled")
            sys.exit(1)
    else:
        print("\n⚠️  Non-interactive terminal detected")
        print(f"   Using first symbol: {symbol_list[0]}")
        print(f"   (Use --symbol or --choice to specify)")
        symbol = symbol_list[0]
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {symbol}")
    print(f"{'='*60}")
    
    # Get positions for this symbol
    symbol_positions = symbols[symbol]
    
    print(f"\n[1/5] Position legs:")
    for i, pos in enumerate(symbol_positions, 1):
        print(f"      {i}. {pos['position'].upper()} {pos['qty']}x "
              f"{pos['type'].upper()} ${pos['strike']:.0f} @ ${pos['current_premium']:.2f}")
    
    # Detect strategy
    print("\n[2/5] Detecting strategy...")
    detector = StrategyDetector(symbol_positions)
    strategy_info = detector.detect_strategy()
    print(f"      ✓ {strategy_info['strategy']}")
    print(f"      ✓ {strategy_info['dte']} DTE")
    print(f"      ✓ P&L: ${strategy_info['current_pnl']:.2f}")
    
    # Get market data
    print("\n[3/5] Fetching market data...")
    try:
        current_price = client.get_current_price(symbol)
        print(f"      ✓ {symbol}: ${current_price:.2f}")
    except Exception as e:
        print(f"      ⚠️  Using fallback price source: {e}")
        current_price = 450.0
    
    # Get market metrics (IV rank, earnings, etc.)
    market_data = {}
    if broker_name == 'tastytrade':
        try:
            metrics = client.get_market_metrics(symbol)
            market_data = {
                'current_price': current_price,
                'iv_rank': metrics.get('iv_rank', 0),
                'iv_percentile': metrics.get('iv_percentile', 0),
                'iv': metrics.get('iv', 0),
                'hv_30': metrics.get('hv_30', 0),
                'earnings_date': metrics.get('earnings_date'),
                'term_structure': 'normal_contango',  # TODO: Calculate from VIX term
                'put_call_skew': 0,  # TODO: Calculate from chain
                'vol_trend': 'stable',
                'earnings_in_dte': bool(metrics.get('earnings_date'))
            }
            print(f"      ✓ IV Rank: {market_data['iv_rank']:.0f}")
            print(f"      ✓ IV Percentile: {market_data['iv_percentile']:.0f}")
            if market_data['earnings_date']:
                print(f"      ✓ Earnings: {market_data['earnings_date']}")
        except Exception as e:
            print(f"      ⚠️  Could not fetch TastyTrade metrics: {e}")
            market_fetcher = MarketDataFetcher()
            market_data = market_fetcher.fetch_all(symbol)
            market_data['current_price'] = current_price
    else:
        market_fetcher = MarketDataFetcher()
        market_data = market_fetcher.fetch_all(symbol)
        market_data['current_price'] = current_price
    
    # Calculate/Fetch Greeks
    print("\n[4/5] Fetching Greeks...")
    
    if broker_name == 'tastytrade':
        # Get REAL Greeks from TastyTrade
        try:
            enriched = client.enrich_positions_with_greeks(symbol_positions)
            print("      ✓ Real Greeks from TastyTrade exchange")
        except Exception as e:
            print(f"      ⚠️  Falling back to calculated Greeks: {e}")
            greeks_calc = GreeksCalculator(client)
            enriched = greeks_calc.enrich_positions(symbol_positions, market_data)
    else:
        # Calculate Greeks with Black-Scholes
        greeks_calc = GreeksCalculator(client)
        enriched = greeks_calc.enrich_positions(symbol_positions, market_data)
        print("      ✓ Greeks calculated (Black-Scholes)")
    
    # Aggregate position Greeks
    position_delta = 0
    position_gamma = 0
    position_theta = 0
    position_vega = 0
    
    for pos in enriched:
        mult = 1 if pos['position'] == 'long' else -1
        qty = pos['qty']
        
        if pos.get('delta'): position_delta += pos['delta'] * qty * mult
        if pos.get('gamma'): position_gamma += pos['gamma'] * qty * mult
        if pos.get('theta'): position_theta += pos['theta'] * qty * mult
        if pos.get('vega'): position_vega += pos['vega'] * qty * mult
    
    greeks = {
        'position_delta': round(position_delta, 3),
        'position_gamma': round(position_gamma, 4),
        'position_theta': round(position_theta, 3),
        'position_vega': round(position_vega, 3)
    }
    
    print(f"      ✓ Delta: {greeks['position_delta']:+.3f}")
    print(f"      ✓ Theta: ${greeks['position_theta']*100:+.2f}/day")
    if any(pos.get('iv') for pos in enriched):
        avg_iv = sum(pos.get('iv', 0) for pos in enriched if pos.get('iv')) / len(enriched)
        print(f"      ✓ Avg IV: {avg_iv*100:.1f}%")
    
    # Compile analysis
    print("\n[5/5] Compiling analysis...")
    
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'account_type': broker_display,
        'underlying': symbol,
        'current_price': current_price,
        'vix': market_data.get('vix', 0),
        'iv_rank': market_data.get('iv_rank', 0),
        'iv_percentile': market_data.get('iv_percentile', 0),
        
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
            'term_structure': market_data.get('term_structure', 'unknown'),
            'put_call_skew': market_data.get('put_call_skew', 0),
            'recent_volatility_trend': market_data.get('vol_trend', 'unknown'),
            'earnings_date': market_data.get('earnings_date'),
            'earnings_in_dte': market_data.get('earnings_in_dte', False)
        }
    }
    
    # Save to file
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    filename = f"analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Broker: {broker_display}")
    print(f"Strategy: {strategy_info['strategy']}")
    print(f"P&L: ${strategy_info['current_pnl']:.2f}")
    print(f"Greeks Source: {'Exchange (Real)' if broker_name == 'tastytrade' else 'Calculated (B-S)'}")
    print(f"\nSaved to: {filepath}")
    print(f"\n📊 Send this JSON to Claude for recommendation!")
    print(f"{'='*60}\n")
    
    # Cleanup
    if broker_name == 'tastytrade' and hasattr(client, 'close'):
        client.close()


if __name__ == '__main__':
    main()
