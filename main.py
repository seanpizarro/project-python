#!/usr/bin/env python3
"""Project Python - Main Entry Point"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

from brokers.alpaca_client import AlpacaClient
from analyzers.strategy_detector import StrategyDetector
from analyzers.greeks_calculator import GreeksCalculator
from analyzers.market_data import MarketDataFetcher
from config import load_config


def main():
    """Main program"""
    
    # Parse command-line arguments first
    parser = argparse.ArgumentParser(description='Options Analyzer')
    parser.add_argument('--symbol', '-s', type=str, help='Symbol to analyze (e.g., SPY)')
    parser.add_argument('--choice', '-c', type=int, help='Symbol choice number (1-based)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PROJECT PYTHON - OPTIONS ANALYZER")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Initialize Alpaca client
    client = AlpacaClient(
        api_key=config['alpaca_paper_key'],
        secret_key=config['alpaca_paper_secret'],
        paper=True
    )
    
    # Fetch account balance
    print("\nFetching account balance...")
    try:
        balance = client.get_account_balance()
        print(f"  💰 Equity: ${balance['equity']:,.2f}")
        print(f"  💵 Cash: ${balance['cash']:,.2f}")
        print(f"  📊 Portfolio Value: ${balance['portfolio_value']:,.2f}")
        print(f"  💳 Buying Power: ${balance['buying_power']:,.2f}")
    except Exception as e:
        print(f"  ⚠️  Could not fetch balance: {e}")
    
    # Fetch positions
    print("\nFetching positions from Alpaca Paper...")
    positions = client.get_all_positions()
    
    if not positions:
        print("\n❌ No option positions found in Alpaca Paper")
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
    
    # Validate that only one selection method is used
    if args.symbol and args.choice is not None:
        print("\n❌ Error: Cannot specify both --symbol and --choice")
        print("   Please use only one option")
        sys.exit(1)
    
    if args.symbol:
        # Direct symbol name provided
        if args.symbol not in symbol_list:
            print(f"\n❌ Symbol '{args.symbol}' not found in positions")
            print(f"   Available symbols: {', '.join(symbol_list)}")
            sys.exit(1)
        symbol = args.symbol
    elif args.choice is not None:
        # Choice number provided
        if args.choice < 1 or args.choice > len(symbol_list):
            print(f"\n❌ Invalid choice: {args.choice}")
            print(f"   Please choose between 1 and {len(symbol_list)}")
            sys.exit(1)
        symbol = symbol_list[args.choice - 1]
    elif sys.stdin.isatty():
        # Interactive terminal - use input()
        try:
            user_input = input("\nSelect symbol (number or name): ").strip()
            
            # Try as number first
            if user_input.isdigit():
                choice = int(user_input)
                if choice < 1 or choice > len(symbol_list):
                    print(f"\n❌ Invalid choice: {choice}")
                    print(f"   Please choose between 1 and {len(symbol_list)}")
                    sys.exit(1)
                symbol = symbol_list[choice - 1]
            else:
                # Try as symbol name (case-insensitive)
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
        # Non-interactive - read from stdin or use first symbol
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
        print(f"      ⚠️  Using default price: {e}")
        current_price = 450.0
    
    market_fetcher = MarketDataFetcher()
    market_data = market_fetcher.fetch_all(symbol)
    market_data['current_price'] = current_price
    
    # Calculate Greeks
    print("\n[4/5] Calculating Greeks...")
    greeks_calc = GreeksCalculator(client)
    enriched = greeks_calc.enrich_positions(symbol_positions, market_data)
    
    greeks = strategy_info['position_greeks']
    print(f"      ✓ Delta: {greeks['position_delta']:+.3f}")
    print(f"      ✓ Theta: ${greeks['position_theta']*100:+.2f}/day")
    
    # Compile analysis
    print("\n[5/5] Compiling analysis...")
    
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'account_type': 'Alpaca Paper Trading',
        'underlying': symbol,
        'current_price': current_price,
        'vix': market_data['vix'],
        'iv_rank': market_data['iv_rank'],
        'iv_percentile': market_data['iv_percentile'],
        
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
            'term_structure': market_data['term_structure'],
            'put_call_skew': market_data['put_call_skew'],
            'recent_volatility_trend': market_data['vol_trend'],
            'earnings_in_dte': market_data['earnings_in_dte']
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
    print(f"Strategy: {strategy_info['strategy']}")
    print(f"P&L: ${strategy_info['current_pnl']:.2f}")
    print(f"\nSaved to: {filepath}")
    print(f"\n📊 Send this JSON to Claude for recommendation!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()