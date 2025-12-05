#!/usr/bin/env python3
"""
Shadow Trader - Mirror Alpaca positions to TastyTrade Sandbox
Keeps both accounts in sync for comparison testing
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from brokers.alpaca_client import AlpacaClient
from brokers.tastytrade_trader import TastyTradeTrader
from config import load_config


def normalize_position(pos: Dict) -> Dict:
    """Normalize position format for comparison"""
    return {
        'underlying': pos.get('underlying_symbol', ''),
        'strike': pos.get('strike', 0),
        'type': pos.get('type', '').lower(),
        'position': pos.get('position', '').lower(),
        'qty': abs(pos.get('qty', 0)),
        'expiration': pos.get('expiration', ''),
        'symbol': pos.get('symbol', '').replace(' ', '')
    }


def positions_match(p1: Dict, p2: Dict) -> bool:
    """Check if two positions represent the same contract"""
    return (
        p1['underlying'] == p2['underlying'] and
        abs(p1['strike'] - p2['strike']) < 0.01 and
        p1['type'] == p2['type'] and
        p1['expiration'] == p2['expiration']
    )


def find_differences(alpaca_positions: List[Dict], tt_positions: List[Dict]) -> Dict:
    """
    Find differences between Alpaca and TastyTrade positions
    
    Returns:
        Dict with:
        - to_open: positions to open in TastyTrade
        - to_close: positions to close in TastyTrade
        - qty_adjustments: positions with different quantities
        - matched: positions that match
    """
    alpaca_norm = [normalize_position(p) for p in alpaca_positions]
    tt_norm = [normalize_position(p) for p in tt_positions]
    
    to_open = []
    to_close = []
    qty_adjustments = []
    matched = []
    
    # Find positions to open (in Alpaca but not in TT)
    for ap in alpaca_norm:
        found = False
        for tp in tt_norm:
            if positions_match(ap, tp):
                found = True
                if ap['qty'] != tp['qty'] or ap['position'] != tp['position']:
                    qty_adjustments.append({
                        'alpaca': ap,
                        'tastytrade': tp,
                        'diff': ap['qty'] - tp['qty'] if ap['position'] == tp['position'] else ap['qty'] + tp['qty']
                    })
                else:
                    matched.append(ap)
                break
        
        if not found:
            to_open.append(ap)
    
    # Find positions to close (in TT but not in Alpaca)
    for tp in tt_norm:
        found = False
        for ap in alpaca_norm:
            if positions_match(ap, tp):
                found = True
                break
        
        if not found:
            to_close.append(tp)
    
    return {
        'to_open': to_open,
        'to_close': to_close,
        'qty_adjustments': qty_adjustments,
        'matched': matched
    }


def sync_positions(
    alpaca: AlpacaClient,
    tastytrade: TastyTradeTrader,
    dry_run: bool = True
) -> Dict:
    """
    Sync Alpaca positions to TastyTrade sandbox
    
    Args:
        alpaca: Alpaca client
        tastytrade: TastyTrade client (sandbox)
        dry_run: If True, only show what would happen
    
    Returns:
        Sync results dict
    """
    print("\n" + "="*60)
    print("SHADOW TRADER - POSITION SYNC")
    print("="*60)
    print(f"Mode: {'DRY RUN' if dry_run else '🔴 LIVE SYNC'}")
    
    # Get positions from both
    print("\n[1/4] Fetching Alpaca positions...")
    alpaca_positions = alpaca.get_all_positions()
    print(f"      Found {len(alpaca_positions)} positions")
    
    print("\n[2/4] Fetching TastyTrade positions...")
    tt_positions = tastytrade.get_positions()
    print(f"      Found {len(tt_positions)} positions")
    
    # Find differences
    print("\n[3/4] Comparing positions...")
    diff = find_differences(alpaca_positions, tt_positions)
    
    print(f"      ✓ Matched: {len(diff['matched'])}")
    print(f"      ➕ To open in TT: {len(diff['to_open'])}")
    print(f"      ➖ To close in TT: {len(diff['to_close'])}")
    print(f"      📊 Qty adjustments: {len(diff['qty_adjustments'])}")
    
    # Show details
    if diff['to_open']:
        print("\n      Positions to OPEN in TastyTrade:")
        for pos in diff['to_open']:
            action = 'Buy' if pos['position'] == 'long' else 'Sell'
            print(f"        → {action} {pos['qty']}x {pos['underlying']} "
                  f"{pos['type'].upper()} ${pos['strike']:.0f} {pos['expiration']}")
    
    if diff['to_close']:
        print("\n      Positions to CLOSE in TastyTrade:")
        for pos in diff['to_close']:
            action = 'Sell' if pos['position'] == 'long' else 'Buy'
            print(f"        → {action} {pos['qty']}x {pos['underlying']} "
                  f"{pos['type'].upper()} ${pos['strike']:.0f} {pos['expiration']}")
    
    # Execute sync
    results = {
        'opened': [],
        'closed': [],
        'errors': [],
        'dry_run': dry_run
    }
    
    if not dry_run:
        print("\n[4/4] Executing sync...")
        
        # Open missing positions
        for pos in diff['to_open']:
            action = 'Buy to Open' if pos['position'] == 'long' else 'Sell to Open'
            print(f"      Opening: {pos['symbol']}...")
            
            result = tastytrade.place_option_order(
                underlying=pos['underlying'],
                expiration=pos['expiration'],
                strike=pos['strike'],
                option_type=pos['type'],
                action='buy_to_open' if pos['position'] == 'long' else 'sell_to_open',
                quantity=int(pos['qty']),
                order_type='Market'
            )
            
            if result.get('success'):
                results['opened'].append(pos)
                print(f"        ✓ Opened")
            else:
                results['errors'].append({'position': pos, 'error': result.get('error')})
                print(f"        ✗ Error: {result.get('error')}")
        
        # Close extra positions
        for pos in diff['to_close']:
            action = 'Sell to Close' if pos['position'] == 'long' else 'Buy to Close'
            print(f"      Closing: {pos['symbol']}...")
            
            result = tastytrade.place_option_order(
                underlying=pos['underlying'],
                expiration=pos['expiration'],
                strike=pos['strike'],
                option_type=pos['type'],
                action='sell_to_close' if pos['position'] == 'long' else 'buy_to_close',
                quantity=int(pos['qty']),
                order_type='Market'
            )
            
            if result.get('success'):
                results['closed'].append(pos)
                print(f"        ✓ Closed")
            else:
                results['errors'].append({'position': pos, 'error': result.get('error')})
                print(f"        ✗ Error: {result.get('error')}")
    else:
        print("\n[4/4] Dry run - no trades executed")
        print("      Run with --execute to sync positions")
    
    return results


def compare_accounts(alpaca: AlpacaClient, tastytrade: TastyTradeTrader) -> Dict:
    """Compare account balances and positions between brokers"""
    print("\n" + "="*60)
    print("BROKER COMPARISON")
    print("="*60)
    
    # Balances
    print("\n📊 ACCOUNT BALANCES")
    print("-"*40)
    
    alpaca_bal = alpaca.get_account_balance()
    tt_bal = tastytrade.get_account_balance()
    
    print(f"{'Metric':<25} {'Alpaca':>15} {'TastyTrade':>15}")
    print("-"*55)
    print(f"{'Equity':<25} ${alpaca_bal.get('equity', 0):>14,.2f} ${tt_bal.get('equity', 0):>14,.2f}")
    print(f"{'Cash':<25} ${alpaca_bal.get('cash', 0):>14,.2f} ${tt_bal.get('cash', 0):>14,.2f}")
    print(f"{'Buying Power':<25} ${alpaca_bal.get('buying_power', 0):>14,.2f} ${tt_bal.get('buying_power', 0):>14,.2f}")
    
    # Positions
    print("\n📊 POSITIONS")
    print("-"*40)
    
    alpaca_pos = alpaca.get_all_positions()
    tt_pos = tastytrade.get_positions()
    
    print(f"Alpaca: {len(alpaca_pos)} positions")
    print(f"TastyTrade: {len(tt_pos)} positions")
    
    # Compare
    diff = find_differences(alpaca_pos, tt_pos)
    
    print(f"\nSync Status:")
    print(f"  ✓ Matched: {len(diff['matched'])}")
    print(f"  ➕ Only in Alpaca: {len(diff['to_open'])}")
    print(f"  ➖ Only in TastyTrade: {len(diff['to_close'])}")
    
    return {
        'alpaca_balance': alpaca_bal,
        'tastytrade_balance': tt_bal,
        'alpaca_positions': len(alpaca_pos),
        'tastytrade_positions': len(tt_pos),
        'matched': len(diff['matched']),
        'differences': len(diff['to_open']) + len(diff['to_close'])
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Shadow Trader - Alpaca to TastyTrade Sync')
    parser.add_argument('--sync', action='store_true', help='Sync positions (dry run)')
    parser.add_argument('--execute', action='store_true', help='Execute sync (live)')
    parser.add_argument('--compare', action='store_true', help='Compare accounts')
    parser.add_argument('--status', action='store_true', help='Show sync status')
    args = parser.parse_args()
    
    if not any([args.sync, args.execute, args.compare, args.status]):
        args.status = True  # Default to status
    
    config = load_config()
    
    print("\n" + "="*60)
    print("SHADOW TRADER")
    print("="*60)
    print("Alpaca Paper → TastyTrade Sandbox")
    
    # Initialize clients
    print("\nConnecting to brokers...")
    
    try:
        alpaca = AlpacaClient(
            api_key=config['alpaca_paper_key'],
            secret_key=config['alpaca_paper_secret'],
            paper=True
        )
        print("  ✓ Alpaca Paper connected")
    except Exception as e:
        print(f"  ✗ Alpaca connection failed: {e}")
        sys.exit(1)
    
    try:
        # Use TastyTrade Sandbox for paper trading
        tastytrade = TastyTradeTrader(
            username=config.get('tastytrade_sandbox_username') or config['tastytrade_username'],
            password=config.get('tastytrade_sandbox_password') or config['tastytrade_password'],
            sandbox=True  # Use cert/sandbox environment
        )
        if not tastytrade._authenticated:
            raise Exception("Authentication failed")
        print(f"  ✓ TastyTrade Sandbox connected (Account: {tastytrade.account_number})")
    except Exception as e:
        print(f"  ✗ TastyTrade connection failed: {e}")
        sys.exit(1)
    
    # Execute requested action
    if args.compare or args.status:
        compare_accounts(alpaca, tastytrade)
    
    if args.sync or args.execute:
        sync_positions(alpaca, tastytrade, dry_run=not args.execute)
    
    print("\n" + "="*60 + "\n")
    
    # Cleanup
    tastytrade.close()


if __name__ == '__main__':
    main()

