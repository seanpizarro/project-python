"""Configuration management"""

import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv


def load_config() -> Dict[str, str]:
    """Load configuration from .env file"""
    
    # Load .env file
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
    
    config = {
        # Alpaca credentials
        'alpaca_paper_key': os.getenv('ALPACA_PAPER_API_KEY', ''),
        'alpaca_paper_secret': os.getenv('ALPACA_PAPER_SECRET_KEY', ''),
        'alpaca_live_key': os.getenv('ALPACA_LIVE_API_KEY', ''),
        'alpaca_live_secret': os.getenv('ALPACA_LIVE_SECRET_KEY', ''),
        
        # TastyTrade Live (OAuth2)
        'tastytrade_account': os.getenv('TASTYTRADE_ACCOUNT', ''),
        'tastytrade_client_id': os.getenv('TASTYTRADE_CLIENT_ID', ''),
        'tastytrade_client_secret': os.getenv('TASTYTRADE_CLIENT_SECRET', ''),
        'tastytrade_refresh_token': os.getenv('TASTYTRADE_REFRESH_TOKEN', ''),
        
        # TastyTrade Sandbox (OAuth2)
        'tastytrade_sandbox_client_id': os.getenv('TASTYTRADE_SANDBOX_CLIENT_ID', ''),
        'tastytrade_sandbox_client_secret': os.getenv('TASTYTRADE_SANDBOX_CLIENT_SECRET', ''),
        
        # TastyTrade session auth (alternative)
        'tastytrade_username': os.getenv('TASTYTRADE_USERNAME', ''),
        'tastytrade_password': os.getenv('TASTYTRADE_PASSWORD', ''),
        
        # TastyTrade Sandbox (cert environment)
        'tastytrade_sandbox_username': os.getenv('TASTYTRADE_SANDBOX_USERNAME', ''),
        'tastytrade_sandbox_password': os.getenv('TASTYTRADE_SANDBOX_PASSWORD', ''),
        
        # API keys for market data
        'polygon_api_key': os.getenv('POLYGON_API_KEY', ''),
        'finnhub_api_key': os.getenv('FINNHUB_API_KEY', ''),
        
        # Schwab credentials
        'schwab_app_key': os.getenv('SCHWAB_APP_KEY', ''),
        'schwab_client_secret': os.getenv('SCHWAB_CLIENT_SECRET', ''),
        'schwab_refresh_token': os.getenv('SCHWAB_REFRESH_TOKEN', ''),
    }
    
    return config