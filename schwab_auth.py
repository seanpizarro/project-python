#!/usr/bin/env python3
"""
Schwab OAuth2 Authorization Helper
Gets a fresh refresh token via browser login
"""

import os
import sys
import webbrowser
import base64
import urllib.parse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from dotenv import load_dotenv, set_key

load_dotenv('.env')

# Schwab OAuth config
CLIENT_ID = os.environ.get('SCHWAB_APP_KEY', '')
CLIENT_SECRET = os.environ.get('SCHWAB_CLIENT_SECRET', '')
REDIRECT_URI = 'https://127.0.0.1:8182/callback'
TOKEN_URL = 'https://api.schwabapi.com/v1/oauth/token'
AUTH_URL = 'https://api.schwab.com/v1/oauth/authorize'

# Global to store auth code
auth_code = None
server_ready = threading.Event()


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        
        # Parse the callback URL
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        
        if 'code' in params:
            auth_code = params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''
                <html><body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>&#10004; Authorization Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                </body></html>
            ''')
        else:
            error = params.get('error', ['Unknown error'])[0]
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f'''
                <html><body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>&#10008; Authorization Failed</h1>
                <p>Error: {error}</p>
                </body></html>
            '''.encode())
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def start_callback_server():
    """Start HTTPS callback server"""
    import ssl
    import tempfile
    
    # Generate self-signed cert for HTTPS
    cert_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem')
    key_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem')
    
    # Create self-signed cert using openssl
    os.system(f'openssl req -x509 -newkey rsa:2048 -keyout {key_file.name} -out {cert_file.name} '
              f'-days 1 -nodes -subj "/CN=localhost" 2>/dev/null')
    
    server = HTTPServer(('127.0.0.1', 8182), CallbackHandler)
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file.name, key_file.name)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    
    server_ready.set()
    server.handle_request()  # Handle one request then stop
    
    # Cleanup temp files
    os.unlink(cert_file.name)
    os.unlink(key_file.name)
    
    return server


def get_authorization_url():
    """Build Schwab authorization URL"""
    params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'scope': 'api'
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(code):
    """Exchange authorization code for tokens"""
    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    
    headers = {
        'Authorization': f'Basic {basic}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI
    }
    
    response = requests.post(TOKEN_URL, headers=headers, data=data, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    global auth_code
    
    print("\n" + "="*60)
    print("SCHWAB OAUTH2 AUTHORIZATION")
    print("="*60)
    
    if not CLIENT_ID or not CLIENT_SECRET:
        print("\n❌ Missing SCHWAB_APP_KEY or SCHWAB_CLIENT_SECRET in .env")
        sys.exit(1)
    
    print(f"\nClient ID: {CLIENT_ID[:15]}...")
    print(f"Redirect URI: {REDIRECT_URI}")
    
    # Start callback server in background
    print("\n[1/4] Starting callback server...")
    server_thread = threading.Thread(target=start_callback_server, daemon=True)
    server_thread.start()
    server_ready.wait(timeout=5)
    print("      ✓ Server listening on https://127.0.0.1:8182")
    
    # Open browser
    auth_url = get_authorization_url()
    print("\n[2/4] Opening browser for Schwab login...")
    print(f"      URL: {auth_url[:80]}...")
    print("\n      ⚠️  If browser doesn't open, copy this URL manually:")
    print(f"      {auth_url}")
    
    webbrowser.open(auth_url)
    
    # Wait for callback
    print("\n[3/4] Waiting for authorization...")
    print("      (Log in to Schwab and authorize the app)")
    
    server_thread.join(timeout=120)  # Wait up to 2 minutes
    
    if not auth_code:
        print("\n❌ Timeout waiting for authorization")
        sys.exit(1)
    
    print(f"      ✓ Received authorization code")
    
    # Exchange for tokens
    print("\n[4/4] Exchanging code for tokens...")
    try:
        tokens = exchange_code_for_tokens(auth_code)
        
        access_token = tokens.get('access_token')
        refresh_token = tokens.get('refresh_token')
        expires_in = tokens.get('expires_in', 1800)
        
        if not refresh_token:
            print("❌ No refresh token in response")
            print(tokens)
            sys.exit(1)
        
        # Save to .env
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        set_key(env_path, 'SCHWAB_REFRESH_TOKEN', refresh_token)
        
        print(f"      ✓ Access token obtained (expires in {expires_in}s)")
        print(f"      ✓ Refresh token saved to .env")
        
        # Test the token
        print("\n[TEST] Fetching account balance...")
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        resp = requests.get('https://api.schwabapi.com/trader/v1/accounts', 
                          headers=headers, timeout=20)
        
        if resp.status_code == 200:
            accounts = resp.json()
            print("\n✓ Schwab accounts:")
            for acct in accounts:
                sec = acct.get('securitiesAccount', {})
                acct_id = sec.get('accountNumber', 'unknown')
                bal = sec.get('currentBalances', {})
                liq = bal.get('liquidationValue', 0)
                cash = bal.get('cashBalance', bal.get('availableFunds', 0))
                bp = bal.get('buyingPower', 0)
                print(f"   - {acct_id}: Equity ${liq:,.2f} | Cash ${cash:,.2f} | BP ${bp:,.2f}")
        else:
            print(f"   ⚠️ Account fetch returned {resp.status_code}")
            print(resp.text[:300])
        
        print("\n" + "="*60)
        print("✓ Authorization complete! You can now use Schwab API.")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Token exchange failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

