import webbrowser
import http.server
import socketserver
import urllib.parse
import json
import httpx
from sagan.config import config

PORT = 9876
FIREBASE_API_KEY = "AIzaSyC2iRJ8t34VuOMT2y1KEeWjNAaJ-vA3GBE"

class AuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        
        # We expect id_token (from Google OAuth) in the callback
        if "id_token" in params:
            id_token = params["id_token"][0]
            
            # 1. Extract User ID (sub) from JWT without full validation (for speed, 
            # in a real app you'd validate the signature)
            try:
                import base64
                parts = id_token.split('.')
                if len(parts) >= 2:
                    payload = json.loads(base64.urlsafe_b64decode(parts[1] + "==").decode('utf-8'))
                    config.user_id = payload.get("sub")
                    print(f"\nUser Identified: {config.user_id}")
            except Exception as e:
                print(f"Error parsing ID Token: {e}")

            # 2. Exchange Google ID Token for Firebase ID Token
            firebase_token = self._exchange_for_firebase(id_token)
            if firebase_token:
                config.api_key = firebase_token
                config.save_client_config()
                
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h1>Authenticated successfully.</h1><p>You can close this window now.</p>")
                
                print("Authenticated successfully with Firebase. Welcome to Sagan XAI.")
                self.server.authenticated = True
            else:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"<h1>Authentication Failed</h1><p>Could not exchange token with Firebase.</p>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h1>Missing Token</h1>")

    def _exchange_for_firebase(self, id_token: str) -> str:
        """Exchange Google ID Token for Firebase ID Token."""
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key={FIREBASE_API_KEY}"
        payload = {
            "postBody": f"id_token={id_token}&providerId=google.com",
            "requestUri": "http://localhost",
            "returnIdpCredential": True,
            "returnSecureToken": True
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data.get("idToken")
        except Exception as e:
            print(f"Firebase exchange error: {e}")
            return ""

def run_auth():
    """Start local server and open browser for OAuth flow."""
    # In a real app, this URL would point to a hosted relay that initiates Google OAuth
    # and redirects back to http://localhost:9876/?id_token=...
    auth_url = "https://usage.sagan-labs.com/auth/google"
    print(f"Opening browser for authentication: {auth_url}")
    webbrowser.open(auth_url)

    with socketserver.TCPServer(("", PORT), AuthHandler) as httpd:
        httpd.authenticated = False
        print(f"Waiting for callback on port {PORT}...")
        while not httpd.authenticated:
            httpd.handle_request()
