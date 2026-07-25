import sys
import json
import time
import random
import requests
import asyncio
import aiohttp
from parameters import (
    REGION, DELAY, UNIVERSE, TRUNCATE, NEUTRALIZATION, DECAY,
    MAX_CONCURRENT_REQUESTS, RETRY_DELAY, MAX_RETRIES
)

class WQBrainAPI:
    def __init__(self, credentials_path="credentials.json", dry_run=False):
        self.dry_run = dry_run
        self.session = requests.Session()
        self.base_url = "https://api.worldquantbrain.com"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if not self.dry_run:
            self._load_credentials(credentials_path)
            self.authenticate()
        else:
            print("[Mock] Skipping authentication in dry-run mode.")

    def _load_credentials(self, path):
        try:
            with open(path, 'r') as f:
                creds = json.load(f)
                self.email = creds.get("email")
                self.password = creds.get("password")
                self.session_id = creds.get("session_id")
        except FileNotFoundError:
            raise Exception(f"Credentials file {path} not found.")

    def authenticate(self):
        # Direct Session Cookie Bypass & Interactive Prompt
        while True:
            if self.session_id and len(self.session_id) > 5:
                print("[*] Using session_id for authentication...")
                self.session.cookies.set("csession", self.session_id, domain="api.worldquantbrain.com")
                
                # Verify if cookie is active
                check_response = self.session.get(f"{self.base_url}/users/self")
                if check_response.status_code == 200:
                    print(f"[+] Direct Session Cookie is VALID! Logged in as {self.email}")
                    return
                else:
                    print("[!] Session Cookie has EXPIRED or is invalid!")
            
            # If expired or missing, prompt the user
            print("\n[!] Valid 'csession' token is required to bypass WorldQuant Brain MFA.")
            if sys.stdin.isatty() or True: # Always prompt; if in background, EOFError will crash it cleanly
                try:
                    self.session_id = input("[?] Please paste your new 'csession' cookie from the browser: ").strip()
                except EOFError:
                    raise Exception("Background cron run detected with expired session. Please run manually to update token.")
                    
                if not self.session_id:
                    print("[!] No token provided. Exiting.")
                    sys.exit(1)
                
                # Save the new token back to credentials.json
                with open("credentials.json", "r") as f:
                    creds = json.load(f)
                creds["session_id"] = self.session_id
                with open("credentials.json", "w") as f:
                    json.dump(creds, f, indent=4)
                print("[*] Saved new session token to credentials.json for future runs.\n")
                # Loop back to top to verify the newly entered cookie

    async def async_simulate(self, session, alpha_expr: str):
        if self.dry_run:
            # Simulate latency
            await asyncio.sleep(random.uniform(0.1, 0.5))
            # Return a mocked simulation result
            return {
                "id": f"mock_sim_{random.randint(1000, 9999)}",
                "is_sharpe": random.uniform(-1, 3.5),
                "is_turnover": random.uniform(0.005, 0.95),
                "is_fitness": random.uniform(0.1, 2.5)
            }
            
        sim_url = f"{self.base_url}/simulations"
        payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": REGION,
                "universe": UNIVERSE,
                "delay": DELAY,
                "decay": DECAY,
                "neutralization": NEUTRALIZATION,
                "truncate": TRUNCATE,
                "pasteClasses": "false",
                "language": "FASTEXPR"
            },
            "regular": alpha_expr
        }
        
        for attempt in range(MAX_RETRIES):
            async with session.post(sim_url, json=payload, headers=self.headers) as response:
                if response.status == 201:
                    data = await response.json()
                    return data
                elif response.status == 429:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return None
        return None

    async def async_batch_simulate(self, alpha_expressions: list):
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for expr in alpha_expressions:
                tasks.append(self.async_simulate(session, expr))
            
            results = await asyncio.gather(*tasks)
            return results

    def submit_alpha(self, alpha_id: str):
        if self.dry_run:
            print(f"[Mock] Alpha {alpha_id} successfully submitted!")
            return True
            
        submit_url = f"{self.base_url}/alphas"
        payload = {"simulation": alpha_id}
        response = self.session.post(submit_url, json=payload, headers=self.headers)
        if response.status_code == 201:
            print(f"Alpha {alpha_id} successfully submitted!")
            return True
        else:
            print(f"Failed to submit {alpha_id}: {response.text}")
            return False
