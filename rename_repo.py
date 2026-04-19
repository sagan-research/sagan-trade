import urllib.request
import json
import os

token = "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"
url = "https://api.github.com/repos/That-Tech-Geek/sagan"
data = json.dumps({"name": "sagan-trade"}).encode("utf-8")

req = urllib.request.Request(url, data=data, method="PATCH")
req.add_header("Authorization", f"token {token}")
req.add_header("Accept", "application/vnd.github.v3+json")
req.add_header("Content-Type", "application/json")

try:
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.status}")
        print(response.read().decode())
except Exception as e:
    print(f"Error: {e}")
