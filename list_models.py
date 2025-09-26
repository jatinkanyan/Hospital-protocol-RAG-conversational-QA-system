import os
import json
from dotenv import load_dotenv
import urllib.request

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("GOOGLE_API_KEY not found in environment (.env).")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

with urllib.request.urlopen(url) as resp:
    body = resp.read().decode("utf-8")
    data = json.loads(body)

# print models in a readable form
models = data.get("models") or data.get("model") or data.get("models", [])
if not models:
    print(json.dumps(data, indent=2))
else:
    for m in models:
        name = m.get("name") or m.get("model")
        display = m.get("displayName") or m.get("display_name") or ""
        print((name or str(m)) + (f" - {display}" if display else ""))
