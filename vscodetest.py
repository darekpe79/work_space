#%%
import json
import requests


API_BASE = "https://api.crossref.org/works"
MAILTO = "your_email@example.com"  # TODO: replace with your email for polite API use

query = "machine learning fairness"
rows = 5

params = {
    "query": query,
    "rows": rows,
    "mailto": MAILTO,
}
headers = {
    "User-Agent": f"crossref-demo/0.1 (mailto:{MAILTO})",
    "Accept": "application/json",
}

resp = requests.get(API_BASE, params=params, headers=headers, timeout=20)
resp.raise_for_status()
data = resp.json()

print(json.dumps(data, indent=2, ensure_ascii=False))

# %%
