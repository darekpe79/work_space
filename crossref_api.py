#%%
import json
import requests
import re

import pandas as pd


API_BASE = "https://api.crossref.org/works"
MAILTO = "dariusz.perlinski@ibl.waw.pl"  # TODO: replace with your email for polite API use

TERMS_EN = [
    "employee experience definitions conceptualization",
    "employee experience models theoretical frameworks",
    "employee experience employee lifecycle hire to retire",
    "employee experience touchpoints",
    "employee experience design",
    "employee experience management",
    "employee experience organizational culture",
    "employee experience workplace technology environment",
]

BASE_FILTERS = [
    "type:journal-article",
    "from-pub-date:2015-01-01",
    "until-pub-date:2026-12-31",
    "has-abstract:1",
    # "has-references:1",   # opcjonalnie
    # "has-full-text:1",    # opcjonalnie
]

headers = {
    "User-Agent": f"ibl-pan-crossref/0.1 (mailto:{MAILTO})",
    "Accept": "application/json",
}

for term in TERMS_EN[:1]:
    print(term)
    params = {
        "query.bibliographic": term,
        "filter": ",".join(BASE_FILTERS),
        "rows": 20,
        "sort": "relevance",
        "order": "desc",
        "select": "DOI,title,author,issued,container-title,publisher,abstract,type",
        "mailto": MAILTO,  # “polite pool”
    }

    r = requests.get(API_BASE, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    items = data.get("message", {}).get("items", []) or []
    print("\n===", term, "===")
    print("count (this page):", len(items))
    for it in items:
        doi = it.get("DOI", "")
        title = (it.get("title") or [""])[0]
        journal = (it.get("container-title") or [""])[0]
        publisher = it.get("publisher", "")
        typ = it.get("type", "")

        # rok / data
        date_parts = (it.get("issued", {}) or {}).get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None

        # autorzy jako string
        authors = it.get("author", []) or []
        authors_str = "; ".join(
            [f"{a.get('family','')} {a.get('given','')}".strip() for a in authors]
        )

        # abstrakt: Crossref często zwraca JATS XML w stringu
        abstract_raw = it.get("abstract", "") or ""
        abstract_txt = re.sub(r"<[^>]+>", " ", abstract_raw)  # brutalne zdjęcie tagów
        abstract_txt = re.sub(r"\s+", " ", abstract_txt).strip()

        all_rows.append({
            "search_term": term,
            "DOI": doi,
            "title": title,
            "year": year,
            "journal": journal,
            "publisher": publisher,
            "type": typ,
            "authors": authors_str,
            "abstract": abstract_txt,
        })

df = pd.DataFrame(all_rows)

# zapis
out_path = "crossref_employee_experience.xlsx"
df.to_excel(out_path, index=False)
print("Saved:", out_path, "rows:", len(df))

import requests

doi_test = "10.31002/rekomen.v7i1"

url = f"https://api.crossref.org/works/{doi_test}"
r = requests.get(url, timeout=30)

print("status:", r.status_code)

if r.status_code == 200:
    data = r.json()["message"]
    print("type:", data.get("type"))
    print("title:", data.get("title"))
    print("container-title:", data.get("container-title"))
    print("publisher:", data.get("publisher"))
else:
    print("Not found in Crossref")