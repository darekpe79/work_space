import requests
import json

TERMS_EN = [
    # A. Employee experience: definicje, modele, wymiary
    "employee experience (EX) – definitions and conceptualization",
    "employee experience – models and theoretical frameworks",
    "employee experience – employee lifecycle (hire to retire)",
    "employee experience – process-based and organizational view",
    "employee experience as a relational and dynamic construct",
    "multidimensionality of employee experience",
    "employee experience touchpoints",
    "employee experience design",
    "employee experience management",
    "employee experience and organizational culture",
    "employee experience and workplace technology/environment",

    # B. Pojęcia pokrewne i integracja konstruktu
    "job satisfaction",
    "work engagement",
    "employee engagement",
    "workplace well-being",
    "burnout",
    "psychological safety",
    "organizational justice",
    "perceived organizational support",
    "value congruence",
    "person-organization fit",
    "meaningful work",
    "work meaningfulness",
    "organizational climate",

    # C. Zachowania istotne ekonomicznie: mechanizmy i konsekwencje
    "organizational commitment",
    "voluntary vs necessity-based loyalty",
    "employee retention",
    "employee turnover",
    "turnover intention",
    "turnover costs",
    "employee turnover cost",
    "recruitment, onboarding and ramp-up costs",
    "organizational knowledge loss costs",
    "decision to stay as an economic decision",
    "job embeddedness",
    "perceived alternatives",
    "discretionary effort",
    "work withdrawal behaviors",
    "low engagement",
    "minimum performance",
    "organizational citizenship behavior (OCB)",
    "work productivity",
    "work quality",
    "errors",
    "rework",

    # D. Absencja, presenteeism i zdrowie jako kanały ekonomiczne
    "employee absenteeism – definitions and determinants",
    "economic cost of absenteeism",
    "work-related sickness absence",
    "presenteeism – definitions and determinants",
    "economic cost of presenteeism",
    "productivity loss",
    "absenteeism vs presenteeism relationship",
    "job stress and absenteeism/presenteeism",
    "burnout and absenteeism/presenteeism",
    "well-being as a mediating variable",

    # E. Efektywność organizacyjna i efektywność procesów
    "organizational effectiveness – measures and models",
    "cost efficiency",
    "indirect costs",
    "internal process efficiency",
    "process improvement",
    "continuous improvement",
    "Lean management in administration",
    "bureaucratization and process formalization",
    "administrative process digitalization",
    "administrative service quality",
    "internal customer concept",
    "cross-functional collaboration",
    "project-based work",

    # F. Kontekst: uczelnia publiczna, administracja akademicka, sektor publiczny
    "university administration",
    "academic administration",
    "central university administration",
    "shared services",
    "non-academic staff",
    "professional services staff",
    "public sector human resource management",
    "public sector employment characteristics",
    "public service motivation (PSM)",
    "university as a public organization",
    "university governance",
    "administration–faculty relations",
    "university competitiveness",
    "competitive advantage",
    "demographic decline and higher education challenges",
    "Polish classical universities",
    "KRUP",

    # G. Metodyka i operacjonalizacja EX
    "measuring employee experience – instruments and scales",
    "employee satisfaction surveys – methodology",
    "engagement surveys – instruments and validation",
    "mediation modeling (EX -> well-being -> behaviors)",
    "case study in organizational research",
    "data triangulation",
    "HRIS/HR analytics data as research sources",
    "metrics: retention, absenteeism, turnover",
    "comparative analysis of universities",

    # H. Raporty / benchmarki (przykłady)
    "reports: employee experience trends",
    "reports: employee engagement (data)",
    "reports: labor market mobility and retention",
    "reports: absenteeism and presenteeism – costs",
    "HR benchmarks",
    "HR analytics",
    "reports: workplace well-being and mental health",
    "reports: remote/hybrid work organization",
    "reports: higher education management",
]


url = "https://api.gotriple.eu/api/skg-if/products"

params = {
    "filter": "cf.search.title_abstract:artificial intelligence",
    "page": 1,
    "page_size": 100,
}



headers = {
    "Accept": "application/json"
}

r = requests.get(url, params=params, headers=headers, timeout=30)
r.raise_for_status()

data = r.json()


print(json.dumps(data, indent=2, ensure_ascii=False))

import math
meta = data["meta"]
total_results = meta["count"]
page_size = int(meta["page_size"])

total_pages = math.ceil(total_results / page_size)
print(total_pages)

print(total_pages)
rows = []

results = data["results"]
for i, rec in enumerate(results[:5], 0):
    row = {} 
    row["local_identifier"] = rec.get("local_identifier")
    row["entity_type"] = rec.get("entity_type")
    row["product_type"] = rec.get("product_type")
    titles = rec.get("titles", {})
    row["title_langs"] = ",".join(titles.keys()) if isinstance(titles, dict) else None
    
    titles = rec.get("titles", {})
    
    all_titles = []
    if isinstance(titles, dict):
        for lang, vals in titles.items():
            if isinstance(vals, list):
                for v in vals:
                    all_titles.append(f"{lang}: {v}")
    
    row["titles_all"] = " | ".join(all_titles) if all_titles else None
    row["title_langs"] = ",".join(titles.keys()) if isinstance(titles, dict) else None
    abstracts = rec.get("abstracts", {})
    abstract = None
    if isinstance(abstracts, dict):
        if "en" in abstracts and abstracts["en"]:
            abstract = abstracts["en"][0]
        else:
            for v in abstracts.values():
                if isinstance(v, list) and v:
                    abstract = v[0]
                    break
    row["abstract"] = abstract
    # identifiers
    ids = rec.get("identifiers", [])
    row["identifiers_n"] = len(ids) if isinstance(ids, list) else 0
    
    schemes = []
    doi = None
    if isinstance(ids, list):
        for it in ids:
            if isinstance(it, dict):
                sch = it.get("scheme")
                val = it.get("value")
                if sch:
                    schemes.append(sch)
                if sch == "doi" and val and doi is None:
                    doi = val
    row["id_schemes"] = ",".join(sorted(set(schemes))) if schemes else None
    row["doi"] = doi
    
    # topics
    topics = rec.get("topics", [])
    terms = []
    if isinstance(topics, list):
        for t in topics:
            if isinstance(t, dict) and t.get("term"):
                terms.append(t["term"])
    row["topics_n"] = len(terms)
    row["topics"] = "; ".join(terms) if terms else None
    
    # contributions
    contribs = rec.get("contributions", [])
    authors = []
    publishers = []
    if isinstance(contribs, list):
        for c in contribs:
            if isinstance(c, dict):
                role = c.get("role")
                by = c.get("by")
                if role == "author" and by:
                    authors.append(by)
                if role == "publisher" and by:
                    publishers.append(by)
    row["authors_n"] = len(authors)
    row["authors"] = "; ".join(authors) if authors else None
    row["publishers"] = "; ".join(publishers) if publishers else None
    
    # manifestations
    mans = rec.get("manifestations", [])
    row["manifestations_n"] = len(mans) if isinstance(mans, list) else 0
    
    pub_date = None
    access_status = None
    licence = None
    if isinstance(mans, list) and mans and isinstance(mans[0], dict):
        m0 = mans[0]
        dates = m0.get("dates", {})
        if isinstance(dates, dict):
            pub_date = dates.get("publication")
        ar = m0.get("access_rights", {})
        if isinstance(ar, dict):
            access_status = ar.get("status")
        licence = m0.get("licence")
    
    row["publication"] = pub_date
    row["access"] = access_status
    row["licence"] = licence
    
    rows.append(row)
    
    print("  titles_all:", row["titles_all"])
    print("  doi:", row["doi"])
    print("  pub/access:", row["publication"], "/", row["access"])
    print("  topics_n:", row["topics_n"], "authors_n:", row["authors_n"])
    
  
search_info = {
    "url": url,
    "filter": params.get("filter"),
    "page": params.get("page"),
    "page_size": params.get("page_size"),
    "accept_header": headers.get("Accept"),
    "meta.count": data.get("meta", {}).get("count"),
    "meta.page": data.get("meta", {}).get("page"),
    "meta.page_size": data.get("meta", {}).get("page_size"),
    "meta.total_pages": total_pages,
}    
import pandas as pd

df = pd.DataFrame(rows)
df_search = pd.DataFrame([search_info])

with pd.ExcelWriter("gotriple_test_page1.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="results", index=False)
    df_search.to_excel(writer, sheet_name="search_params", index=False)
    
    
#%%    
import requests
import json
import math
import time
from urllib.parse import quote
import pandas as pd


TERMS_EN = [
    "employee experience (EX) – definitions and conceptualization",
    "employee experience – models and theoretical frameworks",
    "employee experience – employee lifecycle (hire to retire)",
    "employee experience – process-based and organizational view",
    "employee experience as a relational and dynamic construct",
    "multidimensionality of employee experience",
    "employee experience touchpoints",
    "employee experience design",
    "employee experience management",
    "employee experience and organizational culture",
    "employee experience and workplace technology/environment",
    "job satisfaction",
    "work engagement",
    "employee engagement",
    "workplace well-being",
    "burnout",
    "psychological safety",
    "organizational justice",
    "perceived organizational support",
    "value congruence",
    "person-organization fit",
    "meaningful work",
    "work meaningfulness",
    "organizational climate",
    "organizational commitment",
    "voluntary vs necessity-based loyalty",
    "employee retention",
    "employee turnover",
    "turnover intention",
    "turnover costs",
    "employee turnover cost",
    "recruitment, onboarding and ramp-up costs",
    "organizational knowledge loss costs",
    "decision to stay as an economic decision",
    "job embeddedness",
    "perceived alternatives",
    "discretionary effort",
    "work withdrawal behaviors",
    "low engagement",
    "minimum performance",
    "organizational citizenship behavior (OCB)",
    "work productivity",
    "work quality",
    "errors",
    "rework",
    "employee absenteeism – definitions and determinants",
    "economic cost of absenteeism",
    "work-related sickness absence",
    "presenteeism – definitions and determinants",
    "economic cost of presenteeism",
    "productivity loss",
    "absenteeism vs presenteeism relationship",
    "job stress and absenteeism/presenteeism",
    "burnout and absenteeism/presenteeism",
    "well-being as a mediating variable",
    "organizational effectiveness – measures and models",
    "cost efficiency",
    "indirect costs",
    "internal process efficiency",
    "process improvement",
    "continuous improvement",
    "Lean management in administration",
    "bureaucratization and process formalization",
    "administrative process digitalization",
    "administrative service quality",
    "internal customer concept",
    "cross-functional collaboration",
    "project-based work",
    "university administration",
    "academic administration",
    "central university administration",
    "shared services",
    "non-academic staff",
    "professional services staff",
    "public sector human resource management",
    "public sector employment characteristics",
    "public service motivation (PSM)",
    "university as a public organization",
    "university governance",
    "administration–faculty relations",
    "university competitiveness",
    "competitive advantage",
    "demographic decline and higher education challenges",
    "Polish classical universities",
    "KRUP",
    "measuring employee experience – instruments and scales",
    "employee satisfaction surveys – methodology",
    "engagement surveys – instruments and validation",
    "mediation modeling (EX -> well-being -> behaviors)",
    "case study in organizational research",
    "data triangulation",
    "HRIS/HR analytics data as research sources",
    "metrics: retention, absenteeism, turnover",
    "comparative analysis of universities",
    "reports: employee experience trends",
    "reports: employee engagement (data)",
    "reports: labor market mobility and retention",
    "reports: absenteeism and presenteeism – costs",
    "HR benchmarks",
    "HR analytics",
    "reports: workplace well-being and mental health",
    "reports: remote/hybrid work organization",
    "reports: higher education management",
]


API_URL = "https://api.gotriple.eu/api/skg-if/products"
HEADERS = {"Accept": "application/json"}
PAGE_SIZE = 100
SLEEP_BETWEEN_REQUESTS = 0.15
TIMEOUT = 30
MAX_PAGES_PER_TERM = None  # np. 2 do testów / None = wszystko

# wydruki kontrolne
PRINT_EVERY_N_PAGES = 10           # progress co ile stron
PRINT_FIRST_N_LOCAL_IDS = 3        # pokaż kilka przykładowych local_identifier z 1 strony
PRINT_NEW_UNIQUES_EVERY_TERM = True


def gotriple_document_url(local_identifier: str) -> str | None:
    if not local_identifier:
        return None
    return "https://gotriple.eu/explore/document/" + quote(local_identifier, safe="")


def build_filter(term: str) -> str:
    term_clean = term.replace('"', "").strip()
    # AND wg dokumentacji: przecinek
    return f'product_type:literature,cf.search.title_abstract:"{term_clean}"'


def parse_record(rec: dict, term: str) -> dict:
    row = {}
    row["term"] = term

    row["local_identifier"] = rec.get("local_identifier")
    row["gotriple_url"] = gotriple_document_url(row["local_identifier"])

    row["entity_type"] = rec.get("entity_type")
    row["product_type"] = rec.get("product_type")

    titles = rec.get("titles", {})
    all_titles = []
    if isinstance(titles, dict):
        for lang, vals in titles.items():
            if isinstance(vals, list):
                for v in vals:
                    all_titles.append(f"{lang}: {v}")
    row["titles_all"] = " | ".join(all_titles) if all_titles else None
    row["title_langs"] = ",".join(titles.keys()) if isinstance(titles, dict) else None

    abstracts = rec.get("abstracts", {})
    abstract = None
    if isinstance(abstracts, dict):
        if "en" in abstracts and abstracts["en"]:
            abstract = abstracts["en"][0]
        else:
            for v in abstracts.values():
                if isinstance(v, list) and v:
                    abstract = v[0]
                    break
    row["abstract"] = abstract

    ids = rec.get("identifiers", [])
    row["identifiers_n"] = len(ids) if isinstance(ids, list) else 0

    schemes = []
    doi = None
    if isinstance(ids, list):
        for it in ids:
            if isinstance(it, dict):
                sch = it.get("scheme")
                val = it.get("value")
                if sch:
                    schemes.append(sch)
                if sch == "doi" and val and doi is None:
                    doi = val
    row["id_schemes"] = ",".join(sorted(set(schemes))) if schemes else None
    row["doi"] = doi

    topics = rec.get("topics", [])
    terms_list = []
    if isinstance(topics, list):
        for t in topics:
            if isinstance(t, dict) and t.get("term"):
                terms_list.append(t["term"])
    row["topics_n"] = len(terms_list)
    row["topics"] = "; ".join(terms_list) if terms_list else None

    contribs = rec.get("contributions", [])
    authors = []
    publishers = []
    if isinstance(contribs, list):
        for c in contribs:
            if isinstance(c, dict):
                role = c.get("role")
                by = c.get("by")
                if role == "author" and by:
                    authors.append(by)
                if role == "publisher" and by:
                    publishers.append(by)
    row["authors_n"] = len(authors)
    row["authors"] = "; ".join(authors) if authors else None
    row["publishers"] = "; ".join(publishers) if publishers else None

    mans = rec.get("manifestations", [])
    row["manifestations_n"] = len(mans) if isinstance(mans, list) else 0

    pub_date = None
    access_status = None
    licence = None
    if isinstance(mans, list) and mans and isinstance(mans[0], dict):
        m0 = mans[0]
        dates = m0.get("dates", {})
        if isinstance(dates, dict):
            pub_date = dates.get("publication")
        ar = m0.get("access_rights", {})
        if isinstance(ar, dict):
            access_status = ar.get("status")
        licence = m0.get("licence")

    row["publication"] = pub_date
    row["access"] = access_status
    row["licence"] = licence

    return row


# --- main ---
session = requests.Session()

all_rows = []
term_stats = []
terms_no_results = []

dedup = {}        # local_identifier -> row (pierwszy napotkany)
dedup_terms = {}  # local_identifier -> set(terms)

total_terms = len(TERMS_EN)
global_start = time.time()

for idx, term in enumerate(TERMS_EN, 1):
    term_start = time.time()
    flt = build_filter(term)

    term_total = 0
    pages_downloaded = 0
    any_results = False

    # --- request page 1 (meta) ---
    params = {"filter": flt, "page": 1, "page_size": PAGE_SIZE}
    r = session.get(API_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    meta = data.get("meta", {}) or {}
    count = int(meta.get("count", 0) or 0)
    page_size = int(meta.get("page_size", PAGE_SIZE) or PAGE_SIZE)
    total_pages = math.ceil(count / page_size) if count else 0

    if MAX_PAGES_PER_TERM is not None:
        total_pages = min(total_pages, MAX_PAGES_PER_TERM)

    # --- wydruk start termu ---
    print("\n" + "=" * 90)
    print(f"[{idx}/{total_terms}] TERM: {term}")
    print(f"filter: {flt}")
    print(f"meta.count={count} | page_size={page_size} | total_pages={total_pages}")
    if total_pages == 0:
        print("-> 0 stron (brak wyników wg meta.count).")
        terms_no_results.append(term)
        term_stats.append({
            "term_idx": idx,
            "term": term,
            "filter_used": flt,
            "meta_count": count,
            "pages_downloaded": 0,
            "rows_downloaded": 0,
            "page_size": PAGE_SIZE,
            "unique_local_id_added": 0,
        })
        continue

    # snapshot ilu unikatów było przed tym termem
    uniques_before = len(dedup)

    # --- process page 1 ---
    results = data.get("results", []) or []
    pages_downloaded = 1
    if results:
        any_results = True

    if PRINT_FIRST_N_LOCAL_IDS and results:
        sample_ids = [x.get("local_identifier") for x in results[:PRINT_FIRST_N_LOCAL_IDS]]
        print("sample local_identifier (page 1):")
        for s in sample_ids:
            print("  -", s)

    for rec in results:
        row = parse_record(rec, term)
        all_rows.append(row)

        lid = row.get("local_identifier")
        if not lid:
            continue
        if lid not in dedup:
            dedup[lid] = row.copy()
            dedup_terms[lid] = set()
        dedup_terms[lid].add(term)

    term_total += len(results)
    print(f"page 1/{total_pages}: rows={len(results)} | term_rows_total={term_total} | uniques_now={len(dedup)}")

    # --- process pages 2..N ---
    for page in range(2, total_pages + 1):
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        params = {"filter": flt, "page": page, "page_size": PAGE_SIZE}
        r = session.get(API_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", []) or []
        pages_downloaded = page

        if results:
            any_results = True

        for rec in results:
            row = parse_record(rec, term)
            all_rows.append(row)

            lid = row.get("local_identifier")
            if not lid:
                continue
            if lid not in dedup:
                dedup[lid] = row.copy()
                dedup_terms[lid] = set()
            dedup_terms[lid].add(term)

        term_total += len(results)

        # wydruk progress co N stron + na końcu
        if (page % PRINT_EVERY_N_PAGES == 0) or (page == total_pages):
            print(f"page {page}/{total_pages}: rows={len(results)} | term_rows_total={term_total} | uniques_now={len(dedup)}")

    if not any_results:
        terms_no_results.append(term)

    uniques_added = len(dedup) - uniques_before
    elapsed = time.time() - term_start
    if PRINT_NEW_UNIQUES_EVERY_TERM:
        print(f"TERM DONE: downloaded_rows={term_total} | pages={pages_downloaded} | unique_local_id_added={uniques_added} | time={elapsed:.1f}s")

    term_stats.append({
        "term_idx": idx,
        "term": term,
        "filter_used": flt,
        "meta_count": count,
        "pages_downloaded": pages_downloaded,
        "rows_downloaded": term_total,
        "page_size": PAGE_SIZE,
        "unique_local_id_added": uniques_added,
    })

# dopisz terms_matched do zdeduplikowanych wyników
dedup_rows = []
for lid, row in dedup.items():
    row2 = row.copy()
    matched = sorted(dedup_terms.get(lid, set()))
    row2["terms_matched"] = " | ".join(matched)
    row2["terms_matched_n"] = len(matched)
    dedup_rows.append(row2)

df_all = pd.DataFrame(all_rows)
df_dedup = pd.DataFrame(dedup_rows)
df_terms = pd.DataFrame(term_stats)
df_no = pd.DataFrame({"term_no_results": terms_no_results})

out_main = "gotriple_terms_results.xlsx"
out_terms_only = "gotriple_terms_only.xlsx"

with pd.ExcelWriter(out_main, engine="openpyxl") as writer:
    df_dedup.to_excel(writer, sheet_name="results_dedup", index=False)
    df_all.to_excel(writer, sheet_name="results_all_with_term", index=False)
    df_terms.to_excel(writer, sheet_name="terms_stats", index=False)
    df_no.to_excel(writer, sheet_name="terms_no_results", index=False)

with pd.ExcelWriter(out_terms_only, engine="openpyxl") as writer:
    pd.DataFrame({"term": TERMS_EN}).to_excel(writer, sheet_name="terms_all", index=False)
    df_no.to_excel(writer, sheet_name="terms_no_results", index=False)
    df_terms.to_excel(writer, sheet_name="terms_stats", index=False)

global_elapsed = time.time() - global_start
print("\n" + "=" * 90)
print("DONE")
print("Main:", out_main)
print("Terms only:", out_terms_only)
print(f"All rows (with term duplicates): {len(df_all)}")
print(f"Dedup rows (unique local_identifier): {len(df_dedup)}")
print(f"Terms with 0 results: {len(terms_no_results)}")
print(f"Total time: {global_elapsed:.1f}s")

#%%

import requests
import math
from urllib.parse import quote_plus

API_URL = "https://api.gotriple.eu/api/skg-if/products"
HEADERS = {"Accept": "application/json"}
PAGE_SIZE = 100
TIMEOUT = 30

def run_test(label: str, flt: str, page: int = 1):
    params = {"filter": flt, "page": page, "page_size": PAGE_SIZE}

    # podgląd “jak to idzie po drucie” (czasem pomaga zobaczyć czy znaki się nie psują)
    qs = "&".join([f"{k}={quote_plus(str(v))}" for k, v in params.items()])
    print("\n" + "="*80)
    print(f"TEST: {label}")
    print("filter:", flt)
    print("request_url:", f"{API_URL}?{qs}")

    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
    print("http_status:", r.status_code)
    r.raise_for_status()
    data = r.json()

    meta = data.get("meta", {}) or {}
    count = meta.get("count")
    page_size = int(meta.get("page_size") or PAGE_SIZE)
    total_pages = math.ceil(count / page_size) if isinstance(count, int) and count > 0 else 0

    print(f"meta.count={count} | page_size={page_size} | total_pages={total_pages}")

    results = data.get("results", []) or []
    print(f"page_rows={len(results)}")
    print("sample local_identifier:")
    for rec in results[:3]:
        print("  -", rec.get("local_identifier"))

# -----------------------------------------------------------------------------
# ZMIENIASZ tylko TERM, reszta to testy diagnostyczne
TERM = "employee experience (EX) – definitions and conceptualization"

# 1) Fraza w cudzysłowie (tak jak robisz)
flt_phrase = f'product_type:literature,cf.search.title_abstract:"{TERM}"'

# 2) Bez cudzysłowu
flt_no_quotes = f"product_type:literature,cf.search.title_abstract:{TERM}"

# 3) Minimalny token (żeby zobaczyć skalę działania tego pola)
flt_employee = 'product_type:literature,cf.search.title_abstract:employee'

# 4) Kontrolny nonsens (powinno dać 0, jeśli to jest sensowny full-text)
flt_nonsense = 'product_type:literature,cf.search.title_abstract:"zzqwxptkqqq 12345 no_such_phrase"'

# 5) Dodatkowy test: tylko product_type (żeby zobaczyć “ile jest literatury” ogólnie)
flt_only_type = "product_type:literature"

run_test("1) phrase with quotes", flt_phrase)
run_test("2) same without quotes", flt_no_quotes)
run_test("3) simple token 'employee'", flt_employee)
run_test("4) nonsense control", flt_nonsense)
run_test("5) only product_type:literature", flt_only_type)

#%%
import pandas as pd

df = pd.read_excel("C:/Users/darek/gotriple_B_employee_engagement.xlsx")

PHRASE = "employee engagement"

def contains_phrase(row):
    text = ""
    if pd.notna(row.get("titles_all")):
        text += " " + row["titles_all"]
    if pd.notna(row.get("abstract")):
        text += " " + row["abstract"]
    return PHRASE in text.lower()

df["has_employee_engagement"] = df.apply(contains_phrase, axis=1)

# statystyka
total = len(df)
hits = df["has_employee_engagement"].sum()

print(f"Total records: {total}")
print(f"With exact phrase '{PHRASE}': {hits}")
print(f"Without exact phrase: {total - hits}")

#%%
import requests, re
from urllib.parse import quote

API_URL = "https://api.gotriple.eu/api/skg-if/products"
HEADERS = {"Accept": "application/json"}
TIMEOUT = 30

PHRASE = "employee engagement"

def collect_text(rec):
    titles = rec.get("titles", {}) or {}
    abstracts = rec.get("abstracts", {}) or {}
    t = ""

    # wszystkie tytuły
    if isinstance(titles, dict):
        for lang, vals in titles.items():
            if isinstance(vals, list):
                for v in vals:
                    t += " " + str(v)

    # wszystkie abstrakty (dla testu)
    if isinstance(abstracts, dict):
        for lang, vals in abstracts.items():
            if isinstance(vals, list):
                for v in vals:
                    t += " " + str(v)

    t = re.sub(r"\s+", " ", t).strip()
    return t

flt = 'product_type:literature,cf.search.title_abstract:employee engagement'
flt = 'product_type:literature,cf.search.title_abstract:"employee engagement"'
flt = 'product_type:literature,cf.search.title_abstract:employee AND engagement'
flt = 'product_type:literature,cf.search.title_abstract:employee OR engagement'
flt = 'product_type:literature,cf.search.title_abstract:employee,engagement'
flt = (
    'product_type:literature,'
    'cf.search.title_abstract:employee,'
    'cf.search.title_abstract:engagement'
)
params = {"filter": flt, "page": 1, "page_size": 100}

r = requests.get(API_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
r.raise_for_status()
data = r.json()

results = data.get("results", []) or []
texts = [collect_text(rec) for rec in results]
hits = [("employee engagement" in txt.lower()) for txt in texts]

print("API meta.count:", data.get("meta", {}).get("count"))
print("Rows on page:", len(results))
print("Exact phrase present in returned page:", sum(hits), "/", len(hits))
print("Example request URL:", r.url)
miss = []
for rec, txt, ok in zip(results, texts, hits):
    if not ok:
        miss.append({
            "local_identifier": rec.get("local_identifier"),
            "titles": rec.get("titles"),
            "abstracts": rec.get("abstracts"),
            "text_snippet": txt
        })

print("\nMISSING exact phrase count:", len(miss))
for m in miss[:10]:
    print("\nlocal_identifier:", m["local_identifier"])
    print("snippet:", m["text_snippet"])
