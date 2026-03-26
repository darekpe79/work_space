


#%% piline całosciowy
import time
import json
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://citation-index-api-graphia-app1-staging.apps.bst2.paas.psnc.pl"


# =========================
# USTAWIENIA
# =========================
PDF_PATHS = [
    r"C:/Users/darek/Downloads/recenzje1_compressed.pdf",
    r"C:/Users/darek/Downloads/Art15_Tułacz.pdf", r"C:/Users/darek/Downloads/recenzje4_compressed.pdf", r"C:/Users/darek/Downloads/Art14_Dąbrowska.pdf",
    r"C:/Users/darek/Downloads/Rec02_Juchniewicz.pdf"
    
]

TEXT_EXTRACTOR = "pymupdf"
TEXT_MARKDOWN = "true"

REF_METHOD = "full_text"
REF_PROMPT = "prompts/reference_extraction.md"
REF_TEMPERATURE = 0.3

PARSE_PARSER = "llm"
PARSE_PROMPT = "prompts/reference_parsing.md"
PARSE_TEMPERATURE = 0

STATUS_POLL_INTERVAL = 2
PARSE_EMPTY_RETRY_COUNT = 1


# =========================
# FUNKCJE POMOCNICZE
# =========================
def wait_for_job(job_id, max_wait_seconds=180, poll_interval=2):
    start_time = time.time()

    while True:
        status_response = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
        status_response.raise_for_status()
        status_data = status_response.json()

        status = status_data["status"]

        if status == "completed":
            return status_data

        if status == "failed":
            raise RuntimeError(
                f"Job failed: {json.dumps(status_data, indent=2, ensure_ascii=False)}"
            )

        if time.time() - start_time > max_wait_seconds:
            raise TimeoutError(
                f"Job {job_id} did not complete within {max_wait_seconds} seconds."
            )

        time.sleep(poll_interval)


def get_job_result(job_id):
    response = requests.get(
        f"{BASE_URL}/jobs/{job_id}",
        params={"format": "json"}
    )
    response.raise_for_status()
    return response.json()


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/extract/text",
            params={
                "extractor": TEXT_EXTRACTOR,
                "markdown": TEXT_MARKDOWN
            },
            files={
                "file": (Path(pdf_path).name, f, "application/pdf")
            },
            timeout=300
        )

    response.raise_for_status()
    job_data = response.json()
    job_id = job_data["job_id"]

    wait_for_job(job_id, max_wait_seconds=180, poll_interval=STATUS_POLL_INTERVAL)
    result_data = get_job_result(job_id)

    return result_data["text"], job_id, result_data


def extract_references_from_text(text):
    response = requests.post(
        f"{BASE_URL}/extract/references",
        params={
            "method": REF_METHOD,
            "prompt_name": REF_PROMPT,
            "temperature": REF_TEMPERATURE
        },
        json={"text": text},
        timeout=300
    )

    response.raise_for_status()
    job_data = response.json()
    job_id = job_data["job_id"]

    wait_for_job(job_id, max_wait_seconds=300, poll_interval=STATUS_POLL_INTERVAL)
    result_data = get_job_result(job_id)

    return result_data.get("references", []), job_id, result_data


def parse_single_reference(raw_reference):
    response = requests.post(
        f"{BASE_URL}/parse/references",
        params={
            "parser": PARSE_PARSER,
            "prompt_name": PARSE_PROMPT,
            "temperature": PARSE_TEMPERATURE
        },
        json={"references": [raw_reference]},
        timeout=300
    )

    response.raise_for_status()
    job_data = response.json()
    job_id = job_data["job_id"]

    wait_for_job(job_id, max_wait_seconds=120, poll_interval=STATUS_POLL_INTERVAL)
    result_data = get_job_result(job_id)

    refs = result_data.get("references", [])
    if refs:
        return refs[0], result_data, job_id

    return None, result_data, job_id


def persons_to_string(persons):
    if not persons:
        return None

    parts = []
    for p in persons:
        first_name = (p.get("first_name") or "").strip()
        middle_name = (p.get("middle_name") or "").strip(" ,")
        surname = (p.get("surname") or "").strip()
        role_name = (p.get("role_name") or "").strip()

        name = " ".join(x for x in [first_name, middle_name, surname] if x)
        if role_name:
            name = f"{name} [{role_name}]"
        parts.append(name)

    return " | ".join(parts) if parts else None


def identifiers_to_string(identifiers):
    if not identifiers:
        return None

    vals = []
    for ident in identifiers:
        if isinstance(ident, dict):
            vals.append(json.dumps(ident, ensure_ascii=False))
        else:
            vals.append(str(ident))
    return " | ".join(vals)


def safe_json_dump(obj):
    if obj in (None, {}, []):
        return None
    return json.dumps(obj, ensure_ascii=False)


# =========================
# GŁÓWNY WORKFLOW
# =========================
all_rows = []
all_errors = []
all_extracted_references = []

for pdf_path in PDF_PATHS:
    source_file = Path(pdf_path).name
    print(f"\n==============================")
    print(f"PDF: {source_file}")
    print(f"==============================")

    try:
        text, text_job_id, text_result_data = extract_text_from_pdf(pdf_path)
        references, ref_job_id, ref_result_data = extract_references_from_text(text)

        print(f"Znaleziono {len(references)} referencji w pliku {source_file}")

        for idx, raw_ref in enumerate(references, start=1):
            all_extracted_references.append({
                "source_file": source_file,
                "reference_index": idx,
                "raw_reference": raw_ref,
                "text_job_id": text_job_id,
                "reference_extraction_job_id": ref_job_id
            })

        for idx, raw_ref in enumerate(references, start=1):
            print(f"  Parsowanie referencji {idx}/{len(references)}")

            parsed_ref = None
            parse_meta = None
            parse_job_id = None

            try:
                for attempt in range(PARSE_EMPTY_RETRY_COUNT + 1):
                    parsed_ref, parse_meta, parse_job_id = parse_single_reference(raw_ref)

                    if parsed_ref is not None:
                        break

                    if attempt < PARSE_EMPTY_RETRY_COUNT:
                        print(f"    Empty parse result -> retry {attempt + 1}")
                        time.sleep(1)

                if parsed_ref is None:
                    all_errors.append({
                        "source_file": source_file,
                        "reference_index": idx,
                        "raw_reference": raw_ref,
                        "error": "empty_parse_result",
                        "text_job_id": text_job_id,
                        "reference_extraction_job_id": ref_job_id,
                        "reference_parsing_job_id": parse_job_id,
                        "raw_api_result": json.dumps(parse_meta, ensure_ascii=False)
                    })
                    continue

                row = {
                    "source_file": source_file,
                    "reference_index": idx,
                    "raw_reference": raw_ref,

                    "full_title": parsed_ref.get("full_title"),
                    "journal_title": parsed_ref.get("journal_title"),
                    "publisher": parsed_ref.get("publisher"),
                    "publication_place": parsed_ref.get("publication_place"),
                    "publication_year": parsed_ref.get("publication_year"),
                    "publication_date_raw": parsed_ref.get("publication_date_raw"),
                    "ref_type": parsed_ref.get("ref_type"),

                    "authors": persons_to_string(parsed_ref.get("authors")),
                    "editors": persons_to_string(parsed_ref.get("editors")),
                    "translator": persons_to_string(parsed_ref.get("translator")),

                    "volume": parsed_ref.get("volume"),
                    "issue": parsed_ref.get("issue"),
                    "pages": parsed_ref.get("pages"),
                    "cited_range": parsed_ref.get("cited_range"),
                    "footnote_number": parsed_ref.get("footnote_number"),

                    "identifiers": identifiers_to_string(parsed_ref.get("identifiers")),
                    "raw_json": safe_json_dump(parsed_ref.get("raw")),

                    "parser": parse_meta.get("parser"),
                    "parse_count": parse_meta.get("count"),

                    "text_job_id": text_job_id,
                    "reference_extraction_job_id": ref_job_id,
                    "reference_parsing_job_id": parse_job_id,
                }

                all_rows.append(row)

            except Exception as e:
                all_errors.append({
                    "source_file": source_file,
                    "reference_index": idx,
                    "raw_reference": raw_ref,
                    "error": str(e),
                    "text_job_id": text_job_id,
                    "reference_extraction_job_id": ref_job_id,
                    "reference_parsing_job_id": parse_job_id,
                    "raw_api_result": json.dumps(parse_meta, ensure_ascii=False) if parse_meta else None
                })

    except Exception as e:
        all_errors.append({
            "source_file": source_file,
            "reference_index": None,
            "raw_reference": None,
            "error": f"pdf_level_error: {e}",
            "text_job_id": None,
            "reference_extraction_job_id": None,
            "reference_parsing_job_id": None,
            "raw_api_result": None
        })


# =========================
# ZAPIS DO XLSX
# =========================
df = pd.DataFrame(all_rows)
df_errors = pd.DataFrame(all_errors)
df_extracted = pd.DataFrame(all_extracted_references)

output_path = "citation_index_results3.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="parsed_references", index=False)
    df_errors.to_excel(writer, sheet_name="errors", index=False)
    df_extracted.to_excel(writer, sheet_name="extracted_references", index=False)

print(f"\nZapisano wynik do: {output_path}")
print(f"Liczba sparsowanych rekordów: {len(df)}")
print(f"Liczba błędów: {len(df_errors)}")
print(f"Liczba wyciągniętych referencji: {len(df_extracted)}")
