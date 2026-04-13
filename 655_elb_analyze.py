# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 08:16:42 2026

@author: darek
"""

import os
import pandas as pd
from pymarc import MARCMakerReader


INPUT_FILES = [
    "C:/Users/darek/Downloads/pl_pbl_articles_2024-02-08.mrk",
    "C:/Users/darek/Downloads/pl_pbl_articles_2026-01-27.mrk",]

OUTPUT_FILE = "elb_655_x_audit_allPBL.xlsx"
OUTPUT_CSV = "elb_655_x_audit_all.csv"

VALID_380 = {"Secondary literature", "Literature"}
VALID_381 = {"Drama", "Fiction", "Other", "Lyrical poetry"}


def get_major_genre_values(record, tag, allowed_values):
    values = []

    for field in record.get_fields(tag):
        if "Major genre" not in field.get_subfields("i"):
            continue

        for a in field.get_subfields("a"):
            if a in allowed_values:
                values.append(a)

    return list(dict.fromkeys(values))


def get_title(record):
    field = record.get("245")
    if not field:
        return ""

    a_vals = field.get_subfields("a")
    b_vals = field.get_subfields("b")

    a = " ".join(v.strip(" /:;") for v in a_vals if v and v.strip())
    b = " ".join(v.strip(" /:;") for v in b_vals if v and v.strip())

    if a and b:
        return f"{a} {b}"
    return a or b


def get_raw_subfields(field, code):
    subs = getattr(field, "subfields", [])

    if not subs:
        return []

    # nowszy pymarc: lista obiektów Subfield(code=..., value=...)
    if hasattr(subs[0], "code"):
        return [sf.value for sf in subs if sf.code == code]

    # starszy pymarc: lista naprzemienna [code, value, code, value, ...]
    raw = []
    for i in range(0, len(subs), 2):
        if subs[i] == code:
            raw.append(subs[i + 1])
    return raw


def analyze_subfield(field, code):
    """
    Zwraca słownik:
    - status: brak / puste / wartosc
    - display: brak / [puste] / połączone wartości
    - count_raw: liczba wystąpień podpola
    - count_nonempty: liczba niepustych wartości
    """
    raw_values = get_raw_subfields(field, code)

    if not raw_values:
        return {
            "status": "brak",
            "display": "brak",
            "count_raw": 0,
            "count_nonempty": 0
        }

    clean_values = []
    for v in raw_values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            clean_values.append(s)

    if clean_values:
        return {
            "status": "wartosc",
            "display": " | ".join(clean_values),
            "count_raw": len(raw_values),
            "count_nonempty": len(clean_values)
        }

    return {
        "status": "puste",
        "display": "[puste]",
        "count_raw": len(raw_values),
        "count_nonempty": 0
    }


rows = []

total_records = 0
records_with_655 = 0
records_with_x_present = 0
records_with_x_value = 0
records_with_major_genre = 0
records_exported = 0

for input_file in INPUT_FILES:
    file_name = os.path.basename(input_file)
    print(f"\n--- Przetwarzam: {file_name} ---")

    with open(input_file, "r", encoding="utf-8") as fh:
        reader = MARCMakerReader(fh)

        for record in reader:
            total_records += 1

            record_id = record["001"].value() if record["001"] else ""
            title = get_title(record)
            fields_655 = record.get_fields("655")

            if not fields_655:
                continue

            records_with_655 += 1

            all_655_a = []
            all_655_x = []
            all_655_pairs = []
            all_655_full = []

            has_x_present = False
            has_x_value = False

            # liczniki pól 655
            liczba_655_x_obecne = 0      # ile pól 655 ma x (także puste)
            liczba_655_z_x = 0           # ile pól 655 ma niepuste x

            # liczniki samych podpól x
            liczba_x_obecne_w_rekordzie = 0   # ile wszystkich podpól x jest w rekordzie
            liczba_x_w_rekordzie = 0          # ile niepustych podpól x jest w rekordzie

            for field in fields_655:
                a_info = analyze_subfield(field, "a")
                x_info = analyze_subfield(field, "x")

                all_655_a.append(a_info["display"])
                all_655_x.append(x_info["display"])
                all_655_pairs.append(f"{a_info['display']} -> {x_info['display']}")
                all_655_full.append(field.format_field())

                if x_info["status"] in ("puste", "wartosc"):
                    has_x_present = True
                    liczba_655_x_obecne += 1

                if x_info["status"] == "wartosc":
                    has_x_value = True
                    liczba_655_z_x += 1

                liczba_x_obecne_w_rekordzie += x_info["count_raw"]
                liczba_x_w_rekordzie += x_info["count_nonempty"]

            if has_x_present:
                records_with_x_present += 1

            if has_x_value:
                records_with_x_value += 1

            major_380 = get_major_genre_values(record, "380", VALID_380)
            major_381 = get_major_genre_values(record, "381", VALID_381)

            has_major_genre = bool(major_380 or major_381)
            if has_major_genre:
                records_with_major_genre += 1

            # filtr: tylko rekordy z realną wartością w 655$x
            if not has_x_value:
                continue

            if not has_major_genre:
                continue

            rows.append({
                "plik_zrodlowy": file_name,
                "001": record_id,
                "tytul": title,
                "liczba_655": len(fields_655),
                "liczba_655_x_obecne": liczba_655_x_obecne,
                "liczba_655_z_x": liczba_655_z_x,
                "liczba_x_obecne_w_rekordzie": liczba_x_obecne_w_rekordzie,
                "liczba_x_w_rekordzie": liczba_x_w_rekordzie,
                "655_a_wszystkie": " || ".join(all_655_a),
                "655_x_wszystkie": " || ".join(all_655_x),
                "655_ax_pary": " || ".join(all_655_pairs),
                "655_pola_pelne": " || ".join(all_655_full),
                "380_Major_genre": " | ".join(major_380),
                "381_Major_genre": " | ".join(major_381),
                "ma_380_Literature": int("Literature" in major_380),
                "ma_380_Secondary_literature": int("Secondary literature" in major_380),
                "ma_381_Fiction": int("Fiction" in major_381),
                "ma_381_Drama": int("Drama" in major_381),
                "ma_381_Other": int("Other" in major_381),
                "ma_381_Lyrical_poetry": int("Lyrical poetry" in major_381),
            })

            records_exported += 1

df = pd.DataFrame(rows)

column_order = [
    "plik_zrodlowy",
    "001",
    "tytul",
    "liczba_655",
    "liczba_655_x_obecne",
    "liczba_655_z_x",
    "liczba_x_obecne_w_rekordzie",
    "liczba_x_w_rekordzie",
    "655_a_wszystkie",
    "655_x_wszystkie",
    "655_ax_pary",
    "655_pola_pelne",
    "380_Major_genre",
    "381_Major_genre",
    "ma_380_Literature",
    "ma_380_Secondary_literature",
    "ma_381_Fiction",
    "ma_381_Drama",
    "ma_381_Other",
    "ma_381_Lyrical_poetry",
]

if not df.empty:
    df = df[column_order].sort_values(
        ["plik_zrodlowy", "liczba_x_w_rekordzie", "liczba_655_z_x", "001"],
        ascending=[True, False, False, True]
    )
else:
    df = pd.DataFrame(columns=column_order)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="audit")
    ws = writer.sheets["audit"]
    ws.freeze_panes = "A2"

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\n=== KONTROLA ===")
print(f"Wszystkie rekordy: {total_records}")
print(f"Rekordy z 655: {records_with_655}")
print(f"Rekordy z x obecnym (także pustym): {records_with_x_present}")
print(f"Rekordy z x mającym wartość: {records_with_x_value}")
print(f"Rekordy z Major genre w 380/381: {records_with_major_genre}")
print(f"Wyeksportowane rekordy: {records_exported}")

if not df.empty:
    print("\n=== PIERWSZE 10 REKORDÓW ===")
    print(df[[
        "plik_zrodlowy",
        "001",
        "tytul",
        "liczba_655",
        "liczba_655_x_obecne",
        "liczba_655_z_x",
        "liczba_x_obecne_w_rekordzie",
        "liczba_x_w_rekordzie",
        "655_a_wszystkie",
        "655_x_wszystkie",
        "380_Major_genre",
        "381_Major_genre"
    ]].head(10).to_string(index=False))
else:
    print("\nBrak rekordów spełniających warunki.")

print(f"\nZapisano plik XLSX: {OUTPUT_FILE}")
print(f"Zapisano plik CSV:  {OUTPUT_CSV}")

#%% przegląd bardziej ogólny
# -*- coding: utf-8 -*-
"""
Przegląd pól 655 niezależnie od obecności Major genre w 380/381.
Rekord trafia do tabeli, jeśli w którymkolwiek 655 ma niepuste a i/lub x.
"""

import os
import pandas as pd
from pymarc import MARCMakerReader


INPUT_FILES = [
    "C:/Users/darek/Downloads/pl_bn_articles_2025-03-04.mrk",
    "C:/Users/darek/Downloads/pl_bn_articles2_2025-03-04.mrk",
    "C:/Users/darek/Downloads/pl_bn_articles-2025-02-11_(poprawki_2025-03-30).mrk",
    "C:/Users/darek/Downloads/pl_bn_books2_2024-02-08_(poprawki_2025-03-30).mrk",
    "C:/Users/darek/Downloads/pl_bn_books-2025-02-11_(poprawki_2025-03-30).mrk",
]

OUTPUT_FILE = "elb_655_przeglad_allbn.xlsx"
OUTPUT_CSV = "elb_655_przeglad_allBN.csv"

VALID_380 = {"Secondary literature", "Literature"}
VALID_381 = {"Drama", "Fiction", "Other", "Lyrical poetry"}


def get_major_genre_values(record, tag, allowed_values):
    values = []

    for field in record.get_fields(tag):
        if "Major genre" not in field.get_subfields("i"):
            continue

        for a in field.get_subfields("a"):
            if a in allowed_values:
                values.append(a)

    return list(dict.fromkeys(values))


def get_title(record):
    field = record.get("245")
    if not field:
        return ""

    a_vals = field.get_subfields("a")
    b_vals = field.get_subfields("b")

    a = " ".join(v.strip(" /:;") for v in a_vals if v and v.strip())
    b = " ".join(v.strip(" /:;") for v in b_vals if v and v.strip())

    if a and b:
        return f"{a} {b}"
    return a or b


def get_raw_subfields(field, code):
    subs = getattr(field, "subfields", [])

    if not subs:
        return []

    # nowszy pymarc: lista obiektów Subfield(code=..., value=...)
    if hasattr(subs[0], "code"):
        return [sf.value for sf in subs if sf.code == code]

    # starszy pymarc: lista naprzemienna [code, value, code, value, ...]
    raw = []
    for i in range(0, len(subs), 2):
        if subs[i] == code:
            raw.append(subs[i + 1])
    return raw


def analyze_subfield(field, code):
    """
    Zwraca słownik:
    - status: brak / puste / wartosc
    - display: brak / [puste] / połączone wartości
    - count_raw: liczba wystąpień podpola
    - count_nonempty: liczba niepustych wartości
    """
    raw_values = get_raw_subfields(field, code)

    if not raw_values:
        return {
            "status": "brak",
            "display": "brak",
            "count_raw": 0,
            "count_nonempty": 0
        }

    clean_values = []
    for v in raw_values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            clean_values.append(s)

    if clean_values:
        return {
            "status": "wartosc",
            "display": " | ".join(clean_values),
            "count_raw": len(raw_values),
            "count_nonempty": len(clean_values)
        }

    return {
        "status": "puste",
        "display": "[puste]",
        "count_raw": len(raw_values),
        "count_nonempty": 0
    }


def escape_mrk_value(value):
    """Proste zabezpieczenie wartości do zapisu w stylu .mrk."""
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\r", " ")


def field_to_mrk(field):
    """
    Zwraca pole MARC w stylu .mrk, np.
    =655  \\4$aRecenzja
    =655  \\7$aPowieść polska$xstylistyka$2DBN
    """
    tag = field.tag

    # pola kontrolne
    if getattr(field, "is_control_field", lambda: False)():
        return f"={tag}  {escape_mrk_value(field.data)}"

    ind1 = field.indicator1 if field.indicator1 not in (None, " ") else "\\"
    ind2 = field.indicator2 if field.indicator2 not in (None, " ") else "\\"

    parts = [f"={tag}  {ind1}{ind2}"]

    subs = getattr(field, "subfields", [])

    if subs:
        # nowszy pymarc: lista obiektów Subfield
        if hasattr(subs[0], "code"):
            for sf in subs:
                parts.append(f"${sf.code}{escape_mrk_value(sf.value)}")
        # starszy pymarc: lista naprzemienna [code, value, code, value, ...]
        else:
            for i in range(0, len(subs), 2):
                code = subs[i]
                value = subs[i + 1]
                parts.append(f"${code}{escape_mrk_value(value)}")

    return "".join(parts)


def field_to_mrk_excel(field):
    """
    Apostrof na początku zabezpiecza przed interpretacją jako formuła Excela.
    """
    return "'" + field_to_mrk(field)


rows = []

total_records = 0
records_with_655 = 0
records_with_a_present = 0
records_with_a_value = 0
records_with_x_present = 0
records_with_x_value = 0
records_with_major_genre = 0
records_exported = 0

for input_file in INPUT_FILES:
    file_name = os.path.basename(input_file)
    print(f"\n--- Przetwarzam: {file_name} ---")

    with open(input_file, "r", encoding="utf-8") as fh:
        reader = MARCMakerReader(fh)

        for record in reader:
            total_records += 1

            record_id = record["001"].value() if record["001"] else ""
            title = get_title(record)
            fields_655 = record.get_fields("655")

            if not fields_655:
                continue

            records_with_655 += 1

            all_655_a = []
            all_655_x = []
            all_655_pairs = []
            all_655_full = []

            has_a_present = False
            has_a_value = False
            has_x_present = False
            has_x_value = False

            # liczniki pól 655
            liczba_655_a_obecne = 0
            liczba_655_z_a = 0
            liczba_655_x_obecne = 0
            liczba_655_z_x = 0

            # liczniki samych podpól
            liczba_a_obecne_w_rekordzie = 0
            liczba_a_w_rekordzie = 0
            liczba_x_obecne_w_rekordzie = 0
            liczba_x_w_rekordzie = 0

            for field in fields_655:
                a_info = analyze_subfield(field, "a")
                x_info = analyze_subfield(field, "x")

                all_655_a.append(a_info["display"])
                all_655_x.append(x_info["display"])
                all_655_pairs.append(f"{a_info['display']} -> {x_info['display']}")
                all_655_full.append(field_to_mrk_excel(field))

                # pola 655 z a
                if a_info["status"] in ("puste", "wartosc"):
                    has_a_present = True
                    liczba_655_a_obecne += 1

                if a_info["status"] == "wartosc":
                    has_a_value = True
                    liczba_655_z_a += 1

                # pola 655 z x
                if x_info["status"] in ("puste", "wartosc"):
                    has_x_present = True
                    liczba_655_x_obecne += 1

                if x_info["status"] == "wartosc":
                    has_x_value = True
                    liczba_655_z_x += 1

                # wszystkie podpola a/x
                liczba_a_obecne_w_rekordzie += a_info["count_raw"]
                liczba_a_w_rekordzie += a_info["count_nonempty"]
                liczba_x_obecne_w_rekordzie += x_info["count_raw"]
                liczba_x_w_rekordzie += x_info["count_nonempty"]

            if has_a_present:
                records_with_a_present += 1

            if has_a_value:
                records_with_a_value += 1

            if has_x_present:
                records_with_x_present += 1

            if has_x_value:
                records_with_x_value += 1

            major_380 = get_major_genre_values(record, "380", VALID_380)
            major_381 = get_major_genre_values(record, "381", VALID_381)

            has_major_genre = bool(major_380 or major_381)
            if has_major_genre:
                records_with_major_genre += 1

            # bierzemy rekord, jeśli ma w 655 niepuste a lub niepuste x
            if not (has_a_value or has_x_value):
                continue

            rows.append({
                "plik_zrodlowy": file_name,
                "001": record_id,
                "tytul": title,

                "liczba_655": len(fields_655),

                "liczba_655_a_obecne": liczba_655_a_obecne,
                "liczba_655_z_a": liczba_655_z_a,
                "liczba_a_obecne_w_rekordzie": liczba_a_obecne_w_rekordzie,
                "liczba_a_w_rekordzie": liczba_a_w_rekordzie,

                "liczba_655_x_obecne": liczba_655_x_obecne,
                "liczba_655_z_x": liczba_655_z_x,
                "liczba_x_obecne_w_rekordzie": liczba_x_obecne_w_rekordzie,
                "liczba_x_w_rekordzie": liczba_x_w_rekordzie,

                "655_a_wszystkie": " || ".join(all_655_a),
                "655_x_wszystkie": " || ".join(all_655_x),
                "655_ax_pary": " || ".join(all_655_pairs),
                "655_pola_pelne": "\n".join(all_655_full),

                "380_Major_genre": " | ".join(major_380) if major_380 else "brak major genre",
                "381_Major_genre": " | ".join(major_381) if major_381 else "brak major genre",

                "ma_380_Literature": int("Literature" in major_380),
                "ma_380_Secondary_literature": int("Secondary literature" in major_380),
                "ma_381_Fiction": int("Fiction" in major_381),
                "ma_381_Drama": int("Drama" in major_381),
                "ma_381_Other": int("Other" in major_381),
                "ma_381_Lyrical_poetry": int("Lyrical poetry" in major_381),
            })

            records_exported += 1

df = pd.DataFrame(rows)

column_order = [
    "plik_zrodlowy",
    "001",
    "tytul",

    "liczba_655",

    "liczba_655_a_obecne",
    "liczba_655_z_a",
    "liczba_a_obecne_w_rekordzie",
    "liczba_a_w_rekordzie",

    "liczba_655_x_obecne",
    "liczba_655_z_x",
    "liczba_x_obecne_w_rekordzie",
    "liczba_x_w_rekordzie",

    "655_a_wszystkie",
    "655_x_wszystkie",
    "655_ax_pary",
    "655_pola_pelne",

    "380_Major_genre",
    "381_Major_genre",

    "ma_380_Literature",
    "ma_380_Secondary_literature",
    "ma_381_Fiction",
    "ma_381_Drama",
    "ma_381_Other",
    "ma_381_Lyrical_poetry",
]

if not df.empty:
    df = df[column_order].sort_values(
        [
            "plik_zrodlowy",
            "liczba_a_w_rekordzie",
            "liczba_x_w_rekordzie",
            "liczba_655_z_a",
            "liczba_655_z_x",
            "001",
        ],
        ascending=[True, False, False, False, False, True]
    )
else:
    df = pd.DataFrame(columns=column_order)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="przeglad_655")
    ws = writer.sheets["przeglad_655"]
    ws.freeze_panes = "A2"

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\n=== KONTROLA ===")
print(f"Wszystkie rekordy: {total_records}")
print(f"Rekordy z 655: {records_with_655}")
print(f"Rekordy z a obecnym (także pustym): {records_with_a_present}")
print(f"Rekordy z a mającym wartość: {records_with_a_value}")
print(f"Rekordy z x obecnym (także pustym): {records_with_x_present}")
print(f"Rekordy z x mającym wartość: {records_with_x_value}")
print(f"Rekordy z Major genre w 380/381: {records_with_major_genre}")
print(f"Wyeksportowane rekordy: {records_exported}")

if not df.empty:
    print("\n=== PIERWSZE 10 REKORDÓW ===")
    print(df[[
        "plik_zrodlowy",
        "001",
        "tytul",
        "liczba_655",
        "liczba_655_a_obecne",
        "liczba_655_z_a",
        "liczba_a_w_rekordzie",
        "liczba_655_x_obecne",
        "liczba_655_z_x",
        "liczba_x_w_rekordzie",
        "380_Major_genre",
        "381_Major_genre"
    ]].head(10).to_string(index=False))
else:
    print("\nBrak rekordów spełniających warunki.")

print(f"\nZapisano plik XLSX: {OUTPUT_FILE}")
print(f"Zapisano plik CSV:  {OUTPUT_CSV}")


#%% Statystyki
import os
from collections import Counter
import pandas as pd
from pymarc import MARCMakerReader


INPUT_FILES = [
    "C:/Users/darek/Downloads/pl_bn_articles_2025-03-04.mrk",
    "C:/Users/darek/Downloads/pl_bn_articles2_2025-03-04.mrk",
    "C:/Users/darek/Downloads/pl_bn_articles-2025-02-11_(poprawki_2025-03-30).mrk",
    "C:/Users/darek/Downloads/pl_bn_books2_2024-02-08_(poprawki_2025-03-30).mrk",
    "C:/Users/darek/Downloads/pl_bn_books-2025-02-11_(poprawki_2025-03-30).mrk",
]

OUTPUT_FILE = "elb_655_slownik_poziomo.xlsx"
OUTPUT_CSV_LONG = "elb_655_slownik_long.csv"
OUTPUT_CSV_WIDE = "elb_655_slownik_poziomo.csv"


def clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def get_subfields_as_pairs(field):
    """
    Zwraca listę par (code, value) dla pola MARC, niezależnie od wersji pymarc.
    """
    subs = getattr(field, "subfields", [])
    if not subs:
        return []

    # nowszy pymarc: lista obiektów Subfield
    if hasattr(subs[0], "code"):
        return [(sf.code, sf.value) for sf in subs]

    # starszy pymarc: lista naprzemienna [code, value, code, value, ...]
    pairs = []
    for i in range(0, len(subs), 2):
        pairs.append((subs[i], subs[i + 1]))
    return pairs


# globalne liczniki
value_counter = Counter()      # (pole_podpole, wartosc) -> liczba wszystkich wystąpień
record_counter = Counter()     # (pole_podpole, wartosc) -> liczba rekordów

total_records = 0
records_with_655 = 0

for input_file in INPUT_FILES:
    file_name = os.path.basename(input_file)
    print(f"\n--- Przetwarzam: {file_name} ---")

    with open(input_file, "r", encoding="utf-8") as fh:
        reader = MARCMakerReader(fh)

        for record in reader:
            total_records += 1
            fields_655 = record.get_fields("655")

            if not fields_655:
                continue

            records_with_655 += 1

            seen_pairs_in_record = set()

            for field in fields_655:
                for code, raw_value in get_subfields_as_pairs(field):
                    value = clean_value(raw_value)

                    # pomijamy puste wartości
                    if not value:
                        continue

                    pole_podpole = f"655${code}"

                    value_counter[(pole_podpole, value)] += 1
                    seen_pairs_in_record.add((pole_podpole, value))

            for pair in seen_pairs_in_record:
                record_counter[pair] += 1


# ===== słownik długi =====
dict_rows = []
for (pole_podpole, value), count in value_counter.items():
    dict_rows.append({
        "pole_podpole": pole_podpole,
        "wartosc": value,
        "liczba_wystapien": count,
        "liczba_rekordow": record_counter[(pole_podpole, value)],
    })

df_long = pd.DataFrame(dict_rows)

if not df_long.empty:
    df_long = df_long.sort_values(
        ["pole_podpole", "liczba_rekordow", "liczba_wystapien", "wartosc"],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)


# ===== słownik poziomy =====
blocks = []

if not df_long.empty:
    podpola = sorted(df_long["pole_podpole"].unique())

    for pole_podpole in podpola:
        temp = df_long[df_long["pole_podpole"] == pole_podpole].copy()

        temp = temp[["wartosc", "liczba_wystapien", "liczba_rekordow"]].reset_index(drop=True)

        # nazwy kolumn dla bloku
        short_name = pole_podpole.replace("$", "")
        temp.columns = [
            short_name,
            f"{short_name}_liczba_wystapien",
            f"{short_name}_liczba_rekordow",
        ]

        blocks.append(temp)

    df_wide = pd.concat(blocks, axis=1)
else:
    df_wide = pd.DataFrame()


# ===== zapis =====
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_wide.to_excel(writer, index=False, sheet_name="655_slownik_poziomo")
    writer.sheets["655_slownik_poziomo"].freeze_panes = "A2"

    df_long.to_excel(writer, index=False, sheet_name="655_slownik_long")
    writer.sheets["655_slownik_long"].freeze_panes = "A2"

df_long.to_csv(OUTPUT_CSV_LONG, index=False, encoding="utf-8-sig")
df_wide.to_csv(OUTPUT_CSV_WIDE, index=False, encoding="utf-8-sig")

print("\n=== KONTROLA ===")
print(f"Wszystkie rekordy: {total_records}")
print(f"Rekordy z 655: {records_with_655}")
print(f"Liczba unikalnych par (pole_podpole, wartosc): {len(df_long)}")

if not df_long.empty:
    print("\n=== ZNALEZIONE PODPOLA ===")
    print(", ".join(sorted(df_long['pole_podpole'].unique())))

print(f"\nZapisano plik XLSX: {OUTPUT_FILE}")
print(f"Zapisano CSV long: {OUTPUT_CSV_LONG}")
print(f"Zapisano CSV wide: {OUTPUT_CSV_WIDE}")

#%% statystyki dla 650
import os
from collections import Counter
import pandas as pd
from pymarc import MARCMakerReader


INPUT_FILES = [
    "C:/Users/darek/Downloads/pl_bn_articles_2025-03-04.mrk",
    "C:/Users/darek/Downloads/pl_bn_articles2_2025-03-04.mrk",
    "C:/Users/darek/Downloads/pl_bn_articles-2025-02-11_(poprawki_2025-03-30).mrk",
    "C:/Users/darek/Downloads/pl_bn_books2_2024-02-08_(poprawki_2025-03-30).mrk",
    "C:/Users/darek/Downloads/pl_bn_books-2025-02-11_(poprawki_2025-03-30).mrk",
]

OUTPUT_FILE = "elb_650_slownik_poziomo.xlsx"
OUTPUT_CSV_LONG = "elb_650_slownik_long.csv"
OUTPUT_CSV_WIDE = "elb_650_slownik_poziomo.csv"


def clean_value(value):
    if value is None:
        return ""
    return str(value).strip()


def get_subfields_as_pairs(field):
    """
    Zwraca listę par (code, value) dla pola MARC, niezależnie od wersji pymarc.
    """
    subs = getattr(field, "subfields", [])
    if not subs:
        return []

    # nowszy pymarc: lista obiektów Subfield
    if hasattr(subs[0], "code"):
        return [(sf.code, sf.value) for sf in subs]

    # starszy pymarc: lista naprzemienna [code, value, code, value, ...]
    pairs = []
    for i in range(0, len(subs), 2):
        pairs.append((subs[i], subs[i + 1]))
    return pairs


# globalne liczniki
value_counter = Counter()      # (pole_podpole, wartosc) -> liczba wszystkich wystąpień
record_counter = Counter()     # (pole_podpole, wartosc) -> liczba rekordów

total_records = 0
records_with_650 = 0

for input_file in INPUT_FILES:
    file_name = os.path.basename(input_file)
    print(f"\n--- Przetwarzam: {file_name} ---")

    with open(input_file, "r", encoding="utf-8") as fh:
        reader = MARCMakerReader(fh)

        for record in reader:
            total_records += 1
            fields_650 = record.get_fields("650")

            if not fields_650:
                continue

            records_with_650 += 1

            seen_pairs_in_record = set()

            for field in fields_650:
                for code, raw_value in get_subfields_as_pairs(field):
                    code = str(code).strip() if code is not None else ""
                    value = clean_value(raw_value)

                    # pomijamy puste wartości i puste/dziwne kody
                    if not code or not value:
                        continue

                    pole_podpole = f"650${code}"

                    value_counter[(pole_podpole, value)] += 1
                    seen_pairs_in_record.add((pole_podpole, value))

            for pair in seen_pairs_in_record:
                record_counter[pair] += 1


# ===== słownik długi =====
dict_rows = []
for (pole_podpole, value), count in value_counter.items():
    dict_rows.append({
        "pole_podpole": pole_podpole,
        "wartosc": value,
        "liczba_wystapien": count,
        "liczba_rekordow": record_counter[(pole_podpole, value)],
    })

df_long = pd.DataFrame(dict_rows)

if not df_long.empty:
    df_long = df_long.sort_values(
        ["pole_podpole", "liczba_rekordow", "liczba_wystapien", "wartosc"],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)


# ===== słownik poziomy =====
blocks = []

if not df_long.empty:
    podpola = sorted(df_long["pole_podpole"].unique())

    for pole_podpole in podpola:
        temp = df_long[df_long["pole_podpole"] == pole_podpole].copy()
        temp = temp[["wartosc", "liczba_wystapien", "liczba_rekordow"]].reset_index(drop=True)

        # zostawiamy czytelne nagłówki z $
        temp.columns = [
            pole_podpole,
            f"{pole_podpole}_liczba_wystapien",
            f"{pole_podpole}_liczba_rekordow",
        ]

        blocks.append(temp)

    df_wide = pd.concat(blocks, axis=1)
else:
    df_wide = pd.DataFrame()


# ===== zapis =====
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_wide.to_excel(writer, index=False, sheet_name="650_slownik_poziomo")
    writer.sheets["650_slownik_poziomo"].freeze_panes = "A2"

    df_long.to_excel(writer, index=False, sheet_name="650_slownik_long")
    writer.sheets["650_slownik_long"].freeze_panes = "A2"

df_long.to_csv(OUTPUT_CSV_LONG, index=False, encoding="utf-8-sig")
df_wide.to_csv(OUTPUT_CSV_WIDE, index=False, encoding="utf-8-sig")

print("\n=== KONTROLA ===")
print(f"Wszystkie rekordy: {total_records}")
print(f"Rekordy z 650: {records_with_650}")
print(f"Liczba unikalnych par (pole_podpole, wartosc): {len(df_long)}")

if not df_long.empty:
    print("\n=== ZNALEZIONE PODPOLA 650 ===")
    print(", ".join(sorted(df_long['pole_podpole'].unique())))

print(f"\nZapisano plik XLSX: {OUTPUT_FILE}")
print(f"Zapisano CSV long: {OUTPUT_CSV_LONG}")
print(f"Zapisano CSV wide: {OUTPUT_CSV_WIDE}")
#%% 650 przegląd
import os
import pandas as pd
from pymarc import MARCMakerReader


INPUT_FILES = [
    "C:/Users/darek/Downloads/pl_bn_articles_2025-03-04.mrk",
    "C:/Users/darek/Downloads/pl_bn_articles2_2025-03-04.mrk",
    "C:/Users/darek/Downloads/pl_bn_articles-2025-02-11_(poprawki_2025-03-30).mrk",
    "C:/Users/darek/Downloads/pl_bn_books2_2024-02-08_(poprawki_2025-03-30).mrk",
    "C:/Users/darek/Downloads/pl_bn_books-2025-02-11_(poprawki_2025-03-30).mrk",
]

OUTPUT_FILE = "elb_650_przeglad_allbn.xlsx"
OUTPUT_CSV = "elb_650_przeglad_allBN.csv"

VALID_380 = {"Secondary literature", "Literature"}
VALID_381 = {"Drama", "Fiction", "Other", "Lyrical poetry"}


def get_major_genre_values(record, tag, allowed_values):
    values = []

    for field in record.get_fields(tag):
        if "Major genre" not in field.get_subfields("i"):
            continue

        for a in field.get_subfields("a"):
            if a in allowed_values:
                values.append(a)

    return list(dict.fromkeys(values))


def get_title(record):
    field = record.get("245")
    if not field:
        return ""

    a_vals = field.get_subfields("a")
    b_vals = field.get_subfields("b")

    a = " ".join(v.strip(" /:;") for v in a_vals if v and v.strip())
    b = " ".join(v.strip(" /:;") for v in b_vals if v and v.strip())

    if a and b:
        return f"{a} {b}"
    return a or b


def get_raw_subfields(field, code):
    subs = getattr(field, "subfields", [])

    if not subs:
        return []

    # nowszy pymarc: lista obiektów Subfield(code=..., value=...)
    if hasattr(subs[0], "code"):
        return [sf.value for sf in subs if sf.code == code]

    # starszy pymarc: lista naprzemienna [code, value, code, value, ...]
    raw = []
    for i in range(0, len(subs), 2):
        if subs[i] == code:
            raw.append(subs[i + 1])
    return raw


def analyze_subfield(field, code):
    """
    Zwraca słownik:
    - status: brak / puste / wartosc
    - display: brak / [puste] / połączone wartości
    - count_raw: liczba wystąpień podpola
    - count_nonempty: liczba niepustych wartości
    """
    raw_values = get_raw_subfields(field, code)

    if not raw_values:
        return {
            "status": "brak",
            "display": "brak",
            "count_raw": 0,
            "count_nonempty": 0
        }

    clean_values = []
    for v in raw_values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            clean_values.append(s)

    if clean_values:
        return {
            "status": "wartosc",
            "display": " | ".join(clean_values),
            "count_raw": len(raw_values),
            "count_nonempty": len(clean_values)
        }

    return {
        "status": "puste",
        "display": "[puste]",
        "count_raw": len(raw_values),
        "count_nonempty": 0
    }


def escape_mrk_value(value):
    """Proste zabezpieczenie wartości do zapisu w stylu .mrk."""
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\r", " ")


def field_to_mrk(field):
    """
    Zwraca pole MARC w stylu .mrk, np.
    =650  \\7$aLiteratura polska$xhistoria$2DBN
    """
    tag = field.tag

    # pola kontrolne
    if getattr(field, "is_control_field", lambda: False)():
        return f"={tag}  {escape_mrk_value(field.data)}"

    ind1 = field.indicator1 if field.indicator1 not in (None, " ") else "\\"
    ind2 = field.indicator2 if field.indicator2 not in (None, " ") else "\\"

    parts = [f"={tag}  {ind1}{ind2}"]

    subs = getattr(field, "subfields", [])

    if subs:
        # nowszy pymarc: lista obiektów Subfield
        if hasattr(subs[0], "code"):
            for sf in subs:
                parts.append(f"${sf.code}{escape_mrk_value(sf.value)}")
        # starszy pymarc: lista naprzemienna [code, value, code, value, ...]
        else:
            for i in range(0, len(subs), 2):
                code = subs[i]
                value = subs[i + 1]
                parts.append(f"${code}{escape_mrk_value(value)}")

    return "".join(parts)


def field_to_mrk_excel(field):
    """
    Apostrof na początku zabezpiecza przed interpretacją jako formuła Excela.
    """
    return "'" + field_to_mrk(field)


rows = []

total_records = 0
records_with_650 = 0
records_with_a_present = 0
records_with_a_value = 0
records_with_x_present = 0
records_with_x_value = 0
records_with_major_genre = 0
records_exported = 0

for input_file in INPUT_FILES:
    file_name = os.path.basename(input_file)
    print(f"\n--- Przetwarzam: {file_name} ---")

    with open(input_file, "r", encoding="utf-8") as fh:
        reader = MARCMakerReader(fh)

        for record in reader:
            total_records += 1

            record_id = record["001"].value() if record["001"] else ""
            title = get_title(record)
            fields_650 = record.get_fields("650")

            if not fields_650:
                continue

            records_with_650 += 1

            all_650_a = []
            all_650_x = []
            all_650_pairs = []
            all_650_full = []

            has_a_present = False
            has_a_value = False
            has_x_present = False
            has_x_value = False

            # liczniki pól 650
            liczba_650_a_obecne = 0
            liczba_650_z_a = 0
            liczba_650_x_obecne = 0
            liczba_650_z_x = 0

            # liczniki samych podpól
            liczba_a_obecne_w_rekordzie = 0
            liczba_a_w_rekordzie = 0
            liczba_x_obecne_w_rekordzie = 0
            liczba_x_w_rekordzie = 0

            for field in fields_650:
                a_info = analyze_subfield(field, "a")
                x_info = analyze_subfield(field, "x")

                all_650_a.append(a_info["display"])
                all_650_x.append(x_info["display"])
                all_650_pairs.append(f"{a_info['display']} -> {x_info['display']}")
                all_650_full.append(field_to_mrk_excel(field))

                # pola 650 z a
                if a_info["status"] in ("puste", "wartosc"):
                    has_a_present = True
                    liczba_650_a_obecne += 1

                if a_info["status"] == "wartosc":
                    has_a_value = True
                    liczba_650_z_a += 1

                # pola 650 z x
                if x_info["status"] in ("puste", "wartosc"):
                    has_x_present = True
                    liczba_650_x_obecne += 1

                if x_info["status"] == "wartosc":
                    has_x_value = True
                    liczba_650_z_x += 1

                # wszystkie podpola a/x
                liczba_a_obecne_w_rekordzie += a_info["count_raw"]
                liczba_a_w_rekordzie += a_info["count_nonempty"]
                liczba_x_obecne_w_rekordzie += x_info["count_raw"]
                liczba_x_w_rekordzie += x_info["count_nonempty"]

            if has_a_present:
                records_with_a_present += 1

            if has_a_value:
                records_with_a_value += 1

            if has_x_present:
                records_with_x_present += 1

            if has_x_value:
                records_with_x_value += 1

            major_380 = get_major_genre_values(record, "380", VALID_380)
            major_381 = get_major_genre_values(record, "381", VALID_381)

            has_major_genre = bool(major_380 or major_381)
            if has_major_genre:
                records_with_major_genre += 1

            # bierzemy rekord, jeśli ma w 650 niepuste a lub niepuste x
            if not (has_a_value or has_x_value):
                continue

            rows.append({
                "plik_zrodlowy": file_name,
                "001": record_id,
                "tytul": title,

                "liczba_650": len(fields_650),

                "liczba_650_a_obecne": liczba_650_a_obecne,
                "liczba_650_z_a": liczba_650_z_a,
                "liczba_a_obecne_w_rekordzie": liczba_a_obecne_w_rekordzie,
                "liczba_a_w_rekordzie": liczba_a_w_rekordzie,

                "liczba_650_x_obecne": liczba_650_x_obecne,
                "liczba_650_z_x": liczba_650_z_x,
                "liczba_x_obecne_w_rekordzie": liczba_x_obecne_w_rekordzie,
                "liczba_x_w_rekordzie": liczba_x_w_rekordzie,

                "650_a_wszystkie": " || ".join(all_650_a),
                "650_x_wszystkie": " || ".join(all_650_x),
                "650_ax_pary": " || ".join(all_650_pairs),
                "650_pola_pelne": "\n".join(all_650_full),

                "380_Major_genre": " | ".join(major_380) if major_380 else "brak major genre",
                "381_Major_genre": " | ".join(major_381) if major_381 else "brak major genre",

                "ma_380_Literature": int("Literature" in major_380),
                "ma_380_Secondary_literature": int("Secondary literature" in major_380),
                "ma_381_Fiction": int("Fiction" in major_381),
                "ma_381_Drama": int("Drama" in major_381),
                "ma_381_Other": int("Other" in major_381),
                "ma_381_Lyrical_poetry": int("Lyrical poetry" in major_381),
            })

            records_exported += 1

df = pd.DataFrame(rows)

column_order = [
    "plik_zrodlowy",
    "001",
    "tytul",

    "liczba_650",

    "liczba_650_a_obecne",
    "liczba_650_z_a",
    "liczba_a_obecne_w_rekordzie",
    "liczba_a_w_rekordzie",

    "liczba_650_x_obecne",
    "liczba_650_z_x",
    "liczba_x_obecne_w_rekordzie",
    "liczba_x_w_rekordzie",

    "650_a_wszystkie",
    "650_x_wszystkie",
    "650_ax_pary",
    "650_pola_pelne",

    "380_Major_genre",
    "381_Major_genre",

    "ma_380_Literature",
    "ma_380_Secondary_literature",
    "ma_381_Fiction",
    "ma_381_Drama",
    "ma_381_Other",
    "ma_381_Lyrical_poetry",
]

if not df.empty:
    df = df[column_order].sort_values(
        [
            "plik_zrodlowy",
            "liczba_a_w_rekordzie",
            "liczba_x_w_rekordzie",
            "liczba_650_z_a",
            "liczba_650_z_x",
            "001",
        ],
        ascending=[True, False, False, False, False, True]
    )
else:
    df = pd.DataFrame(columns=column_order)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="przeglad_650")
    ws = writer.sheets["przeglad_650"]
    ws.freeze_panes = "A2"

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\n=== KONTROLA ===")
print(f"Wszystkie rekordy: {total_records}")
print(f"Rekordy z 650: {records_with_650}")
print(f"Rekordy z a obecnym (także pustym): {records_with_a_present}")
print(f"Rekordy z a mającym wartość: {records_with_a_value}")
print(f"Rekordy z x obecnym (także pustym): {records_with_x_present}")
print(f"Rekordy z x mającym wartość: {records_with_x_value}")
print(f"Rekordy z Major genre w 380/381: {records_with_major_genre}")
print(f"Wyeksportowane rekordy: {records_exported}")

if not df.empty:
    print("\n=== PIERWSZE 10 REKORDÓW ===")
    print(df[[
        "plik_zrodlowy",
        "001",
        "tytul",
        "liczba_650",
        "liczba_650_a_obecne",
        "liczba_650_z_a",
        "liczba_a_w_rekordzie",
        "liczba_650_x_obecne",
        "liczba_650_z_x",
        "liczba_x_w_rekordzie",
        "380_Major_genre",
        "381_Major_genre"
    ]].head(10).to_string(index=False))
else:
    print("\nBrak rekordów spełniających warunki.")

print(f"\nZapisano plik XLSX: {OUTPUT_FILE}")
print(f"Zapisano plik CSV:  {OUTPUT_CSV}")