# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 21:42:44 2025

@author: darek
"""

import json

# ścieżka do pliku
path = "D:/Nowa_praca/dane_model_jezykowy/dokumenty po anotacji-20240930T120225Z-001/dokumenty po anotacji/event51.json"

# otwieramy plik i wczytujemy JSON-a jako strukturę Pythonową (dict/list)
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# zobaczmy, jakie ma klucze
print(data.keys())
text1=data.get("annotations", [])[0][1]
text2=text1.get("entities", [])

annotations = data.get("annotations", [])
if annotations:
    text = annotations[0][0]
else:
    text = ""
    
    
for start, end, etype in text2:
    fragment = text[start:end]
    print(etype, "→", fragment)


import os, json, glob, re
import pandas as pd

# 1️⃣ Wskaż katalog z plikami JSON
INPUT_GLOB = r"D:/Nowa_praca/dane_model_jezykowy/dokumenty po anotacji-20240930T120225Z-001/dokumenty po anotacji/*.json"

# 2️⃣ Stały prompt do ewentualnego użycia w trenowaniu
instruction = (
    "Wypisz encje nazwane z tekstu w formacie 'RODZAJ-tytuł'. "
    "Dopuszczalne RODZAJE: BOOK, PLAY, EVENT."
)

# 3️⃣ Inicjalizacja pustej listy na wiersze tabeli
rows = []

# 4️⃣ Iteracja po wszystkich plikach JSON
for path in glob.glob(INPUT_GLOB):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    if not annotations:
        continue  # pomiń puste

    text = annotations[0][0]
    ent_block = annotations[0][1] if isinstance(annotations[0][1], dict) else {}
    entities = ent_block.get("entities", [])

    for start, end, etype in entities:
        # wycinamy fragment tekstu
        if 0 <= start <= end <= len(text):
            span = text[start:end]
        else:
            span = ""

        # czyścimy tytuł
        span = re.sub(r"\[/tytuł\]", "", span)
        span = span.strip("„”\"'[](){} \n\t\r")
        span = re.sub(r"\s+", " ", span)

        formatted = f"{etype}-{span}" if span else ""

        rows.append({
            "file": os.path.basename(path),
            "etype": etype,
            "title": span,
            "formatted": formatted,
            "text": text,
            "instruction": instruction
        })

# 5️⃣ Tworzymy DataFrame
df = pd.DataFrame(rows)

# 6️⃣ Zapis do CSV
OUT_PATH = r"D:/Nowa_praca/dane_model_jezykowy/wszystkie_encje.csv"
df.to_csv(OUT_PATH, index=False, encoding="utf-8")

print(f"✅ Zapisano {len(df)} wierszy do pliku:\n{OUT_PATH}")
df.head()

