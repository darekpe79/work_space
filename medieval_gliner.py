# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:00:29 2025

@author: darek
"""

# from gliner import GLiNER
# from pathlib import Path

# # 1. Wczytaj tekst z pliku
# txt_path = r"C:/pdf_llm_do_roboty/single_pages_simple_text-20250804T064354Z-1-001/single_pages_simple_text/MonumentaPeruana_1 - 0776.txt"
# with open(txt_path, encoding="utf8") as f:
#     text = f.read()

# # 2. Załaduj model GLiNER z backendem Torch (nie ONNX!)
# model = GLiNER.from_pretrained(
#     "medieval-data/gliner_multi-v2.1-medieval-latin",
#     inference_backend="torch"
# )

# # 3. Zdefiniuj etykiety NER
# labels = ["PERSON", "LOC"]

# # 4. Wykonaj ekstrakcję encji
# entities = model.predict_entities(text, labels)

# # 5. Wyświetl wynik
# for entity in entities:
#     print(entity["text"], "=>", entity["label"])
from gliner import GLiNER
from pathlib import Path
import pandas as pd

# 1. Wczytaj tekst z pliku
txt_path = r"C:/pdf_llm_do_roboty/single_pages_simple_text-20250804T064354Z-1-001/single_pages_simple_text/MonumentaPeruana_1 - 0776.txt"
with open(txt_path, encoding="utf8") as f:
    text = f.read()

# 2. Załaduj model GLiNER z backendem Torch
model = GLiNER.from_pretrained(
    "medieval-data/gliner_multi-v2.1-medieval-latin",
    inference_backend="torch"
)

# 3. Etykiety NER
labels = ["PERSON", "LOC"]

# 4. Predykcja encji
entities = model.predict_entities(text, labels)

# 5. Przekształć do DataFrame + dodaj nazwę pliku
data = []
for e in entities:
    data.append({
        "text": e["text"],
        "label": e["label"],
        "start": e["start"],
        "end": e["end"],
        "score": round(e.get("score", 0), 4),  # dodaj jeśli jest dostępne
        "file": Path(txt_path).name
    })

df = pd.DataFrame(data)

# 6. Eksport do Excela
output_excel = Path(txt_path).with_name("entities_gliner_output.xlsx")
df.to_excel(output_excel, index=False)
print(f"Zapisano do: {output_excel}")


    