# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:14:00 2024

@author: dariu
"""

from pymarc import JSONWriter, MARCReader
from io import StringIO
import pandas as pd
import json
# Ścieżka do pliku MARC21
input_file_path = "D:/Nowa_praca/08.02.2024_marki/es_articles__08-02-2024.mrc"
output_file_path = "es_ksiazki.json"
output_excel_path = "output_flat.xlsx"
# Odczyt rekordów MARC i zapis do JSON
with open(input_file_path, 'rb') as marc_file, open(output_file_path, 'wt') as json_file:
    reader = MARCReader(marc_file)
    writer = JSONWriter(json_file)
    for record in reader:
        writer.write(record)
    writer.close()  # Ważne, aby zamknąć writer!



# Wczytanie JSON do DataFrame
with open(output_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
def process_record(record):
    """
    Przetwarza pojedynczy rekord MARC, spłaszczając pola w `fields`.
    """
    flat_record = {"leader": record.get("leader", "")}
    for field in record.get("fields", []):
        for key, value in field.items():
            flat_record[key] = value
    return flat_record

# Przekształcenie wszystkich rekordów
flat_data = [process_record(record) for record in data]
df = pd.DataFrame(flat_data)
df.to_excel(output_excel_path, index=False, engine='openpyxl')
