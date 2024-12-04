# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:55:10 2024

@author: dariu
"""

import os
import shutil

# Ścieżki do katalogów źródłowych i docelowych
base_dir = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001'
json_dir = os.path.join(base_dir, 'jsony')
destination_dir = 'D:/Nowa_praca/dane_model_jezykowy/kopia_dla'

# Tworzenie katalogu docelowego, jeśli nie istnieje
os.makedirs(destination_dir, exist_ok=True)

# Pobranie list plików JSON i Excel
json_files = {os.path.splitext(f)[0]: os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')}
excel_files = {os.path.splitext(f)[0]: os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.xlsx')}

# Znalezienie wspólnych nazw plików
common_files = set(json_files.keys()).intersection(excel_files.keys())

# Kopiowanie plików
for file_name in common_files:
    try:
        # Kopiowanie JSON
        json_src = json_files[file_name]
        json_dst = os.path.join(destination_dir, os.path.basename(json_src))
        shutil.copy(json_src, json_dst)
        print(f"Skopiowano JSON: {json_src} -> {json_dst}")
        
        # Kopiowanie Excel
        excel_src = excel_files[file_name]
        excel_dst = os.path.join(destination_dir, os.path.basename(excel_src))
        shutil.copy(excel_src, excel_dst)
        print(f"Skopiowano Excel: {excel_src} -> {excel_dst}")
    except Exception as e:
        print(f"Błąd przy kopiowaniu pliku {file_name}: {e}")

print("Kopiowanie zakończone.")
