# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:59:28 2025

@author: darek
"""

import os
import glob
from pymarc import MARCReader, JSONWriter

# Ścieżki do katalogów (dostosuj do własnych potrzeb)
INPUT_DIR = 'D:/Nowa_praca/marki_po_updatach 2025,2024/'
OUTPUT_DIR = 'D:/Nowa_praca/marki_po_updatach 2025,2024/'

# Upewnij się, że katalog wyjściowy istnieje; jeśli nie - utwórz go
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Przechodzimy przez wszystkie pliki .mrc w katalogu INPUT_DIR
for file_path in glob.glob(os.path.join(INPUT_DIR, '*.mrc')):
    # Wyciągnięcie nazwy pliku bez ścieżki i rozszerzenia
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Przygotowanie ścieżki dla pliku wynikowego .json
    output_file_path = os.path.join(OUTPUT_DIR, base_name + '.json')

    # Otwieramy plik MARC w trybie binarnym oraz plik docelowy w trybie zapisu tekstowego
    with open(file_path, 'rb') as mrc_file, open(output_file_path, 'wt', encoding='utf-8') as json_file:
        reader = MARCReader(mrc_file)
        writer = JSONWriter(json_file)
        
        for record in reader:
            writer.write(record)
        
        # Zakończenie pracy writer – ważne, aby zamknąć, bo inaczej plik nie zostanie sfinalizowany poprawnie
        writer.close()

    print(f"Przekonwertowano: {file_path} -> {output_file_path}")
