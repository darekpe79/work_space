# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:35:52 2023

@author: dariu
"""

import re

# Funkcja do zamiany URL
def replace_urls(data):
    pattern = r'https://data\.bn\.org\.pl/api/institutions/authorities\.json\?id=(\d+)'
    replacement = lambda m: f"https://dbn.bn.org.pl/descriptor-details/a{'0'*(13-len(m.group(1)))}{m.group(1)}"
    return re.sub(pattern, replacement, data)

# Ścieżka do pliku .ttl
file_path = 'D:/Nowa_praca/słowniki_wszystkie_final/skos_genre (2).turtle'

# Odczytanie zawartości pliku
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

# Zastosowanie funkcji replace_urls
modified_data = replace_urls(data)

# Wyświetlenie zmodyfikowanych danych
print(modified_data)

# Opcjonalnie: Zapisanie zmodyfikowanych danych do nowego pliku
with open('D:/Nowa_praca/słowniki_wszystkie_final/skos_genre.ttl', 'w', encoding='utf-8') as modified_file:
    modified_file.write(modified_data)
    
    
def replace(data):
    pattern = r'(\(?\d{3}\)?[-\s]?)?\d{3}[-\s]?\d{4}'
    replacement = lambda m: "Python (wspaniały język programowania)"
    return re.sub(pattern, replacement, data, flags=re.IGNORECASE)


replace('python to słabo')

def normalize_dates(dates):
    pattern = r'(\d{2})\D*(\d{2})\D*(\d{4})'
    replacement = lambda m: f"[{m.group(1)}-{m.group(2)}-{m.group(3)}]"
    return [re.sub(pattern, replacement, date) for date in dates]

normalize_dates(['Spotkanie 12-05-2023'])