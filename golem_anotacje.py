# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:31:11 2024

@author: dariu
"""

import os
import json

# Ścieżka do katalogu z plikami JSON
katalog = 'D:/KDL/KDL_final/inforex_export_953/documents/'

# Zbieranie listy plików JSON
pliki_json = [plik for plik in os.listdir(katalog) if plik.endswith('.json')]

# Sprawdzanie, czy istnieje przynajmniej jeden plik JSON
if pliki_json:
    # Ścieżka do pierwszego pliku JSON
    sciezka_do_pierwszego_pliku = os.path.join(katalog, pliki_json[0])
    
    # Otwieranie i wczytywanie pierwszego pliku JSON
    with open(sciezka_do_pierwszego_pliku, 'r', encoding='utf-8') as plik:
        dane = json.load(plik)
    
    # Możesz teraz pracować z danymi z pierwszego pliku JSON
    print(f"Dane z {pliki_json[0]}: {dane}")
else:
    print("Nie znaleziono plików JSON w podanym katalogu.")


# Teraz możesz pracować z danymi JSON jako ze słownikiem w Pythonie
data=dane['annotations']['name']


import os
import json
from collections import Counter

# Ścieżka do katalogu z plikami JSON
katalog = 'D:/KDL/KDL_agreement/inforex_export_954/documents/'

# Zbieranie listy plików JSON
pliki_json = [plik for plik in os.listdir(katalog) if plik.endswith('.json')]

# Inicjalizacja licznika dla wartości "name"
licznik_name = Counter()

# Przejście przez każdy plik JSON w katalogu
for nazwa_pliku in pliki_json:
    sciezka_do_pliku = os.path.join(katalog, nazwa_pliku)
    with open(sciezka_do_pliku, 'r', encoding='utf-8') as plik:
        dane = json.load(plik)
        
        # Zakładamy, że struktura danych jest taka sama jak w przykładzie podanym przez użytkownika
        # i że dane są zagnieżdżone w kluczu 'annotations'
        # Jeśli struktura różni się między plikami, ten kod może wymagać dostosowania
        for annotation in dane['annotations']:
            name = annotation['type']
            licznik_name[name] += 1

# Wyświetlenie wyników
for name, liczba in licznik_name.items():
    print(f"{name}: {liczba}")

