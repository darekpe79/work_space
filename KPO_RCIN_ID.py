# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:13:49 2025

@author: darek
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

# Wczytujemy tylko arkusz "Czasopisma" z Excela
plik_wejsciowy = "D:/Nowa_praca/KPO/KDL_wszystkie zasoby + metadane.xlsx"
plik_wyjsciowy = "D:/Nowa_praca/KPO/KDL_wszystkie_zasoby_z_rcin.xlsx"

df_czasopisma = pd.read_excel(plik_wejsciowy, sheet_name="czasopisma")

# Filtrujemy tylko wiersze, gdzie source == "Teksty Drugie"
df_filtered = df_czasopisma[df_czasopisma["source"] == "Teksty Drugie"].copy()

# Pobieramy unikalne tytuły bez NaN
tytuly = df_filtered["title"].dropna().unique().tolist()
# Funkcja do pobierania ID z RCIN
def get_rcin_id(title):
    base_url = "https://rcin.org.pl/dlibra/results"
    
    # Dodajemy cudzysłowy do tytułu, jeśli ich nie ma
    quoted_title = f'"{title}"'  # Python automatycznie obsłuży polskie znaki i cudzysłowy
    
    params = {"q": quoted_title, "action": "SimpleSearchAction", "type": "-6", "p": "0", "tab": "all"}
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Błąd dla: {title}")
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Szukamy ID w sekcji "objectbox--main"
    objectbox = soup.find("div", class_="objectbox objectbox--main")
    if objectbox:
        id_link = objectbox.find("a", href=True)
        if id_link and "publication/" in id_link["href"]:
            href_parts = id_link["href"].split("/")
            if "edition" in href_parts:
                edition_index = href_parts.index("edition") + 1  # Pobranie ID edycji
                return href_parts[edition_index] if edition_index < len(href_parts) else None
        
    return None  # Jeśli nie znaleziono



# Pobranie identyfikatorów dla każdego tytułu
wyniki = {}
for tytul in tytuly:
    print(f"Przetwarzam: {tytul}")
    identyfikator = get_rcin_id(tytul)
    print(identyfikator)
    wyniki[tytul] = identyfikator
    time.sleep(1)

# Zapisanie wyników do Excela

df_czasopisma["RCIN_ID"] = df_czasopisma["title"].map(wyniki)

# Zapisujemy arkusze do nowego pliku, ale nie wczytujemy całego pliku ponownie!
with pd.ExcelWriter(plik_wyjsciowy, engine="openpyxl") as writer:
    # Zapisujemy zaktualizowany arkusz "Czasopisma"
    df_czasopisma.to_excel(writer, sheet_name="Czasopisma", index=False)
    
    # Wczytujemy i zapisujemy pozostałe arkusze bez zmian
    xls = pd.ExcelFile(plik_wejsciowy)
    for sheet in xls.sheet_names:
        if sheet != "Czasopisma":  # Pomijamy ten, który już mamy zaktualizowany
            df_temp = pd.read_excel(xls, sheet_name=sheet)
            df_temp.to_excel(writer, sheet_name=sheet, index=False)

print("✅ Zakończono! Wyniki zapisane w 'KDL_wszystkie_zasoby_z_rcin.xlsx'.")