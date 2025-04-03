# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 13:36:05 2025

@author: darek
"""

import pandas as pd

# Lista plików do przetworzenia
files = [
    "C:/Users/darek/Downloads/Korpus programów wydarzeń -- AW.xlsx",
    "C:/Users/darek/Downloads/Korpus programów wydarzeń -- PCL.xlsx",
    "C:/Users/darek/Downloads/Korpus programów wydarzeń -- MŚ.xlsx",
    "C:/Users/darek/Downloads/Korpus programów wydarzeń -- MSz.xlsx"
]

# Inicjalizacja pustej listy na dane
all_events = []
df=[]
# Przetwarzanie każdego pliku
for file in files:
    df = pd.read_excel(file, usecols=["Typ wydarzenia"])  # Wczytaj tylko interesującą nas kolumnę
    all_events.extend(df["Typ wydarzenia"].dropna().tolist())  # Usuń puste wartości i dodaj do listy
    
for events in all_events:
    

# Rozdzielanie typów wydarzeń po ";", ","
split_events = []
for event in all_events:
    event_list = event.replace(";", ",").split(",")  # Zamieniamy ";" na "," i rozdzielamy
    split_events.extend([e.strip().lower() for e in event_list])  # Usuwamy zbędne spacje i zmieniamy na małe litery

# Poprawki literówek w nazwach wydarzeń
corrections = {
    "festiwal fimowy": "festiwal filmowy",
    "fstiwal teatralny": "festiwal teatralny",
    "festiwa teatralny": "festiwal teatralny",
    "wydarzenie (teatrologiczne)": "wydarzenie naukowe (teatrologiczne)"
}

# Zastosowanie poprawek
split_events = [corrections.get(event, event) for event in split_events]

# Tworzenie DataFrame z policzonymi wartościami
event_counts = pd.Series(split_events).value_counts()

import pandas as pd

# Słownik poprawek literówek
corrections = {
    "festiwal fimowy": "festiwal filmowy",
    "fstiwal teatralny": "festiwal teatralny",
    "festiwa teatralny": "festiwal teatralny",
    "wydarzenie (teatrologiczne)": "wydarzenie naukowe (teatrologiczne)"
}

all_events = []
num_rows = 0

# Wczytanie danych
for file in files:
    df = pd.read_excel(file, usecols=["Typ wydarzenia"])
    num_rows += len(df)
    all_events.extend(df["Typ wydarzenia"].dropna().tolist())

simplified_events = []

for event in all_events:
    # Rozdzielenie po ";" i ","
    event_list = event.replace(";", ",").split(",")

    # Usunięcie zbędnych spacji i zamiana na małe litery
    event_list = [e.strip().lower() for e in event_list]

    # Zastosowanie korekt literówek
    event_list = [corrections.get(e, e) for e in event_list]

    # Każdy element event_list może zawierać "festiwal" lub "wydarzenie naukowe", bądź nic z tych rzeczy
    tmp_list = []
    for e in event_list:
        # Sprawdzamy, czy "festiwal" występuje w e
        found_any = False
        if "festiwal" in e:
            tmp_list.append("festiwal")
            found_any = True
        
        # Sprawdzamy, czy "wydarzenie naukowe" występuje w e
        if "wydarzenie naukowe" in e:
            tmp_list.append("wydarzenie naukowe")
            found_any = True

        # Jeśli nie znaleziono "festiwal" ani "wydarzenie naukowe", zostaw oryginał
        if not found_any:
            tmp_list.append(e)

    # Dodanie zebranych kategorii z tego wiersza do głównej listy
    simplified_events.extend(tmp_list)

event_counts = pd.Series(simplified_events).value_counts()

df_result = pd.DataFrame(event_counts, columns=["Liczba wystąpień"])
df_result.index.name = "Typ wydarzenia"

print(f"Liczba wierszy wziętych na warsztat: {num_rows}")
print(df_result)


