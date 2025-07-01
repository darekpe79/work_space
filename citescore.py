# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 08:27:42 2025

@author: darek
"""
import pandas as pd
import numpy as np

# 1. Wczytaj pliki
df1 = pd.read_excel("C:/Users/darek/Downloads/20240105_Wykaz_czasopism_naukowych_2024_styczeń (1).xlsx")
df2 = pd.read_excel("C:/Users/darek/Downloads/CiteScore 2024.xlsx", sheet_name="Arkusz1")

# 2. Oczyść pola ISSN w df1
issn_cols_df1 = ['issn1', 'e-issn1', 'issn', 'e-issn']
import re

def clean_issn(x):
    # zostawiamy tylko cyfry
    s = re.sub(r'\D', '', str(x))
    return s if s else np.nan

# przykład dla wszystkich kolumn:
for col in ['issn1','e-issn1','issn','e-issn']:
    df1[col] = df1[col].apply(clean_issn)
df2['Print ISSN'] = df2['Print ISSN'].apply(clean_issn)
df2['E-ISSN']    = df2['E-ISSN'].apply(clean_issn)

# 3. Oczyść pola ISSN w df2
df2['Print ISSN'] = df2['Print ISSN'].astype(str).str.strip()
df2['E-ISSN']    = df2['E-ISSN'].astype(str).str.strip()

# 4. Zbuduj słownik ISSN → Top 10%
mapping = {}
print(mapping)
for _, row in df2.iterrows():
    top10 = row['Top 10% (CiteScore Percentile)']
    if pd.notna(row['Print ISSN']) and row['Print ISSN'] != '':
        mapping[row['Print ISSN']] = top10
    if pd.notna(row['E-ISSN']) and row['E-ISSN'] != '':
        mapping[row['E-ISSN']] = top10

# 5. Funkcja, która dla listy czterech wartości ISSN zwraca pierwszą pasującą Top10
def find_top10(series):
    for val in series:
        if val in mapping:
            return mapping[val]
    return np.nan

# 6. Dodaj kolumnę do df1
df1['Top 10% (CiteScore Percentile)'] = df1[issn_cols_df1].apply(find_top10, axis=1)

# 7. (Opcjonalnie) Statystyka uzupełnień
filled = df1['Top 10% (CiteScore Percentile)'].notna().sum()
total  = len(df1)
print(f"Uzupełniono {filled} z {total} rekordów ({filled/total:.1%})")




