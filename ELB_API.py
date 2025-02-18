# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:17:15 2025

@author: darek
"""

import requests
from pymarc import MARCReader, MARCWriter

# 1. Ustalmy podstawowy endpoint wyszukiwania.
base_search_url = "https://testlibri.ucl.cas.cz/pl/api/biblio"

# 2. Zbudujmy parametry wyszukiwania.
#    Możemy modyfikować:
#    - 'lookfor' : tekst wyszukiwany
#    - 'type'    : pole, w którym szukamy (allfields, title, author)
#    - 'page'    : numer strony (domyślnie 1)
#    - 'limit'   : ilość wyników na stronę (1-100)
#    - 'sort'    : sortowanie (np. 'relevance', 'title_sort asc', 'author_sort desc' itp.)
#    - 'resultSize' : 'small' (domyślne) lub 'extended'
#    - 'withFacets' / 'useFacet' : jeśli chcemy używać dodatkowych filtrów.
#    Tutaj przykład wyszukiwania po autorze "Adam" i limit 5:

params = {
    'lookfor': 'Adam Mickiewicz',
    'type': 'author',
    'limit': 5,
    'resultSize': 'small'
}

params = {
    "lookfor": "Mickiewicz, Cezary",
    "type": "author",
    "withFacets": "subjects_str_mv",
    "useFacet": 'subjects_str_mv:"Hasła osobowe (literatura polska)"',
    "limit": 100,
    "page": 1
}

# 3. Wykonujemy zapytanie do API z powyższymi parametrami.
response = requests.get(base_search_url, params=params)

if response.status_code == 200:
    # 4. Odbieramy wyniki w formacie JSON.
    data = response.json()
    
    # Wyciągamy informację o całkowitej liczbie wyników.
    total_results = data.get('totalResults', 0)
    print(f"Znaleziono {total_results} wyników dla zapytania: {params['lookfor']}")

    # Z listy 'docs' dostaniemy poszczególne rekordy (w wersji 'small' mamy m.in. 'id', 'title', 'author', 'lp').
    docs = data.get('docs', [])
    
    if not docs:
        print("Brak wyników do pobrania.")
    else:
        # 5. Przygotowujemy plik MARC do zapisu (tryb 'wb' - zapis binarny).
        writer = MARCWriter(open('moje_rekordy.mrc', 'wb'))
        
        # 6. Iterujemy po znalezionych rekordach i pobieramy ich pełną reprezentację .mrc.
        for doc in docs[:4]:
            record_id = doc.get('id')
            title = doc.get('title', '---')
            
            if record_id:
                # URL do pojedynczego rekordu w formacie MARC
                single_record_url = f"https://testlibri.ucl.cas.cz/pl/results/biblio/record/{record_id}.mrc"
                
                print(f"Pobieram rekord o ID {record_id} (tytuł: {title})...")
                single_record_resp = requests.get(single_record_url)
                print (single_record_resp)
                
                if single_record_resp.status_code == 200:
                    # Pobrane dane .mrc czytamy za pomocą MARCReader.
                    # single_record_resp.content to ciąg bajtów.
                    marc_data = single_record_resp.content
                    
                    print(marc_data)
                    
                    # Używamy MARCReader do wczytania i zapisujemy do pliku przez writer.
                    for record in MARCReader(marc_data):
                        if record is not None:
                            print (record)
                            writer.write(record)
                else:
                    print(f"Nie udało się pobrać rekordu {record_id}. Kod HTTP: {single_record_resp.status_code}")
            else:
                print("Brak ID rekordu (nie można pobrać wersji .mrc).")
        
        # 7. Zamykanie pliku z zapisanymi rekordami MARC.
        writer.close()
        print("Zakończono pobieranie i zapisywanie rekordów do pliku 'moje_rekordy.mrc'.")
else:
    print(f"Błąd przy połączeniu z API. Kod HTTP: {response.status_code}")
