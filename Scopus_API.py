# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:01:08 2024

@author: dariu
"""

import requests
import json

# Klucz API oraz token instytucji
APIKey = '2caaa4e31438819db1f38cddb8a083fa'
InstToken = '806bdb779375cb74279272f2e5b61370'
DOI = '10.1016/j.jssc.2020.121858'  # Przykładowy DOI, który możesz zamienić na dowolny inny

# Adres URL bazowy dla API Scopus
url = 'https://api.elsevier.com/content/search/scopus'

# Nagłówki z wymaganymi informacjami autoryzacyjnymi
headers = {
    'X-ELS-APIKey': APIKey,
    'X-ELS-Insttoken': InstToken,
    'Accept': 'application/json'
}

# Parametry zapytania
params = {
    'query': DOI,
    'view': 'COMPLETE'
}

# Wysłanie żądania GET do API
response = requests.get(url, headers=headers, params=params)
response.status_code
# Sprawdzenie, czy odpowiedź jest prawidłowa (kod 200 oznacza sukces)
if response.status_code == 200:
    # Przetwarzanie danych JSON
    data = response.json()
    print('Pobrano', len(json.dumps(data)), 'znaków')
    print(json.dumps(data, indent=4))  # Wydrukowanie sformatowanego JSON-a (opcjonalnie)
else:
    print('Błąd w zapytaniu:', response.status_code)
    print(response.text)
import requests
import json

# Klucz API i instytucjonalny token
APIKey = '2caaa4e31438819db1f38cddb8a083fa'
InstToken = '806bdb779375cb74279272f2e5b61370'
author_id = '36903319600'  # Przykładowy Author ID, zamień na aktualny ID autora

# Adres URL dla pobrania profilu autora
url = f'https://api.elsevier.com/content/author/author_id/{author_id}'

# Nagłówki z kluczem API i instytucjonalnym tokenem
headers = {
    'X-ELS-APIKey': APIKey,
    'X-ELS-Insttoken': InstToken,
    'Accept': 'application/json'
}

# Parametry zapytania, z widokiem `ENHANCED` dla pełniejszych informacji
params = {
    'view': 'ENHANCED'
}

# Wysłanie żądania GET do API
response = requests.get(url, headers=headers, params=params)

# Sprawdzenie, czy odpowiedź jest poprawna (kod 200 oznacza sukces)
if response.status_code == 200:
    # Przetwarzanie danych JSON
    data = response.json()
    print('Pobrano dane autora:')
    print(json.dumps(data, indent=4))  # Wyświetlenie sformatowanych danych JSON
else:
    print('Błąd w zapytaniu:', response.status_code)
    print(response.text)
    
    
import requests
import json

# Klucz API oraz token instytucji
APIKey = '2caaa4e31438819db1f38cddb8a083fa'
InstToken = '806bdb779375cb74279272f2e5b61370'
ScopusID = '2-s2.0-85202154270'  # Przykładowy Scopus ID

# Adres URL bazowy dla API Scopus
url = 'https://api.elsevier.com/content/search/scopus'

# Nagłówki z wymaganymi informacjami autoryzacyjnymi
headers = {
    'X-ELS-APIKey': APIKey,
    'X-ELS-Insttoken': InstToken,
    'Accept': 'application/json'
}

# Parametry zapytania z użyciem Scopus ID
params = {
    'query': f'EID({ScopusID})',  # Wyszukiwanie po Scopus ID
    'view': 'COMPLETE'
}

# Wysłanie żądania GET do API
response = requests.get(url, headers=headers, params=params)

# Sprawdzenie, czy odpowiedź jest prawidłowa
if response.status_code == 200:
    data = response.json()
    print('Pobrano dane publikacji:')
    print(json.dumps(data, indent=4))  # Wyświetlenie sformatowanych danych JSON
else:
    print('Błąd w zapytaniu:', response.status_code)
    print(response.text)
with open('author_scopus.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Ścieżka do WebDrivera (przykład dla Chrome, dostosuj dla Firefox)
driver_path = '/path/to/chromedriver'  # lub 'geckodriver' dla Firefox
url = 'https://www.scopus.com/record/display.uri?eid=2-s2.0-85096179373&origin=resultslist&sort=plf-f&src=s&sid=b8d8ebc91eae462aac2f596b6a1e3c98&sot=b&sdt=b&s=DOI%2810.1016%2Fj.jssc.2020.121858%29&sl=31&sessionSearchId=b8d8ebc91eae462aac2f596b6a1e3c98&relpos=0'

# Ustawienia przeglądarki
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Opcjonalnie, aby uruchomić w trybie bez interfejsu
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Inicjalizacja przeglądarki
driver = webdriver.Chrome(executable_path=driver_path, options=options)

# Przejście do strony
driver.get(url)

# Czekamy, aż JavaScript załaduje pełną treść (można dostosować czas)
time.sleep(5)

# Pobieranie całego HTML po pełnym załadowaniu strony
page_source = driver.page_source

# Zapisanie lub przetwarzanie źródła strony
print(page_source)  # Wyświetlenie pełnego HTML-a strony

# Zamknięcie przeglądarki
driver.quit()
