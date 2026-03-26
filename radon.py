# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:19:21 2026

@author: darek
"""

import requests
import json

URL = "https://radon.nauka.gov.pl/opendata/polon/impacts"

params = {
    "pageNumber": 0,
    "pageSize": 5
}

response = requests.get(URL, params=params)
response.raise_for_status()

data = response.json()

print("KLUCZE:", data.keys())
print("LICZBA REKORDÓW:", len(data.get("results", [])))

# pokaż pierwszy rekord ładnie
first = data["results"][0]
print(json.dumps(first, indent=2, ensure_ascii=False))

import requests
import json

URL = "https://radon.nauka.gov.pl/opendata/polon/impacts"

for page in [0, 1, 2]:
    params = {
        "pageNumber": page,
        "pageSize": 5
    }

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])

    print(f"\n=== PAGE {page} ===")
    print("pagination:", json.dumps(data.get("pagination", {}), ensure_ascii=False, indent=2))
    print("liczba rekordów:", len(results))

    if results:
        print("pierwszy impactUuid:", results[0].get("impactUuid"))
        print("pierwszy titlePl:", results[0].get("titlePl"))
        
        
import requests
import json

URL = "https://radon.nauka.gov.pl/opendata/polon/impacts"

current_token = None

for page in range(1):
    params = {
        "pageSize": 1
    }
    
    # Jeśli mamy token z poprzedniej strony, dodajemy go do parametrów
    if current_token:
        params["token"] = current_token
    
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    
    results = data.get("results", [])
    pagination = data.get("pagination", {})
    
    # Pobieramy nowy token na potrzeby kolejnej iteracji
    current_token = pagination.get("token")
    
    print(f"\n=== POBRANO STRONĘ {page} ===")
    print(f"Token użyty do przejścia dalej: {current_token}")
    print(f"Liczba rekordów: {len(results)}")
    
    if results:
        # Wyświetlamy np. ostatni ID z tej strony, żeby widzieć, że dane się zmieniają
        print(f"Pierwszy impactUuid na tej stronie: {results[0].get('impactUuid')}")
        print(f"Tytuł: {results[0].get('titlePl')[:100]}...")
        
        
        
        
import requests
import json

URL = "https://radon.nauka.gov.pl/opendata/polon/impacts"

current_token = None

for page in range(1):
    params = {
        "pageSize": 1
    }

    if current_token:
        params["token"] = current_token

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    pagination = data.get("pagination", {})

    current_token = pagination.get("token")

    print(f"\n=== POBRANO STRONĘ {page} ===")
    print(f"Token użyty do przejścia dalej: {current_token}")
    print(f"Liczba rekordów: {len(results)}")

    if results:
        print(f"Pierwszy impactUuid na tej stronie: {results[0].get('impactUuid')}")
        print(f"Tytuł: {results[0].get('titlePl')[:100]}...")

        with open("results_page_0.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("Zapisano do pliku: results_page_0.json")
        
        
        
        
import requests
import json
import time

URL = "https://radon.nauka.gov.pl/opendata/polon/impacts"

current_token = None
all_results = []
seen_ids = set()

while True:
    params = {
        "pageSize": 1   # i tak wygląda, że API zwraca po 10
    }

    if current_token:
        params["token"] = current_token

    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    pagination = data.get("pagination", {})
    new_token = pagination.get("token")
    max_count = pagination.get("maxCount")

    print(f"\n=== NOWA PACZKA ===")
    print(f"Liczba rekordów w paczce: {len(results)}")
    print(f"Nowy token: {new_token}")
    print(f"maxCount z API: {max_count}")

    if not results:
        print("Brak wyników. Koniec.")
        break

    added_now = 0

    for rec in results:
        impact_uuid = rec.get("impactUuid")
        if impact_uuid and impact_uuid not in seen_ids:
            seen_ids.add(impact_uuid)
            all_results.append(rec)
            added_now += 1

    print(f"Dodano nowych rekordów: {added_now}")
    print(f"Łącznie unikalnych rekordów: {len(all_results)}")

    if not new_token or new_token == current_token:
        print("Brak nowego tokenu albo token się nie zmienia. Stop.")
        break

    current_token = new_token
    time.sleep(0.3)

with open("all_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\nGotowe.")
print(f"Zapisano {len(all_results)} rekordów do all_results.json")