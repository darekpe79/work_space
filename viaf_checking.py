# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:40:00 2024

@author: dariu
"""


import re
import requests
from fuzzywuzzy import fuzz
from urllib.parse import urlencode

def preprocess_text(text):
    # Usuwanie dat w formacie YYYY-YYYY lub YYYY
    text = re.sub(r'\b\d{4}-\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    
    # Usuwanie dat w formacie (YYYY-YYYY) lub (YYYY)
    text = re.sub(r'\(\d{4}-\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    
    # Usuwanie nawiasów, które mogą pozostać po usunięciu dat
    text = re.sub(r'\(\)', '', text)
    
    # Usuwanie nadmiarowych spacji, które mogłyby się pojawić po usunięciu dat i nawiasów
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_text_from_main_headings(main_headings):
    if isinstance(main_headings, list):
        return [heading.get('text') for heading in main_headings if heading.get('text')]
    if isinstance(main_headings, dict):
        return [main_headings.get('text')]
    return []

def check_viaf_with_fuzzy_match(entity_name, threshold=87, max_pages=10, entity_type='personalNames'):
    base_url = "https://viaf.org/viaf/search"
    matches = []
    
    try:
        for page in range(1, max_pages + 1):
            query = f'local.{entity_type} all "{entity_name}"'
            query_params = {
                'query': query,
                'maximumRecords': 10,
                'startRecord': (page - 1) * 10 + 1,
                'httpAccept': 'application/json'
            }
            url = f"{base_url}?{urlencode(query_params)}"
            print(f"Query URL: {url}")
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                for record in data['searchRetrieveResponse']['records']:
                    record_data = record['record'].get('recordData', {})
                    viaf_id = record_data.get('viafID')
                    main_headings = record_data.get('mainHeadings', {})
                    main_headings_texts = extract_text_from_main_headings(main_headings.get('data', []))

                    for main_heading in main_headings_texts:
                        score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                        if score_with_date >= threshold:
                            matches.append((viaf_id, score_with_date))
                        
                        term_without_date = preprocess_text(main_heading)
                        score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                        if score_without_date >= threshold:
                            matches.append((viaf_id, score_without_date))
            else:
                break
    
    except requests.RequestException as e:
        print(f"Error querying VIAF: {e}")
    
    # Usuwanie duplikatów
    unique_matches = list(set(matches))
    
    filtered_matches = [match for match in unique_matches if match[1] == 100]
    
    if filtered_matches:
        result_urls = [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches]
    elif unique_matches:
        best_match = max(unique_matches, key=lambda x: x[1])
        result_urls = [(f"https://viaf.org/viaf/{best_match[0]}", best_match[1])]
    else:
        result_urls = []

    return result_urls if result_urls else None
# Przykład użycia
entity_name = 'Nikodem Wołczuk'
results = check_viaf_with_fuzzy_match(entity_name)
if results:
    print(results)
else:
    print("No matches found.")


results_text = ', '.join([ent[0] for ent in results])
results = check_viaf_with_fuzzy_match("World Health Organization", entity_type="corporateNames")

# Przykład użycia

entity_name = 'Domagała, Tomasz'
#%%
import re
import requests
from fuzzywuzzy import fuzz
from urllib.parse import urlencode

def preprocess_text(text):
    # Usuwanie dat w formacie YYYY-YYYY lub YYYY
    text = re.sub(r'\b\d{4}-\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    
    # Usuwanie dat w formacie (YYYY-YYYY) lub (YYYY)
    text = re.sub(r'\(\d{4}-\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    
    # Usuwanie nawiasów, które mogą pozostać po usunięciu dat
    text = re.sub(r'\(\)', '', text)
    
    # Usuwanie nadmiarowych spacji, które mogłyby się pojawić po usunięciu dat i nawiasów
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_main_headings(main_headings):
    if isinstance(main_headings, list):
        return [heading.get('text') for heading in main_headings if heading.get('text')]
    if isinstance(main_headings, dict):
        return [main_headings.get('text')]
    return []

def check_viaf_with_fuzzy_match(entity_name, threshold=87, max_pages=10, entity_type='personalNames'):
    base_url = "https://viaf.org/viaf/search"
    matches = []

    def search_viaf(entity_name_query):
        try:
            for page in range(1, max_pages + 1):
                query = f'local.{entity_type} all "{entity_name_query}"'
                query_params = {
                    'query': query,
                    'maximumRecords': 10,
                    'startRecord': (page - 1) * 10 + 1,
                    'httpAccept': 'application/json'
                }
                url = f"{base_url}?{urlencode(query_params)}"
                print(f"Query URL: {url}")
                
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                
                if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                    for record in data['searchRetrieveResponse']['records']:
                        record_data = record['record'].get('recordData', {})
                        viaf_id = record_data.get('viafID')
                        main_headings = record_data.get('mainHeadings', {})
                        main_headings_texts = extract_text_from_main_headings(main_headings.get('data', []))

                        for main_heading in main_headings_texts:
                            score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                            if score_with_date >= threshold:
                                matches.append((viaf_id, score_with_date))
                            
                            term_without_date = preprocess_text(main_heading)
                            score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                            if score_without_date >= threshold:
                                matches.append((viaf_id, score_without_date))
                else:
                    break
        except requests.RequestException as e:
            print(f"Error querying VIAF: {e}")

    # Pierwsze wyszukiwanie z nazwą encji
    search_viaf(entity_name)
    
    # Jeśli nie znaleziono wyników, drugie wyszukiwanie bez nazwy encji
    if not matches:
        search_viaf("")

    # Usuwanie duplikatów
    unique_matches = list(set(matches))
    
    filtered_matches = [match for match in unique_matches if match[1] == 100]
    
    if filtered_matches:
        result_urls = [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches]
    elif unique_matches:
        best_match = max(unique_matches, key=lambda x: x[1])
        result_urls = [(f"https://viaf.org/viaf/{best_match[0]}", best_match[1])]
    else:
        result_urls = []

    return result_urls if result_urls else None

import re
import requests
from fuzzywuzzy import fuzz
from urllib.parse import urlencode

def preprocess_text(text):
    text = re.sub(r'\b\d{4}-\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\(\d{4}-\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'\(\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_main_headings(main_headings):
    if isinstance(main_headings, list):
        return [heading.get('text') for heading in main_headings if heading.get('text')]
    if isinstance(main_headings, dict):
        return [main_headings.get('text')]
    return []

def check_viaf_with_fuzzy_match(entity_name, threshold=87, max_pages=5, entity_type='personalNames'):
    base_url_search = "https://viaf.org/viaf/search"
    matches = []

    try:
        for page in range(1, max_pages + 1):
            query = f'local.{entity_type} all "{entity_name}"'
            query_params = {
                'query': query,
                'maximumRecords': 10,
                'startRecord': (page - 1) * 10 + 1,
                'httpAccept': 'application/json'
            }
            url = f"{base_url_search}?{urlencode(query_params)}"
            print(f"Query URL: {url}")
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                for record in data['searchRetrieveResponse']['records']:
                    record_data = record['record'].get('recordData', {})
                    viaf_id = record_data.get('viafID')
                    main_headings = record_data.get('mainHeadings', {})
                    main_headings_texts = extract_text_from_main_headings(main_headings.get('data', []))

                    for main_heading in main_headings_texts:
                        score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                        if score_with_date >= threshold:
                            matches.append((viaf_id, score_with_date))
                        
                        term_without_date = preprocess_text(main_heading)
                        score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                        if score_without_date >= threshold:
                            matches.append((viaf_id, score_without_date))
            else:
                break
    except requests.RequestException as e:
        print(f"Error querying VIAF search: {e}")

    # AutoSuggest jeśli brak wyników
    if not matches:
        base_url_suggest = "http://viaf.org/viaf/AutoSuggest"
        query_params = {'query': entity_name}
        try:
            response = requests.get(base_url_suggest, params=query_params)
            response.raise_for_status()
            data = response.json()

            if data and data.get('result') is not None:
                best_score = 0
                best_match = None
                for result in data['result'][:10]:
                    original_term = result.get('term')
                    score_with_date = fuzz.token_sort_ratio(entity_name, original_term)
                    if score_with_date > best_score and score_with_date >= threshold:
                        best_score = score_with_date
                        best_match = result
                    
                    term_without_date = preprocess_text(original_term)
                    score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                    if score_without_date > best_score and score_without_date >= threshold:
                        best_score = score_without_date
                        best_match = result

                if best_match:
                    viaf_id = best_match.get('viafid')
                    return [(f"http://viaf.org/viaf/{viaf_id}", best_score)]
        except requests.RequestException as e:
            print(f"Error querying VIAF AutoSuggest: {e}")

    # Usuwanie duplikatów i wybieranie najlepszych dopasowań
    unique_matches = list(set(matches))
    filtered_matches = [match for match in unique_matches if match[1] == 100]
    
    if filtered_matches:
        result_urls = [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches]
    elif unique_matches:
        best_match = max(unique_matches, key=lambda x: x[1])
        result_urls = [(f"https://viaf.org/viaf/{best_match[0]}", best_match[1])]
    else:
        result_urls = []

    return result_urls if result_urls else None

# Przykład użycia
entity_name = 'Tomasz Domagała'
results = check_viaf_with_fuzzy_match(entity_name)
if results:
    print(results)
else:
    print("No matches found.")




# Przykład użycia
entity_name = 'Nikodem Wołczuk'
results = check_viaf_with_fuzzy_match(entity_name)
if results:
    print(results)
else:
    print("No matches found.")
entity_name = 'Pan Tadeusz'    
viaf_result = check_viaf_with_fuzzy_match(entity_name, threshold=30, entity_type='uniformTitles')

import re
import requests
from fuzzywuzzy import fuzz
from urllib.parse import urlencode

def preprocess_text(text):
    # Usuwanie dat w formacie YYYY-YYYY lub YYYY
    text = re.sub(r'\b\d{4}-\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    
    # Usuwanie dat w formacie (YYYY-YYYY) lub (YYYY)
    text = re.sub(r'\(\d{4}-\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    
    # Usuwanie nawiasów i nadmiarowych spacji
    text = re.sub(r'\(\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_main_headings(main_headings):
    if isinstance(main_headings, list):
        return [heading.get('text') for heading in main_headings if heading.get('text')]
    if isinstance(main_headings, dict):
        return [main_headings.get('text')]
    return []
def extract_text_from_main_headings(record_data):
    main_headings = []
    
    # Przetwarzanie tylko głównych tytułów w 'mainHeadings'
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        
        if isinstance(main_headings_data.get('data'), list):
            main_headings.extend(heading.get('text') for heading in main_headings_data['data'] if heading.get('text'))
        elif isinstance(main_headings_data.get('data'), dict):
            main_headings.append(main_headings_data['data'].get('text'))

    return main_headings

def check_viaf_with_fuzzy_match(entity_name, threshold=87, max_pages=5, entity_type=None):
    base_url_search = "https://viaf.org/viaf/search"
    matches = []

    def search_viaf(query):
        try:
            for page in range(1, max_pages + 1):
                query_params = {
                    'query': query,
                    'maximumRecords': 10,
                    'startRecord': (page - 1) * 10 + 1,
                    'httpAccept': 'application/json'
                }
                url = f"{base_url_search}?{urlencode(query_params)}"
                print(f"Query URL: {url}")
                
                response = requests.get(url)
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError:
                    print("Invalid JSON response:", response.text)
                    return None  # Możesz również zignorować ten krok lub dodać obsługę błędu
                
                # Kontynuuj, jeśli JSON jest poprawny
                if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                    # Przetwarzanie wyników
                    for record in data['searchRetrieveResponse']['records']:
                        record_data = record['record'].get('recordData', {})
                        viaf_id = record_data.get('viafID')
                        main_headings = record_data.get('mainHeadings', {})
                        main_headings_texts = extract_text_from_main_headings(main_headings.get('data', []))
                        
                        for main_heading in main_headings_texts:
                            score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                            if score_with_date >= threshold:
                                matches.append((viaf_id, score_with_date))
                            
                            term_without_date = preprocess_text(main_heading)
                            score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                            if score_without_date >= threshold:
                                matches.append((viaf_id, score_without_date))
                else:
                    break
        except requests.RequestException as e:
            print(f"Error querying VIAF search: {e}")


    # Pierwsze wyszukiwanie z typem encji
    if entity_type:
        query = f'local.{entity_type} all "{entity_name}"'
        search_viaf(query)

    # Jeśli brak wyników, wyszukiwanie bez typu encji
    if not matches:
        print(f"Brak wyników przy wyszukiwaniu z typem encji dla: {entity_name}. Szukanie bez typu encji...")
        query = f'"{entity_name}"'  # Bez 'all'
        search_viaf(query)

    # AutoSuggest jeśli brak wyników po obu powyższych wyszukiwaniach
    if not matches:
        base_url_suggest = "http://viaf.org/viaf/AutoSuggest"
        query_params = {'query': entity_name}
        try:
            response = requests.get(base_url_suggest, params=query_params)
            response.raise_for_status()
            data = response.json()

            if data and data.get('result') is not None:
                best_score = 0
                best_match = None
                for result in data['result'][:10]:
                    original_term = result.get('term')
                    score_with_date = fuzz.token_sort_ratio(entity_name, original_term)
                    if score_with_date > best_score and score_with_date >= threshold:
                        best_score = score_with_date
                        best_match = result
                    
                    term_without_date = preprocess_text(original_term)
                    score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                    if score_without_date > best_score and score_without_date >= threshold:
                        best_score = score_without_date
                        best_match = result

                if best_match:
                    viaf_id = best_match.get('viafid')
                    return [(f"http://viaf.org/viaf/{viaf_id}", best_score)]
        except requests.RequestException as e:
            print(f"Error querying VIAF AutoSuggest: {e}")

    # Usuwanie duplikatów i wybieranie najlepszych dopasowań
    unique_matches = list(set(matches))
    filtered_matches = [match for match in unique_matches if match[1] == 100]
    
    if filtered_matches:
        result_urls = [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches]
    elif unique_matches:
        best_match = max(unique_matches, key=lambda x: x[1])
        result_urls = [(f"https://viaf.org/viaf/{best_match[0]}", best_match[1])]
    else:
        result_urls = []

    return result_urls if result_urls else None

# Przykład użycia
entity_name = 'Wyjechali'
results = check_viaf_with_fuzzy_match(entity_name, entity_type='uniformTitles')
if results:
    print(results)
else:
    print("No matches found.")
def extract_text_from_main_headings(record_data):
    # Przetwarzaj tylko główne tytuły bez tagów alternatywnych (430)
    main_headings = []
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        
        # Przetwarzanie tylko 'mainHeadings' bez sprawdzania 'x400s'
        if isinstance(main_headings_data.get('data'), list):
            main_headings.extend(heading.get('text') for heading in main_headings_data['data'] if heading.get('text'))
        elif isinstance(main_headings_data.get('data'), dict):
            main_headings.append(main_headings_data['data'].get('text'))
    
    return main_headings
def extract_text_from_main_headings(main_headings):
    if isinstance(main_headings, list):
        return [heading.get('text') for heading in main_headings if heading.get('text')]
    if isinstance(main_headings, dict):
        return [main_headings.get('text')]
    return []
#%% LAST ver
import re
import requests
from fuzzywuzzy import fuzz
from urllib.parse import urlencode

def preprocess_text(text):
    text = re.sub(r'\b\d{4}-\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\(\d{4}-\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_main_headings(record_data):
    main_headings = []
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        if isinstance(main_headings_data.get('data'), list):
            main_headings.extend(heading.get('text') for heading in main_headings_data['data'] if heading.get('text'))
        elif isinstance(main_headings_data.get('data'), dict):
            main_headings.append(main_headings_data['data'].get('text'))
    return main_headings

def process_x400s_for_non_430(record_data):
    main_headings = []
    x400s = record_data.get('x400s', {}).get('x400', [])
    
    # Jeśli `x400s` jest listą, iteruj po niej
    if isinstance(x400s, list):
        for x400 in x400s:
            # Sprawdź, czy x400 jest słownikiem, zanim spróbujesz uzyskać dostęp do jego kluczy
            if isinstance(x400, dict) and x400.get('datafield', {}).get('@tag') != '430':
                subfields = x400.get('datafield', {}).get('subfield', [])
                # Obsługa przypadku, gdy subfield jest pojedynczym słownikiem
                if isinstance(subfields, dict):
                    if subfields.get('@code') == 'a':
                        main_headings.append(subfields.get('#text'))
                # Obsługa przypadku, gdy subfield jest listą
                elif isinstance(subfields, list):
                    for subfield in subfields:
                        if subfield.get('@code') == 'a':
                            main_headings.append(subfield.get('#text'))
    return main_headings


def check_viaf_with_fuzzy_match(entity_name, threshold=87, max_pages=5, entity_type=None):
    base_url_search = "https://viaf.org/viaf/search"
    matches = []

    def search_viaf(query):
        try:
            for page in range(1, max_pages + 1):
                query_params = {
                    'query': query,
                    'maximumRecords': 10,
                    'startRecord': (page - 1) * 10 + 1,
                    'httpAccept': 'application/json'
                }
                url = f"{base_url_search}?{urlencode(query_params)}"
                print(f"Query URL: {url}")
                
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                    for record in data['searchRetrieveResponse']['records']:
                        record_data = record['record'].get('recordData', {})
                        viaf_id = record_data.get('viafID')

                        # Przetwarzanie mainHeadings
                        main_headings_texts = extract_text_from_main_headings(record_data)
                        main_headings_texts.extend(process_x400s_for_non_430(record_data))
                        print(f"Przetwarzanie głównych tytułów dla rekordu {viaf_id}: {main_headings_texts}")
                        
                        for main_heading in main_headings_texts:
                            score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                            if score_with_date >= threshold:
                                print(f"Dopasowanie z datą dla {viaf_id}: {main_heading} -> {score_with_date}")
                                matches.append((viaf_id, score_with_date))
                            
                            term_without_date = preprocess_text(main_heading)
                            score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                            if score_without_date >= threshold:
                                print(f"Dopasowanie bez daty dla {viaf_id}: {term_without_date} -> {score_without_date}")
                                matches.append((viaf_id, score_without_date))
                else:
                    break
        except requests.RequestException as e:
            print(f"Error querying VIAF search: {e}")
    
    if entity_type:
        query = f'local.{entity_type} all "{entity_name}"'
        search_viaf(query)
    
    if not matches:
        print(f"Brak wyników przy wyszukiwaniu z typem encji dla: {entity_name}. Szukanie bez typu encji...")
        query = f'"{entity_name}"'
        search_viaf(query)
    
    if not matches:
        base_url_suggest = "http://viaf.org/viaf/AutoSuggest"
        query_params = {'query': entity_name}
        try:
            response = requests.get(base_url_suggest, params=query_params)
            response.raise_for_status()
            data = response.json()
            
            if data and data.get('result') is not None:
                print("AutoSuggest Results Found")
                for result in data['result'][:10]:
                    original_term = result.get('term')
                    viaf_id = result.get('viafid')

                    # Ignorujemy tag 430
                    record_data = requests.get(f"https://viaf.org/viaf/{viaf_id}/viaf.json").json()
                    main_headings_texts = process_x400s_for_non_430(record_data)
                    print(f"Przetwarzanie wyników AutoSuggest dla rekordu {viaf_id}: {main_headings_texts}")
                    
                    for main_heading in main_headings_texts:
                        score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                        if score_with_date >= threshold:
                            print(f"Dopasowanie z datą w AutoSuggest: {main_heading} -> {score_with_date}")
                            matches.append((viaf_id, score_with_date))
                        
                        term_without_date = preprocess_text(main_heading)
                        score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                        if score_without_date >= threshold:
                            print(f"Dopasowanie bez daty w AutoSuggest: {term_without_date} -> {score_without_date}")
                            matches.append((viaf_id, score_without_date))
        except requests.RequestException as e:
            print(f"Error querying VIAF AutoSuggest: {e}")
    
    unique_matches = list(set(matches))
    filtered_matches = [match for match in unique_matches if match[1] == 100]
    
    if filtered_matches:
        result_urls = [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches]
    elif unique_matches:
        best_match = max(unique_matches, key=lambda x: x[1])
        result_urls = [(f"https://viaf.org/viaf/{best_match[0]}", best_match[1])]
    else:
        result_urls = []
    
    return result_urls if result_urls else None

# Przykład użycia
entity_name = 'Wyjechali'
results = check_viaf_with_fuzzy_match(entity_name, entity_type='uniformTitles')
if results:
    print(results)
else:
    print("No matches found.")


def check_viaf_with_fuzzy_match2(entity_name, threshold=84, max_pages=3, entity_type='personalNames'):
    base_url = "https://viaf.org/viaf/search"
    matches = []
    
    try:
        for page in range(1, max_pages + 1):
            query = f'local.{entity_type} all "{entity_name}"'
            query_params = {
                'query': query,
                'maximumRecords': 10,
                'startRecord': (page - 1) * 10 + 1,
                'httpAccept': 'application/json'
            }
            url = f"{base_url}?{urlencode(query_params)}"
            print(f"Query URL: {url}")
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                for record in data['searchRetrieveResponse']['records']:
                    record_data = record['record'].get('recordData', {})
                    viaf_id = record_data.get('viafID')
                    main_headings = record_data.get('mainHeadings', {})
                    main_headings_texts = extract_text_from_main_headings(main_headings.get('data', []))

                    for main_heading in main_headings_texts:
                        score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                        if score_with_date >= threshold:
                            matches.append((viaf_id, score_with_date))
                        
                        term_without_date = preprocess_text(main_heading)
                        score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                        if score_without_date >= threshold:
                            matches.append((viaf_id, score_without_date))
            else:
                break
    
    except requests.RequestException as e:
        print(f"Error querying VIAF: {e}")
    
    # Usuwanie duplikatów
    unique_matches = list(set(matches))
    
    filtered_matches = [match for match in unique_matches if match[1] == 100]
    
    if filtered_matches:
        result_urls = [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches]
    elif unique_matches:
        best_match = max(unique_matches, key=lambda x: x[1])
        result_urls = [(f"https://viaf.org/viaf/{best_match[0]}", best_match[1])]
    else:
        result_urls = []

    return result_urls if result_urls else None


def preprocess_text(text):
    text = re.sub(r'\b\d{4}-\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\(\d{4}-\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_main_headings(record_data):
    main_headings = []
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        if isinstance(main_headings_data.get('data'), list):
            main_headings.extend(heading.get('text') for heading in main_headings_data['data'] if heading.get('text'))
        elif isinstance(main_headings_data.get('data'), dict):
            main_headings.append(main_headings_data['data'].get('text'))
    return main_headings


#%% check prosciej
def check_viaf_with_fuzzy_match2(entity_name, threshold=87, max_pages=3, entity_type=None):
    base_url_search = "https://viaf.org/viaf/search"
    matches = []

    def search_viaf(query):
        try:
            for page in range(1, max_pages + 1):
                query_params = {
                    'query': query,
                    'maximumRecords': 10,
                    'startRecord': (page - 1) * 10 + 1,
                    'httpAccept': 'application/json'
                }
                url = f"{base_url_search}?{urlencode(query_params)}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                    for record in data['searchRetrieveResponse']['records']:
                        record_data = record['record'].get('recordData', {})
                        viaf_id = record_data.get('viafID')

                        main_headings_texts = extract_text_from_main_headings(record_data)
                        
                        # Filtracja nagłówków zaczynających się na 4xx
                        x400s = record_data.get('x400s', {}).get('x400', [])
                        for x400 in x400s:
                            if isinstance(x400, dict):
                                tag = x400.get('datafield', {}).get('@tag')
                                if tag and tag.startswith('4'):
                                    continue
                                
                                subfields = x400.get('datafield', {}).get('subfield', [])
                                if isinstance(subfields, dict) and subfields.get('@code') == 'a':
                                    main_headings_texts.append(subfields.get('#text'))
                                elif isinstance(subfields, list):
                                    for subfield in subfields:
                                        if subfield.get('@code') == 'a':
                                            main_headings_texts.append(subfield.get('#text'))
                        
                        # Dopasowanie z użyciem fuzzy matching
                        for main_heading in main_headings_texts:
                            score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                            if score_with_date >= threshold:
                                matches.append((viaf_id, score_with_date))
                            
                            term_without_date = preprocess_text(main_heading)
                            score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                            if score_without_date >= threshold:
                                matches.append((viaf_id, score_without_date))
                else:
                    break
        except requests.RequestException as e:
            print(f"Error querying VIAF search: {e}")

    # Wyszukiwanie z encją
    if entity_type:
        query = f'local.{entity_type} all "{entity_name}"'
        search_viaf(query)
    if not entity_type:


    # AutoSuggest z filtrowaniem tagów 4xx
    
         base_url_suggest = "http://viaf.org/viaf/AutoSuggest"
         query_params = {'query': entity_name}
         try:
            response = requests.get(base_url_suggest, params=query_params)
            response.raise_for_status()
            data = response.json()
            
            if data and data.get('result') is not None:
                for result in data['result'][:10]:
                    original_term = result.get('term')
                    viaf_id = result.get('viafid')
                    
                    record_data = requests.get(f"https://viaf.org/viaf/{viaf_id}/viaf.json").json()
                    main_headings_texts = extract_text_from_main_headings(record_data)
                    
                    # Pomijamy tagi 4xx również w wynikach AutoSuggest
                    x400s = record_data.get('x400s', {}).get('x400', [])
                    for x400 in x400s:
                        if isinstance(x400, dict):
                            tag = x400.get('datafield', {}).get('@tag')
                            if tag and tag.startswith('4'):
                                continue
                            
                            subfields = x400.get('datafield', {}).get('subfield', [])
                            if isinstance(subfields, dict) and subfields.get('@code') == 'a':
                                main_headings_texts.append(subfields.get('#text'))
                            elif isinstance(subfields, list):
                                for subfield in subfields:
                                    if subfield.get('@code') == 'a':
                                        main_headings_texts.append(subfield.get('#text'))
                    
                    # Dopasowanie
                    for main_heading in main_headings_texts:
                        score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                        if score_with_date >= threshold:
                            matches.append((viaf_id, score_with_date))
                        
                        term_without_date = preprocess_text(main_heading)
                        score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                        if score_without_date >= threshold:
                            matches.append((viaf_id, score_without_date))
         except requests.RequestException as e:
            print(f"Error querying VIAF AutoSuggest: {e}")
         if not matches:
             query = f'"{entity_name}"'
             search_viaf(query)
    unique_matches = list(set(matches))
    filtered_matches = [match for match in unique_matches if match[1] == 100]
    
    return [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches] if filtered_matches else None
def check_viaf_with_fuzzy_match2(entity_name, threshold=84, max_pages=3, entity_type=None):
    base_url_search = "https://viaf.org/viaf/search"
    matches = []

    def search_viaf(query):
        try:
            for page in range(1, max_pages + 1):
                query_params = {
                    'query': query,
                    'maximumRecords': 10,
                    'startRecord': (page - 1) * 10 + 1,
                    'httpAccept': 'application/json'
                }
                url = f"{base_url_search}?{urlencode(query_params)}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                    for record in data['searchRetrieveResponse']['records']:
                        record_data = record['record'].get('recordData', {})
                        viaf_id = record_data.get('viafID')

                        main_headings_texts = extract_text_from_main_headings(record_data)
                        
                        # Filtracja nagłówków zaczynających się na 4xx
                        x400s = record_data.get('x400s', {}).get('x400', [])
                        for x400 in x400s:
                            if isinstance(x400, dict):
                                tag = x400.get('datafield', {}).get('@tag')
                                if tag and tag.startswith('4'):
                                    continue
                                
                                subfields = x400.get('datafield', {}).get('subfield', [])
                                if isinstance(subfields, dict) and subfields.get('@code') == 'a':
                                    main_headings_texts.append(subfields.get('#text'))
                                elif isinstance(subfields, list):
                                    for subfield in subfields:
                                        if subfield.get('@code') == 'a':
                                            main_headings_texts.append(subfield.get('#text'))
                        
                        # Dopasowanie z użyciem fuzzy matching
                        for main_heading in main_headings_texts:
                            score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                            if score_with_date >= threshold:
                                matches.append((viaf_id, score_with_date))
                            
                            term_without_date = preprocess_text(main_heading)
                            score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                            if score_without_date >= threshold:
                                matches.append((viaf_id, score_without_date))
                else:
                    break
        except requests.RequestException as e:
            print(f"Error querying VIAF search: {e}")

    # Wyszukiwanie z encją
    if entity_type:
        query = f'local.{entity_type} all "{entity_name}"'
        search_viaf(query)
    if not entity_type:
        # AutoSuggest z filtrowaniem tagów 4xx
        base_url_suggest = "http://viaf.org/viaf/AutoSuggest"
        query_params = {'query': entity_name}
        try:
            response = requests.get(base_url_suggest, params=query_params)
            response.raise_for_status()
            data = response.json()
            
            if data and data.get('result') is not None:
                for result in data['result'][:10]:
                    original_term = result.get('term')
                    viaf_id = result.get('viafid')
                    
                    record_data = requests.get(f"https://viaf.org/viaf/{viaf_id}/viaf.json").json()
                    main_headings_texts = extract_text_from_main_headings(record_data)
                    
                    # Pomijamy tagi 4xx również w wynikach AutoSuggest
                    x400s = record_data.get('x400s', {}).get('x400', [])
                    for x400 in x400s:
                        if isinstance(x400, dict):
                            tag = x400.get('datafield', {}).get('@tag')
                            if tag and tag.startswith('4'):
                                continue
                            
                            subfields = x400.get('datafield', {}).get('subfield', [])
                            if isinstance(subfields, dict) and subfields.get('@code') == 'a':
                                main_headings_texts.append(subfields.get('#text'))
                            elif isinstance(subfields, list):
                                for subfield in subfields:
                                    if subfield.get('@code') == 'a':
                                        main_headings_texts.append(subfield.get('#text'))
                    
                    # Dopasowanie
                    for main_heading in main_headings_texts:
                        score_with_date = fuzz.token_sort_ratio(entity_name, main_heading)
                        if score_with_date >= threshold:
                            matches.append((viaf_id, score_with_date))
                        
                        term_without_date = preprocess_text(main_heading)
                        score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                        if score_without_date >= threshold:
                            matches.append((viaf_id, score_without_date))
        except requests.RequestException as e:
            print(f"Error querying VIAF AutoSuggest: {e}")
        if not matches:
            query = f'"{entity_name}"'
            search_viaf(query)

    # Usuwanie duplikatów
    unique_matches = list(set(matches))

    # Filtrowanie wyników 100% dopasowania
    filtered_matches = [match for match in unique_matches if match[1] == 100]

    # Zwracanie wszystkich wyników z dopasowaniem 100%
    if filtered_matches:
        return [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches]
    
    # Jeśli brak wyników 100%, zwracamy najlepszy wynik
    if unique_matches:
        best_match = max(unique_matches, key=lambda x: x[1])
        return [(f"https://viaf.org/viaf/{best_match[0]}", best_match[1])]
    
    # Jeśli brak wyników
    return None