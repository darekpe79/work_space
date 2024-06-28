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


