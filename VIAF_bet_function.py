# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:03:54 2024

@author: dariu
"""

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
def extract_text_from_main_headings(record_data):
    main_headings = []
    
    # Sprawdzanie, czy 'mainHeadings' istnieje
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        
        # Obsługa przypadku, gdy dane są listą
        if isinstance(main_headings_data.get('data'), list):
            for heading in main_headings_data['data']:
                if isinstance(heading, dict) and 'text' in heading:
                    main_headings.append(heading['text'])
        
        # Obsługa przypadku, gdy dane są słownikiem
        elif isinstance(main_headings_data.get('data'), dict):
            text = main_headings_data['data'].get('text')
            if text:
                main_headings.append(text)
        
        # Obsługa innych potencjalnych przypadków
        else:
            # Sprawdzenie, czy w mainHeadings_data jest coś, co można przetworzyć
            if 'data' in main_headings_data and isinstance(main_headings_data['data'], str):
                main_headings.append(main_headings_data['data'])
    
    return main_headings


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

entity_name = 'Opera rara'
results = check_viaf_with_fuzzy_match2(entity_name)


entity_name = 'Domagała, Tomasz'
results = check_viaf_with_fuzzy_match2(entity_name,entity_type='personalNames')
