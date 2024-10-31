# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:03:54 2024

@author: dariu
"""

import re
import requests
from fuzzywuzzy import fuzz
from urllib.parse import urlencode
#%%
# for index, row in tqdm(result_df[result_df['True/False'] == "True"].iterrows(),total=result_df[result_df['True/False'] == "True"].shape[0],desc="Processing Rows"):
#     text = row['combined_text']
#     autor = row['Autor']
#     viaf_autor = check_viaf_with_fuzzy_match2(autor,entity_type='personalNames')
#     tokens = tokenizer.tokenize(text)
#     max_tokens = 512  # Przykładowe ograniczenie modelu
#     token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
#     fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]
    
#     # Analiza każdego fragmentu osobno
#     ner_results = []
#     for fragment in fragments:
#         ner_results.extend(nlp1(fragment))
#     combined_entities = combine_tokens(ner_results)
    
#     combined_entities_selected = [entity for entity in combined_entities if entity['score'] >= 0.92]
#     entities = [(entity['word'], entity['type']) for entity in combined_entities_selected]
    
#     doc = nlp(text.lower())
#     lemmatized_text = " ".join([token.lemma_ for token in doc])
    
#     # Lematyzacja bytów i grupowanie
#     lemmatized_entities = []
#     entity_lemmatization_dict = {}
#     for entity in entities:
#         doc_entity = nlp(entity[0].lower())
#         lemmatized_entity = " ".join([token.lemma_ for token in doc_entity])
#         lemmatized_entities.append(lemmatized_entity)
#         if lemmatized_entity not in entity_lemmatization_dict:
#             entity_lemmatization_dict[lemmatized_entity] = {entity}
#         else:
#             entity_lemmatization_dict[lemmatized_entity].add(entity)
    
#     entity_groups = group_similar_entities(lemmatized_entities, threshold)
#     representatives = [sorted(group, key=lambda x: len(x))[0] for group in entity_groups]

#     entity_to_representative_map = {}
#     for group in entity_groups:
#         representative = sorted(group, key=lambda x: (len(x), x))[0]
#         for entity in group:
#             entity_to_representative_map[entity] = representative
    
#     updated_text = replace_entities_with_representatives(lemmatized_text, entity_to_representative_map)
#     list_of_new_entities = list(set(entity_to_representative_map.values()))
    
#     entity_counts = {entity: 0 for entity in list_of_new_entities}
#     title_end_pos = updated_text.find("< /tytuł >")
#     if title_end_pos == -1:
#         title_end_pos = updated_text.find("< /tytuł>")
    
#     for entity in list_of_new_entities:
#         total_occurrences = updated_text.count(entity)
#         entity_counts[entity] += total_occurrences
#         if updated_text.find(entity) < title_end_pos:
#             entity_counts[entity] += 50
    
#     sorted_entity_counts = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
#     choosen_ents = [ent for ent in sorted_entity_counts if ent[1] > 5]
    
#     # Dodawanie informacji o wybranym bycie do list
#     if choosen_ents:
#         first_entity_info = choosen_ents[0]
        
#         original_entities = entity_lemmatization_dict.get(first_entity_info[0], [])
        
#         result_df.at[index, 'Chosen_Entity'] = next(iter(original_entities))[0]
#         viaf_url, entity_type = None, "Not found"
#         if original_entities:
#             chosen_entity = next(iter(original_entities))
#             entity_name = chosen_entity[0]
#             entity_type_code = chosen_entity[1]
            
#             if entity_type_code == "PER":
#                 viaf_url = check_viaf_with_fuzzy_match2(entity_name, entity_type='personalNames')
#             elif entity_type_code == "LOC":
#                 viaf_url = check_viaf_with_fuzzy_match2(entity_name, entity_type='geographicNames')
#             elif entity_type_code == "ORG":
#                 viaf_url = check_viaf_with_fuzzy_match2(entity_name, entity_type='corporateNames')
#             else:
#                 viaf_url = check_viaf_with_fuzzy_match2(entity_name)
            
#             entity_type = entity_type_code
        
#         result_df.at[index, 'VIAF_URL'] = viaf_url[0][0] if viaf_url else "Not found"
#         result_df.at[index, 'Entity_Type'] = entity_type
#     else:
#         result_df.at[index, 'Chosen_Entity'] = pd.NA
#         result_df.at[index, 'VIAF_URL'] = "Not found"
#         result_df.at[index, 'Entity_Type'] = pd.NA
        
#     if viaf_autor:
#         result_df.at[index, 'Viaf_AUTHOR'] = ', '.join([ent[0] for ent in viaf_autor])
#     else:
#         result_df.at[index, 'Viaf_AUTHOR'] = "Not found"
        
#     # ----- Przetwarzanie za pomocą Nowego Modelu NER (PLAY, BOOK, EVENT) -----
#     max_tokens = 514  # Przykładowe ograniczenie modelu
#     tokens = tokenizer_new.tokenize(text)
#     token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
#     fragments = [tokenizer_new.convert_tokens_to_string(fragment) for fragment in token_fragments]
#     ner_results_model2 = []
#     for fragment in fragments:
#         ner_results_model2.extend(nlp_new(fragment))
    
    
#     # Filtracja encji z nowego modelu, biorąc tylko te z odpowiednich typów i powyżej progu
#     filtered_entities_model2 = [
#     (entity['word'], entity['entity_group'])
#     for entity in ner_results_model2
#     if entity['entity_group'] in ['PLAY', 'BOOK', 'EVENT'] and entity['score'] >= 0.80
# ]

#     # Unikanie duplikatów
#     unique_filtered_entities = []
#     seen_entities = set()
#     for entity_name, entity_type_code in filtered_entities_model2:
#         entity_key = (entity_name.lower(), entity_type_code)
#         if entity_key not in seen_entities:
#             unique_filtered_entities.append((entity_name, entity_type_code))
#             seen_entities.add(entity_key)
    
#     # Przypisywanie encji Modelu 2 do dynamicznych kolumn
#     for idx, (entity_name, entity_type_code) in enumerate(unique_filtered_entities, start=2):
#         # Sprawdzenie, czy osiągnięto limit bytów
#         if max_entities_new is not None and (idx - 1) > max_entities_new:
#             break  # Zakończ pętlę, jeśli osiągnięto limit
        
#         # Definicja nazw kolumn
#         byt_col = f"byt{idx}"
#         type_col = f"Type_{idx}"
#         viaf_col = f"VIAF_{idx}"
        
#         # Dodanie kolumn jeśli nie istnieją
#         for col in [byt_col, type_col, viaf_col]:
#             if col not in result_df.columns:
#                 result_df[col] = pd.NA
        
#         # Przypisanie nazwy bytu
#         result_df.at[index, byt_col] = entity_name
        
#         # Przypisanie typu bytu
#         result_df.at[index, type_col] = entity_type_code
        
#         # Sprawdzanie VIAF dla encji
#         viaf_result = None
#         # if entity_type_code == "PLAY":
#         #     viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='dramaticWorks')  # Sprawdzenie w sztukach dramatycznych
#         #     if not viaf_result:
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='artisticWorks')  # Sprawdzenie w pracach artystycznych
#         #     if not viaf_result:
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='uniformTitles')  # Sprawdzenie w uniform titles
#         #     if not viaf_result:
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='UniformTitleWork')
#         #     # if not viaf_result:
#         #     #     viaf_result = check_viaf_with_fuzzy_match2(entity_name)
#         # elif entity_type_code == "BOOK":
#         #     # Najpierw sprawdzanie w 'publications'
#         #     viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='UniformTitleWork')
#         #     # if not viaf_result:
#         #     #     # Jeśli brak wyniku, sprawdza w 'uniformTitles'
#         #     #     viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='uniformTitles')
#         #     if not viaf_result:
#         #         # Jeśli brak wyniku, sprawdza w 'UniformTitleExpression'
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='uniformTitleExpression')
#         #     if not viaf_result:
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='works')
#         #     if not viaf_result:
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='texts')
#         # elif entity_type_code == "EVENT":
#         #     viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='eventNames')
#         #     if not viaf_result:
#         #         # Jeśli brak wyniku, sprawdza w 'corporateNames' tylko dla 'EVENT'
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='corporateNames')
#         #     if not viaf_result:
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='conferenceNames')
#         #     if not viaf_result:
#         #         viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='performances')
#         if entity_type_code in ["PLAY","BOOK"]:
#             viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='uniformTitleWorks')
#             # if not viaf_result:
#             #     viaf_result = check_viaf_with_fuzzy_match2(entity_name)
 
#         elif entity_type_code == "EVENT":
#             viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='corporateNames')

#         # Ustalanie URL-a wyniku lub "Not found"
#         if viaf_result and len(viaf_result) > 0:
#             # Pobieramy wszystkie URL-e VIAF zwrócone przez funkcję
#             viaf_urls = [result[0] for result in viaf_result]
#             viaf_url = ', '.join(viaf_urls)  # Łączymy URL-e w jeden ciąg znaków
#         else:
#             viaf_url = "Not found"

    
        
#         # Przypisanie URL-a VIAF do odpowiedniej kolumny
#         result_df.at[index, viaf_col] = viaf_url
# # Zapisanie wyników do pliku Excel

#%%
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


def check_viaf_with_fuzzy_match2(entity_name, threshold=84, max_pages=1, entity_type='personalNames'):
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
                print(data)

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


entity_name = 'Tomasz Domagała'
results = check_viaf_with_fuzzy_match2(entity_name)

entity_name = 'Polska'
results = check_viaf_with_fuzzy_match2(entity_name,entity_type='geographicNames' )

entity_name = 'Solaris'
results = check_viaf_with_fuzzy_match2(entity_name, entity_type='UniformTitleWork')
entity_name = 'Solaris'
results = check_viaf_with_fuzzy_match2(entity_name)  # Bez użycia entity_type, aby poszukać we wszystkich kategoriach
print(results)



import re
import requests
from fuzzywuzzy import fuzz
from urllib.parse import urlencode

def preprocess_text(text):
    # Usuwanie dat, zbędnych spacji, i innych znaków
    text = re.sub(r'\b\d{4}-\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\(\d{4}-\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    
    # Normalizacja tekstu: usunięcie akcentów i innych znaków specjalnych
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()  # Zwracamy tekst w małych literach

def extract_text_from_main_headings(record_data):
    main_headings = []
    
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        
        if isinstance(main_headings_data.get('data'), list):
            for heading in main_headings_data['data']:
                if isinstance(heading, dict) and 'text' in heading:
                    main_headings.append(heading['text'])
        elif isinstance(main_headings_data.get('data'), dict):
            text = main_headings_data['data'].get('text')
            if text:
                main_headings.append(text)
    
    return main_headings

def check_viaf_with_fuzzy_match2(entity_name, threshold=84, max_pages=5, entity_type=None):
    base_url_search = "https://viaf.org/viaf/search"
    matches = []  # Lista do przechowywania wyników ze wszystkich stron

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
                print(url)
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                print("Response JSON:", data)

                if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
                    for record in data['searchRetrieveResponse']['records']:
                        record_data = record['record'].get('recordData', {})
                        viaf_id = record_data.get('viafID')
                        main_headings_texts = extract_text_from_main_headings(record_data)
                        
                        # Dopasowanie z użyciem fuzzy matching
                        for main_heading in main_headings_texts:
                            score_with_date = fuzz.token_sort_ratio(entity_name.lower(), main_heading.lower())
                            if score_with_date >= threshold:
                                matches.append((viaf_id, score_with_date))
                            
                            term_without_date = preprocess_text(main_heading)
                            score_without_date = fuzz.token_sort_ratio(entity_name.lower(), term_without_date)
                            if score_without_date >= threshold:
                                matches.append((viaf_id, score_without_date))
                else:
                    break
        except requests.RequestException as e:
            print(f"Error querying VIAF: {e}")

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
                    
                    # Dopasowanie
                    for main_heading in main_headings_texts:
                        score_with_date = fuzz.token_sort_ratio(entity_name.lower(), main_heading.lower())
                        if score_with_date >= threshold:
                            matches.append((viaf_id, score_with_date))
                        
                        term_without_date = preprocess_text(main_heading)
                        score_without_date = fuzz.token_sort_ratio(entity_name.lower(), term_without_date)
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


# Przykładowe wywołanie
entity_name = 'Warszawa'
results = check_viaf_with_fuzzy_match2(entity_name,entity_type='geographicNames' )
print(results)

import requests

# Ustawienia zapytania do API VIAF
base_url_search = "http://www.viaf.org/viaf/search"
query = "Solaris"
params = {
    'query': query,
    'maximumRecords': 10,  # Maksymalna liczba rekordów na stronę
    'httpAccept': 'application/json'  # Akceptowanie odpowiedzi JSON
}

# Wysyłamy zapytanie HTTP GET do VIAF
response = requests.get(base_url_search, params=params)
response.raise_for_status()  # Sprawdzenie, czy zapytanie zakończyło się sukcesem

# Parsowanie odpowiedzi jako JSON
json_response = response.json()

# Parsowanie wyników wyszukiwania i szukanie tytułu (kod 't')
for record in json_response['searchRetrieveResponse']['records']:
    if 'recordData' in record['record']:
        # Pobranie VIAF ID
        viaf_id = record['record']['recordData'].get('viafID', 'Brak VIAF ID')

        mainHeadings = record['record']['recordData'].get('mainHeadings', {})
        mainHeadingEl = mainHeadings.get('mainHeadingEl', {})
        
        if 'datafield' in mainHeadingEl:
            subfields = mainHeadingEl['datafield'].get('subfield', [])
            
            # Przetwarzanie wszystkich subfieldów w rekordzie
            for subfield in subfields:
                if subfield['@code'] == 't':  # Szukamy kodu 't' dla tytułu
                    print(f"Tytuł: {subfield['#text']}, VIAF ID: {viaf_id}")
import requests

# Ustawienia zapytania do API VIAF
base_url_search = "http://www.viaf.org/viaf/search"
query = "Solaris"
params = {
    'query': query,
    'maximumRecords': 10,  # Maksymalna liczba rekordów na stronę
    'httpAccept': 'application/json'  # Akceptowanie odpowiedzi JSON
}

# Wysyłamy zapytanie HTTP GET do VIAF
response = requests.get(base_url_search, params=params)
response.raise_for_status()  # Sprawdzenie, czy zapytanie zakończyło się sukcesem

# Parsowanie odpowiedzi jako JSON
json_response = response.json()

# Parsowanie wyników wyszukiwania i szukanie tytułu (kod 't')
for record in json_response['searchRetrieveResponse']['records']:
    if 'recordData' in record['record']:
        # Pobranie VIAF ID
        viaf_id = record['record']['recordData'].get('viafID', 'Brak VIAF ID')
        
        # Pobranie entity type (nameType)
        entity_type = record['record']['recordData'].get('nameType', 'Brak typu encji')

        mainHeadings = record['record']['recordData'].get('mainHeadings', {})
        mainHeadingEl = mainHeadings.get('mainHeadingEl', {})
        
        if 'datafield' in mainHeadingEl:
            subfields = mainHeadingEl['datafield'].get('subfield', [])
            
            # Przetwarzanie wszystkich subfieldów w rekordzie
            for subfield in subfields:
                if subfield['@code'] == 't':  # Szukamy kodu 't' dla tytułu
                    print(f"Tytuł: {subfield['#text']}, VIAF ID: {viaf_id}, Typ encji: {entity_type}")
import requests

# Ustawienia wyszukiwania
base_url_search = "http://www.viaf.org/viaf/search"
entity_name = "Solaris"  # Nazwa encji, którą chcemy wyszukać

# Tworzenie zapytania (bez dodawania typu encji)
query = f'local.all "{entity_name}"'

params = {
    'query': query,
    'maximumRecords': 10,  # Maksymalna liczba rekordów na stronę
    'httpAccept': 'application/json'  # Akceptowanie odpowiedzi JSON
}

# Wysyłamy zapytanie HTTP GET do VIAF
response = requests.get(base_url_search, params=params)
response.raise_for_status()  # Sprawdzenie, czy zapytanie zakończyło się sukcesem

# Parsowanie odpowiedzi jako JSON
json_response = response.json()

# Parsowanie wyników wyszukiwania i filtrowanie na podstawie nameType
for record in json_response['searchRetrieveResponse']['records']:
    if 'recordData' in record['record']:
        # Pobranie VIAF ID
        viaf_id = record['record']['recordData'].get('viafID', 'Brak VIAF ID')
        
        # Pobranie entity type (nameType)
        entity_type = record['record']['recordData'].get('nameType', 'Brak typu encji')
        
        # Filtrowanie wyników po "UniformTitleExpression"
        if entity_type == "UniformTitleExpression":
            mainHeadings = record['record']['recordData'].get('mainHeadings', {})
            mainHeadingEl = mainHeadings.get('mainHeadingEl', {})
            
            if 'datafield' in mainHeadingEl:
                subfields = mainHeadingEl['datafield'].get('subfield', [])
                
                # Przetwarzanie wszystkich subfieldów w rekordzie
                for subfield in subfields:
                    if subfield['@code'] == 't':  # Szukamy kodu 't' dla tytułu
                        print(f"Tytuł: {subfield['#text']}, VIAF ID: {viaf_id}, Typ encji: {entity_type}")
import requests

# Ustawienia wyszukiwania
base_url_search = "http://www.viaf.org/viaf/search"
query = "Solaris"
entity_type_filter = "UniformTitleExpression"  # Typ encji, po którym chcemy filtrować

params = {
    'query': query,
    'maximumRecords': 10,  # Maksymalna liczba rekordów na stronę
    'httpAccept': 'application/json'  # Akceptowanie odpowiedzi JSON
}

# Wysyłamy zapytanie HTTP GET do VIAF
response = requests.get(base_url_search, params=params)
response.raise_for_status()  # Sprawdzenie, czy zapytanie zakończyło się sukcesem

# Parsowanie odpowiedzi jako JSON
json_response = response.json()

# Parsowanie wyników wyszukiwania i szukanie tytułu (kod 't')
for record in json_response['searchRetrieveResponse']['records']:
    if 'recordData' in record['record']:
        # Pobranie VIAF ID
        viaf_id = record['record']['recordData'].get('viafID', 'Brak VIAF ID')
        
        # Pobranie entity type (nameType)
        entity_type = record['record']['recordData'].get('nameType', 'Brak typu encji')
        
        # Filtrowanie według typu encji
        if entity_type == entity_type_filter:
            mainHeadings = record['record']['recordData'].get('mainHeadings', {})
            mainHeadingEl = mainHeadings.get('mainHeadingEl', {})
            
            if 'datafield' in mainHeadingEl:
                subfields = mainHeadingEl['datafield'].get('subfield', [])
                
                # Przetwarzanie wszystkich subfieldów w rekordzie
                for subfield in subfields:
                    if subfield['@code'] == 't':  # Szukamy kodu 't' dla tytułu
                        print(f"Tytuł: {subfield['#text']}, VIAF ID: {viaf_id}, Typ encji: {entity_type}")




base_url_search = "http://www.viaf.org/viaf/search"
query = "Łódź"
entity_type_filter = "UniformTitleExpression"  # Typ encji, po którym chcemy filtrować

params = {
    'query': query,
    'maximumRecords': 10,  # Maksymalna liczba rekordów na stronę
    'httpAccept': 'application/json'  # Akceptowanie odpowiedzi JSON
}

# Wysyłamy zapytanie HTTP GET do VIAF
response = requests.get(base_url_search, params=params)
response.raise_for_status()  # Sprawdzenie, czy zapytanie zakończyło się sukcesem

# Parsowanie odpowiedzi jako JSON
json_response = response.json()

# Parsowanie wyników wyszukiwania i szukanie tytułu (kod 't')
for record in json_response['searchRetrieveResponse']['records']:
    if 'recordData' in record['record']:
        print(record['record'])
        # Pobranie VIAF ID
        viaf_id = record['record']['recordData'].get('viafID', 'Brak VIAF ID')
        
        # Pobranie entity type (nameType)
        entity_type = record['record']['recordData'].get('nameType', 'Brak typu encji')
        
        # Filtrowanie według typu encji
        if entity_type == entity_type_filter:
            mainHeadings = record['record']['recordData'].get('mainHeadings', {})
            mainHeadingEl = mainHeadings.get('mainHeadingEl', {})
            
            if 'datafield' in mainHeadingEl:
                subfields = mainHeadingEl['datafield'].get('subfield', [])
                
                # Przetwarzanie wszystkich subfieldów w rekordzie
                for subfield in subfields:
                    if subfield['@code'] == 't':  # Szukamy kodu 't' dla tytułu
                        print(f"Tytuł: {subfield['#text']}, VIAF ID: {viaf_id}, Typ encji: {entity_type}")




import requests
import re
import json

# Parametry wyszukiwania
search_name = 'Warszawa'
search_type = 'geographicNames'  # Dla nazw geograficznych

# Budowanie URL z odpowiednimi parametrami
query = f'local.{search_type} all "{search_name}"'
encoded_query = re.sub('\s+', '%20', query)
url = f"http://viaf.org/viaf/search?query={encoded_query}&sortKeys=holdingscount&recordSchema=BriefVIAF&httpAccept=application/json"

# Wykonanie żądania HTTP GET
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("Główne klucze w danych:", list(data.keys()))
    print(json.dumps(data, ensure_ascii=False, indent=2))
    # Przetwarzanie danych JSON
    records = data.get('searchRetrieveResponse', {}).get('records', {})
    records[1]
    
    
    
    
    
    # Upewniamy się, że records jest listą
    if isinstance(records, dict):
        records = [records]
    # Iteracja po rekordach
    for record in records:
        record_data = record.get('recordData', {}).get('viaf', {})
        viaf_id = record_data.get('viafID', '')
        main_headings = record_data.get('mainHeadings', {}).get('data', [])
        if isinstance(main_headings, dict):
            main_headings = [main_headings]
        for heading in main_headings:
            text = heading.get('text', '')
            print(f"VIAF ID: {viaf_id}, Nazwa geograficzna: {text}")
else:
    print("Błąd w żądaniu:", response.status_code)
def extract_text_from_main_headings(record_data):
    main_headings = []
    
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        
        if isinstance(main_headings_data.get('data'), list):
            for heading in main_headings_data['data']:
                if isinstance(heading, dict) and 'text' in heading:
                    main_headings.append(heading['text'])
        elif isinstance(main_headings_data.get('data'), dict):
            text = main_headings_data['data'].get('text')
            if text:
                main_headings.append(text)
    
    return main_headings
base_url_search = "https://viaf.org/viaf/search"
entity_name = 'Warszawa'
entity_type = 'geographicNames' 
entity_name = 'Solaris'
entity_type = 'uniformTitleExpressions'
entity_name = 'Pan Tadeusz'
entity_type = 'uniformTitleExpressions'
entity_type = 'uniformTitleWorks'
query = f'local.{entity_type} all "{entity_name}"'
query_params = {
    'query': query,
    'maximumRecords': 100,
    'startRecord': (1 - 1) * 10 + 1,
    'httpAccept': 'application/json'
}
url = f"{base_url_search}?{urlencode(query_params)}"
response = requests.get(url)
response.raise_for_status()
data = response.json()


if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
    for record in data['searchRetrieveResponse']['records']:
        
        #print(record)
        record_data = record['record'].get('recordData', {})
        #print(record_data['titles']['work'])
        
        print(record_data['mainHeadings']['data']['text']) 
        viaf_id = record_data.get('viafID')
        print(viaf_id)
       #
       # # print(main_headings_texts)
        viaf_id = record['record']['recordData'].get('viafID', 'Brak VIAF ID')
        
       #  # Pobranie entity type (nameType)
       #  entity_type = record['record']['recordData'].get('nameType', 'Brak typu encji')
        
       #  # Filtrowanie według typu encji
    
        mainHeadings = record['record']['recordData'].get('mainHeadings', {})
        mainHeadingEl = mainHeadings.get('mainHeadingEl', {})
        
        
        
        
        
        
        
        
        if 'datafield' in mainHeadingEl:
            subfields = mainHeadingEl['datafield'].get('subfield', [])
            print (subfields)
            print(viaf_id)
            
            # Przetwarzanie wszystkich subfieldów w rekordzie
            for subfield in subfields:
                if subfield['@code'] == 't':  # Szukamy kodu 't' dla tytułu
                    print(f"Tytuł: {subfield['#text']}, VIAF ID: {viaf_id}, Typ encji: {entity_type}")


import requests
from urllib.parse import urlencode

# Definiowanie podstawowego URL dla wyszukiwania VIAF
base_url_search = 'http://viaf.org/viaf/search'
entity_type = 'uniformTitleExpressions'
# Ustawienia wyszukiwania
entity_name = 'Solaris'  # Wpisz tutaj szukany termin
desired_viaf_id = '311573672'  # Wpisz tutaj poszukiwany VIAF ID
entity_name = 'Pan Tadeusz'  # Wpisz tutaj szukany termin
desired_viaf_id = '316392742'  # Wpisz tutaj poszukiwany VIAF ID
entity_type = 'uniformTitleWorks'  # Typ encji
# Flaga informująca, czy znaleziono poszukiwany VIAF ID
found = False

# Iteracja po 100 stronach

for page in range(1, 101):
    
    # Obliczenie numeru pierwszego rekordu na aktualnej stronie
    start_record = (page - 1) * 10 + 1

    # Budowanie zapytania
    query = f'local.{entity_type} all "{entity_name}"'
    query_params = {
        'query': query,
        'maximumRecords': 10,  # Ilość rekordów na stronę
        'startRecord': start_record,
        'httpAccept': 'application/json'
    }
    url = f"{base_url_search}?{urlencode(query_params)}"

    # Wykonanie żądania
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Przetwarzanie odpowiedzi
    if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
        for record in data['searchRetrieveResponse']['records']:
            record_data = record['record'].get('recordData', {})
            # Sprawdzanie obecności kluczy
            main_headings = record_data.get('mainHeadings', {})
            main_headings_texts = extract_text_from_main_headings(record_data)
            mainHeadingEl = main_headings.get('mainHeadingEl', {})
            print(mainHeadingEl)
            viaf_id = record_data.get('viafID', 'Brak VIAF ID')
            print(main_headings_texts)
            print(viaf_id)
            if viaf_id == desired_viaf_id:
                print(f"\nZnaleziono poszukiwany VIAF ID: {viaf_id}")
                
                found = True
                break 
        if found:
            break
        
            
            mainHeadingEl = main_headings_texts.get('mainHeadingEl', {})
            
            data_field = main_headings.get('data', {})
            viaf_id = record_data.get('viafID', 'Brak VIAF ID')

            # Sprawdzanie typu data_field i pobieranie nazwy/nazw
            if isinstance(data_field, dict):
                name = data_field.get('text', 'Brak nazwy')
                print(f"Sprawdzam VIAF ID: {viaf_id}, Nazwa: {name}")

                if viaf_id == desired_viaf_id:
                    print(f"\nZnaleziono poszukiwany VIAF ID: {viaf_id}")
                    print(f"Nazwa: {name}")
                    found = True
                    break  # Przerwanie pętli for po znalezieniu VIAF ID
            elif isinstance(data_field, list):
                for item in data_field:
                    name = item.get('text', 'Brak nazwy')
                    print(f"Sprawdzam VIAF ID: {viaf_id}, Nazwa: {name}")

                    if viaf_id == desired_viaf_id:
                        print(f"\nZnaleziono poszukiwany VIAF ID: {viaf_id}")
                        print(f"Nazwa: {name}")
                        found = True
                        break  # Przerwanie pętli for po znalezieniu VIAF ID
                if found:
                    break  # Przerwanie pętli wewnętrznej po znalezieniu VIAF ID
            else:
                print(f"Nieoczekiwany typ data_field: {type(data_field)}")

        if found:
            break  # Przerwanie pętli po stronach po znalezieniu VIAF ID
    else:
        print(f"Brak rekordów na stronie {page}")
        # Opcjonalnie: przerwanie pętli, jeśli nie ma więcej rekordów
        # break

if not found:
    print("Nie znaleziono poszukiwanego VIAF ID.")
import requests
from urllib.parse import urlencode

# Definiowanie podstawowego URL dla wyszukiwania VIAF
base_url_search = 'http://viaf.org/viaf/search'

# Ustawienia wyszukiwania
entity_name = 'Solaris' # Wpisz tutaj szukany termin
entity_type = 'uniformTitleWorks'  # Typ encji
desired_viaf_id = '311573672'  # Wpisz tutaj poszukiwany VIAF ID

# Flaga informująca, czy znaleziono poszukiwany VIAF ID
found = False

for page in range(1, 101):
    # Calculate the first record number on the current page
    start_record = (page - 1) * 10 + 1

    # Build the query
    query = f'local.{entity_type} all "{entity_name}"'
    query_params = {
        'query': query,
        'maximumRecords': 10,  # Number of records per page
        'startRecord': start_record,
        'httpAccept': 'application/json'
    }
    url = f"{base_url_search}?{urlencode(query_params)}"

    # Make the request
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Process the response
    if 'searchRetrieveResponse' in data and 'records' in data['searchRetrieveResponse']:
        for record in data['searchRetrieveResponse']['records']:
            record_data = record['record'].get('recordData', {})
            viaf_id = record_data.get('viafID', 'Brak VIAF ID')

            # Check for 'mainHeadings' presence
            main_headings = record_data.get('mainHeadings', {})

            # Get 'mainHeadingEl', ensure it's a list
            mainHeadingEl = main_headings.get('mainHeadingEl', {})
            if isinstance(mainHeadingEl, dict):
                mainHeadingEl = [mainHeadingEl]

            # Iterate over elements of 'mainHeadingEl'
            for heading_el in mainHeadingEl:
                datafield = heading_el.get('datafield', {})
                subfields = datafield.get('subfield', [])

                # Ensure 'subfields' is a list
                if isinstance(subfields, dict):
                    subfields = [subfields]

                # Iterate over 'subfields' and extract those with '@code' equal to 't'
                for subfield in subfields:
                    if subfield.get('@code') == 't':
                        title = subfield.get('#text', 'Brak tytułu')
                        print(f"Tytuł: {title}")
                        print(f"VIAF ID: {viaf_id}")

                        # Check if this is the desired title or VIAF ID
                        if viaf_id == desired_viaf_id:
                            print(f"\nZnaleziono poszukiwany VIAF ID: {viaf_id}")
                            found = True
                            break  # Break out of subfields loop

                if found:
                    break  # Break out of mainHeadingEl loop

            if found:
                break  # Break out of records loop

        if found:
            break  # Break out of pages loop
    else:
        print(f"Brak rekordów na stronie {page}")
        # Optionally: break the loop if there are no more records
        # break

if not found:
    print("Nie znaleziono poszukiwanego VIAF ID.")
            
            
        
import requests
from urllib.parse import urlencode
from fuzzywuzzy import fuzz
import re

def preprocess_text(text):
    text = re.sub(r'\b\d{4}-\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}\b', '', text)
    text = re.sub(r'\(\d{4}-\d{4}\)', '', text)
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_main_headings(record_data):
    main_headings = []
    
    # Check if 'mainHeadings' exists
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        
        # If 'data' is a list
        if isinstance(main_headings_data.get('data'), list):
            for heading in main_headings_data['data']:
                if isinstance(heading, dict) and 'text' in heading:
                    main_headings.append(heading['text'])
        
        # If 'data' is a dict
        elif isinstance(main_headings_data.get('data'), dict):
            text = main_headings_data['data'].get('text')
            if text:
                main_headings.append(text)
        
        # Other potential cases
        else:
            # Check if 'data' is a string
            if 'data' in main_headings_data and isinstance(main_headings_data['data'], str):
                main_headings.append(main_headings_data['data'])
    
    return main_headings



def check_viaf_with_fuzzy_match2(entity_name, threshold=70, max_pages=20, entity_type='personalNames'):
    base_url_search = "https://viaf.org/viaf/search"
    matches = []

    # Ensure 'entity_name' is a string
    if not isinstance(entity_name, str):
        entity_name = str(entity_name)

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

                        # Different parsing based on 'entity_type'
                        if entity_type=='uniformTitleWorks':
                            main_headings_texts = []

                            # Check for 'mainHeadings'
                            main_headings = record_data.get('mainHeadings', {})

                            # Get 'mainHeadingEl', ensure it's a list
                            mainHeadingEl = main_headings.get('mainHeadingEl', {})
                            if isinstance(mainHeadingEl, dict):
                                mainHeadingEl = [mainHeadingEl]
                            elif not isinstance(mainHeadingEl, list):
                                mainHeadingEl = []

                            # Iterate over elements of 'mainHeadingEl'
                            for heading_el in mainHeadingEl:
                                datafield = heading_el.get('datafield', {})
                                subfields = datafield.get('subfield', [])

                                # Ensure 'subfields' is a list
                                if isinstance(subfields, dict):
                                    subfields = [subfields]
                                elif not isinstance(subfields, list):
                                    subfields = []

                                # Check if any subfields have codes other than 't' or 'a'
                                skip_record = False
                                for subfield in subfields:
                                    code = subfield.get('@code', '')
                                    if code not in ['t', 'a']:
                                        skip_record = True
                                        break  # No need to check further subfields

                                if skip_record:
                                    # Skip this heading_el and continue with the next one
                                    continue

                                # Iterate over 'subfields' and extract those with '@code' equal to 't'
                                for subfield in subfields:
                                    if subfield.get('@code') == 't':
                                        title = subfield.get('#text', 'Brak tytułu')
                                        if title:
                                            title = str(title)  # Ensure 'title' is a string
                                            main_headings_texts.append(title)
                        elif entity_type == 'uniformTitleExpressions':
                            main_headings_texts = []
                        
                            # Pobieramy 'mainHeadings' z danych rekordu
                            main_headings = record_data.get('mainHeadings', {})
                        
                            # Pobieramy 'mainHeadingEl' i upewniamy się, że jest to lista
                            mainHeadingEl = main_headings.get('mainHeadingEl', {})
                            if isinstance(mainHeadingEl, dict):
                                mainHeadingEl = [mainHeadingEl]
                            elif not isinstance(mainHeadingEl, list):
                                mainHeadingEl = []
                        
                            # Iterujemy po elementach 'mainHeadingEl'
                            for heading_el in mainHeadingEl:
                                datafield = heading_el.get('datafield', {})
                                subfields = datafield.get('subfield', [])
                        
                                # Upewniamy się, że 'subfields' jest listą
                                if isinstance(subfields, dict):
                                    subfields = [subfields]
                                elif not isinstance(subfields, list):
                                    subfields = []
                        
                                # **Zbieramy tekst tylko z podpola 't'**
                                for subfield in subfields:
                                    code = subfield.get('@code', '')
                                    if code == 't':
                                        text = subfield.get('#text', '')
                                        if text:
                                            text = str(text)  # Upewniamy się, że 'text' jest łańcuchem znaków
                                            main_headings_texts.append(text)

                        elif entity_type == 'geographicNames':
                            main_headings_texts = []

                            # Check for 'mainHeadings'
                            main_headings = record_data.get('mainHeadings', {})

                            # Get 'mainHeadingEl', ensure it's a list
                            mainHeadingEl = main_headings.get('mainHeadingEl', {})
                            if isinstance(mainHeadingEl, dict):
                                mainHeadingEl = [mainHeadingEl]
                            elif not isinstance(mainHeadingEl, list):
                                mainHeadingEl = []

                            # Iterate over elements of 'mainHeadingEl'
                            for heading_el in mainHeadingEl:
                                datafield = heading_el.get('datafield', {})
                                # Check if the datafield has tag '151'
                                if datafield.get('@tag') not in ['151', '110']:
                                    continue  # Skip if tag is not '151'

                                subfields = datafield.get('subfield', [])

                                # Ensure 'subfields' is a list
                                if isinstance(subfields, dict):
                                    subfields = [subfields]
                                elif not isinstance(subfields, list):
                                    subfields = []

                                # Check if any subfields have codes other than 'a'
                                skip_record = False
                                for subfield in subfields:
                                    code = subfield.get('@code', '')
                                    if code != 'a':
                                        skip_record = True
                                        break  # No need to check further subfields

                                if skip_record:
                                    # Skip this heading_el and continue with the next one
                                    continue

                                # Iterate over 'subfields' and extract those with '@code' equal to 'a'
                                for subfield in subfields:
                                    if subfield.get('@code') == 'a':
                                        name = subfield.get('#text', 'Brak nazwy')
                                        if name:
                                            name = str(name)  # Ensure 'name' is a string
                                            main_headings_texts.append(name)

                        elif entity_type == 'corporateNames':
                            main_headings_texts = []

                            # Check for 'mainHeadings'
                            main_headings = record_data.get('mainHeadings', {})

                            # Get 'mainHeadingEl', ensure it's a list
                            mainHeadingEl = main_headings.get('mainHeadingEl', {})
                            if isinstance(mainHeadingEl, dict):
                                mainHeadingEl = [mainHeadingEl]
                            elif not isinstance(mainHeadingEl, list):
                                mainHeadingEl = []

                            # Iterate over elements of 'mainHeadingEl'
                            for heading_el in mainHeadingEl:
                                datafield = heading_el.get('datafield', {})
                                # Check if the datafield has tag '111'
                                if datafield.get('@tag') not in ['111', '110']:
                                    continue

                                subfields = datafield.get('subfield', [])

                                # Ensure 'subfields' is a list
                                if isinstance(subfields, dict):
                                    subfields = [subfields]
                                elif not isinstance(subfields, list):
                                    subfields = []

                                # Check if any subfields have codes other than 'a'
                                skip_record = False
                                for subfield in subfields:
                                    code = subfield.get('@code', '')
                                    if code != 'a':
                                        skip_record = True
                                        break  # No need to check further subfields

                                if skip_record:
                                    # Skip this heading_el and continue with the next one
                                    continue

                                # Iterate over 'subfields' and extract those with '@code' equal to 'a'
                                for subfield in subfields:
                                    if subfield.get('@code') == 'a':
                                        name = subfield.get('#text', 'Brak nazwy')
                                        if name:
                                            name = str(name)  # Ensure 'name' is a string
                                            main_headings_texts.append(name)
                        else:
                            # Use the existing function for other entity types
                            main_headings_texts = extract_text_from_main_headings(record_data)

                        # Perform fuzzy matching
                        for main_heading in main_headings_texts:
                            if not isinstance(main_heading, str):
                                main_heading = str(main_heading)

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

    # Search with entity type
    if entity_type:
        query = f'local.{entity_type} all "{entity_name}"'
        search_viaf(query)
    else:
        # Code for the case without a specified 'entity_type'
        pass  # You can place your code here or leave it empty

    # Remove duplicates
    unique_matches = list(set(matches))

    # Filter results with 100% match
    filtered_matches = [match for match in unique_matches if match[1] == 100]

    # Return all results with 100% match
    if filtered_matches:
        return [(f"https://viaf.org/viaf/{match[0]}", match[1]) for match in filtered_matches]

    # If no 100% matches, return the best match
    if unique_matches:
        best_match = max(unique_matches, key=lambda x: x[1])
        return [(f"https://viaf.org/viaf/{best_match[0]}", best_match[1])]

    # If no matches
    return None
entity_name = 'Tomasz Domagała'
entity_name = 'Les Émigrants'
entity_type = 'uniformTitleWorks' 
entity_type ='uniformTitleExpressions'


results = check_viaf_with_fuzzy_match2(entity_name, entity_type=entity_type)
entity_name = 'Łódź'
results = check_viaf_with_fuzzy_match2(entity_name,entity_type='geographicNames' )

entity_name = 'Festiwal Filmowy w Cannes'
results = check_viaf_with_fuzzy_match2(entity_name,entity_type='corporateNames' )
#uniform title expression - tytul ujednolicony- warianty nazwy w innych krajach
