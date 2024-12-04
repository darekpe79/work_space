# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:06:07 2024

@author: dariu
"""

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



def check_viaf_with_fuzzy_match2(entity_name, threshold=80, max_pages=5, entity_type='personalNames'):
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


results = check_viaf_with_fuzzy_match2(entity_name)
entity_name = 'Łódź'
results = check_viaf_with_fuzzy_match2(entity_name,entity_type='geographicNames' )

entity_name = 'Festiwal Filmowy w Cannes'
results = check_viaf_with_fuzzy_match2(entity_name,entity_type='corporateNames' )
#uniform title expression - tytul ujednolicony- warianty nazwy w innych krajach