# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:14:40 2024

@author: dariu
"""

import pandas as pd
import json
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import textwrap
import networkx as nx
from spacy.lang.pl import Polish
import spacy
from collections import defaultdict
from fuzzywuzzy import fuzz
from collections import defaultdict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModel, HerbertTokenizerFast
import joblib
from tqdm import tqdm
import numpy as np
from urllib.parse import urlencode


def load_and_merge_data(json_file_path, excel_file_path, common_column='Link'):
    # Spróbuj wczytać dane z pliku JSON z kodowaniem UTF-8, a jeśli się nie uda, użyj kodowania 'latin-1'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_content = file.read()
            if not json_content.strip():
                raise ValueError("Plik JSON jest pusty.")
            json_data = json.loads(json_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        try:
            with open(json_file_path, 'r', encoding='latin-1') as file:
                json_content = file.read()
                if not json_content.strip():
                    raise ValueError("Plik JSON jest pusty.")
                json_data = json.loads(json_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Nie udało się odczytać pliku JSON: {e}")
    
    df_json = pd.DataFrame(json_data)
    
    # Sprawdzenie, czy kolumny 'Link' i 'Tekst artykułu' istnieją w pliku JSON
    if 'Link' not in df_json.columns or 'Tekst artykułu' not in df_json.columns:
        raise ValueError("Brak wymaganych kolumn w pliku JSON.")
    
    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']].astype(str)
    
    # Wczytanie danych z pliku Excel
    df_excel = pd.read_excel(excel_file_path)
    
    # Połączenie DataFrame'ów, zachowując wszystkie wiersze i kolumny z pliku Excel oraz pełny tekst z JSONa
    merged_df = pd.merge(df_excel, df_json, on=common_column, how="left")
    
    # Usunięcie pustych wierszy
    merged_df.dropna(subset=['Link', 'Tekst artykułu'], inplace=True)
    
    return merged_df

# Przykładowe użycie
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/domagala2024-02-08.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/domagala_2024-02-08.xlsx'
df = load_and_merge_data(json_file_path, excel_file_path)
df = df[df['Tytuł artykułu'].apply(lambda x: isinstance(x, str))]
df = df[df['Tekst artykułu'].apply(lambda x: isinstance(x, str))]

model_path = "C:/Users/dariu/model_5epoch_gatunek_large"
model_genre = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = HerbertTokenizerFast.from_pretrained(model_path)

# W późniejszym czasie, aby wczytać LabelEncoder:
label_encoder = joblib.load('C:/Users/dariu/model_5epoch_gatunek_large/label_encoder_gatunek5.joblib')
# TRUE FALSE
model_path = "C:/Users/dariu/model_TRUE_FALSE_4epoch_base_514_tokens/"
model_t_f = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer_t_f =  HerbertTokenizerFast.from_pretrained(model_path)

label_encoder_t_f = joblib.load('C:/Users/dariu/model_TRUE_FALSE_4epoch_base_514_tokens/label_encoder_true_false4epoch_514_tokens.joblib')

model_path_hasla = "model_hasla_8epoch_base"
model_hasla = AutoModelForSequenceClassification.from_pretrained(model_path_hasla)
tokenizer_hasla = HerbertTokenizerFast.from_pretrained(model_path_hasla)




# W późniejszym czasie, aby wczytać LabelEncoder:
label_encoder_hasla = joblib.load('C:/Users/dariu/model_hasla_8epoch_base/label_encoder_hasla_base.joblib')
#sampled_df['combined_text'] =sampled_df['Tytuł artykułu'].astype(str) + " </tytuł>" + sampled_df['Tekst artykułu'].astype(str)
df['combined_text'] =df['Tytuł artykułu'].astype(str) + " </tytuł>" + df['Tekst artykułu'].astype(str)
sampled_df=df[:1]#[['combined_text','Tytuł artykułu','Tekst artykułu', 'Link', 'do PBL']]

sampled_df['do PBL']=sampled_df['do PBL'].astype(str)

def predict_categories(df, text_column):
    predictions_list = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Predicting categories"):
        text = row[text_column]
        
        # Tokenizacja i przewidywanie dla modelu True/False
        inputs_t_f = tokenizer_t_f(text, return_tensors="pt", padding=True, truncation=True, max_length=514)
        outputs_t_f = model_t_f(**inputs_t_f)
        predictions_t_f = torch.softmax(outputs_t_f.logits, dim=1)
        predicted_index_t_f = predictions_t_f.argmax().item()
        predicted_label_t_f = label_encoder_t_f.inverse_transform([predicted_index_t_f])[0]
        confidence_t_f = predictions_t_f.max().item() * 100  # Procent pewności
        
        genre = ''
        haslo = ''
        confidence_genre = ''  # Początkowa wartość pewności dla gatunku
        confidence_haslo = ''  # Początkowa wartość pewności dla hasła
        
        if predicted_label_t_f == 'True':
            # Przewidywanie gatunku
            inputs_genre = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=514)
            outputs_genre = model_genre(**inputs_genre)
            predictions_genre = torch.softmax(outputs_genre.logits, dim=1)
            predicted_index_genre = predictions_genre.argmax().item()
            genre = label_encoder.inverse_transform([predicted_index_genre])[0]
            confidence_genre = predictions_genre.max().item() * 100  # Procent pewności
            
            # Przewidywanie hasła
            inputs_hasla = tokenizer_hasla(text, return_tensors="pt", padding=True, truncation=True, max_length=514)
            outputs_hasla = model_hasla(**inputs_hasla)
            predictions_hasla = torch.softmax(outputs_hasla.logits, dim=1)
            predicted_index_hasla = predictions_hasla.argmax().item()
            haslo = label_encoder_hasla.inverse_transform([predicted_index_hasla])[0]
            confidence_haslo = predictions_hasla.max().item() * 100  # Procent pewności
            
            row_data = [text, predicted_label_t_f, confidence_t_f, genre, confidence_genre, haslo, confidence_haslo] + [row[col] for col in df.columns if col != text_column]
            predictions_list.append(row_data)
    
    new_columns = [text_column, 'True/False', 'Pewnosc T/F', 'Gatunek', 'Pewnosc Gatunku', 'Hasło', 'Pewnosc Hasła'] + [col for col in df.columns if col != text_column]
    
    predicted_df = pd.DataFrame(predictions_list, columns=new_columns)
    
    return predicted_df


# Przykład użycia funkcji:
result_df = predict_categories(sampled_df, 'combined_text')



# result_df['comparison'] = np.where(result_df['do PBL'] == result_df['True/False'], 'Match', 'Mismatch')
# result_df['comparison_gatunek'] = np.where(result_df['forma/gatunek'] == result_df['Gatunek'], 'Match', 'Mismatch')
# result_df.to_excel('nowe_przewidywania28-06.xlsx', index=False)


# ADD BYTY
model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Utworzenie pipeline NER
nlp1 = pipeline("ner", model=model, tokenizer=tokenizer)
def combine_tokens(ner_results):
    combined_entities = []
    current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0, "start": None, "end": None}
    previous_index = None  # Zmienna do przechowywania indeksu poprzedniego tokenu

    for token in ner_results:
        # Sprawdzamy, czy bieżący token jest końcem słowa
        end_of_word = "</w>" in token['word']
        cleaned_word = token['word'].replace("</w>", "")

        # Sprawdzamy różnicę indeksów, jeśli poprzedni indeks jest ustawiony
        index_difference = token['index'] - previous_index if previous_index is not None else 0

        # Rozpoczęcie nowej jednostki
        if token['entity'].startswith('B-') or index_difference > 5:  # Dodatkowy warunek na różnicę indeksów
            if current_entity['word']:
                # Obliczamy średnią ocenę dla skompletowanej jednostki
                current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
                combined_entities.append(current_entity)
            current_entity = {"word": cleaned_word, "type": token['entity'][2:], "score_sum": token['score'],
                              "token_count": 1, "start": token['start'], "end": token['end']}
        # Kontynuacja obecnej jednostki
        elif token['entity'].startswith('I-') and current_entity['type'] == token['entity'][2:]:
            if previous_end_of_word:
                current_entity['word'] += " " + cleaned_word
            else:
                current_entity['word'] += cleaned_word
            current_entity['end'] = token['end']
            current_entity['score_sum'] += token['score']
            current_entity['token_count'] += 1

        previous_end_of_word = end_of_word
        previous_index = token['index']  # Aktualizacja indeksu poprzedniego tokenu

    # Dodajemy ostatnią jednostkę, jeśli istnieje
    if current_entity['word']:
        current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
        combined_entities.append(current_entity)

    return combined_entities    

threshold = 80

def group_similar_entities(entities, threshold):
    groups = []
    for entity in entities:
        added = False
        for group in groups:
            if any(fuzz.token_sort_ratio(entity, member) > threshold for member in group):
                group.append(entity)
                added = True
                break
        if not added:
            groups.append([entity])
    return groups


def replace_entities_with_representatives(text, map):
    # Tworzenie odwrotnego mapowania dla szybkiego sprawdzenia, czy dana fraza została już użyta jako zastąpienie
    reverse_map = {v: k for k, v in map.items()}
    used_replacements = set()  # Zbiór użytych zastąpień
    
    # Sortowanie kluczy według długości tekstu malejąco
    sorted_entities = sorted(map.keys(), key=len, reverse=True)
    
    for entity in sorted_entities:
        representative = map[entity]

        # Jeśli reprezentant był już użyty jako zastąpienie, pomijamy dalsze zastępowania tego reprezentanta
        if representative in used_replacements:
            continue
        
        # Zastępowanie tylko jeśli reprezentant nie jest częścią wcześniej zastąpionych fraz
        if not any(rep in text for rep in used_replacements if rep != entity):
            pattern = r'\b{}\b'.format(re.escape(entity))
            # Aktualizacja tekstu tylko, gdy fraza nie została jeszcze zastąpiona
            if re.search(pattern, text):
                text = re.sub(pattern, representative, text)
                used_replacements.add(representative)

    return text



import requests
import re

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



def extract_text_from_main_headings(record_data):
    main_headings = []
    if 'mainHeadings' in record_data:
        main_headings_data = record_data['mainHeadings']
        if isinstance(main_headings_data.get('data'), list):
            main_headings.extend(heading.get('text') for heading in main_headings_data['data'] if heading.get('text'))
        elif isinstance(main_headings_data.get('data'), dict):
            main_headings.append(main_headings_data['data'].get('text'))
    return main_headings

def check_viaf_with_fuzzy_match2(entity_name, threshold=80, max_pages=5, entity_type=None):
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

#nowy model NER ladowanie i obsulga
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
from transformers import AutoConfig
import re
from collections import defaultdict

# Ścieżka do modelu
model_directory = "C:/Users/dariu/model_ner_3/"

# Ścieżka do pliku tag2id.json
tag2id_path = "C:/Users/dariu/model_ner_3/tag2id.json"

# Ładowanie mapowania tag2id
with open(tag2id_path, 'r') as f:
    tag2id = json.load(f)

# Odwrócenie mapowania tag2id na id2tag
id2tag = {v: k for k, v in tag2id.items()}

# Załaduj konfigurację modelu i zaktualizuj mapowania etykiet
config = AutoConfig.from_pretrained(model_directory)
config.label2id = tag2id
config.id2label = id2tag
config.save_pretrained(model_directory)

# Załaduj tokenizer i model z aktualną konfiguracją
tokenizer_new = AutoTokenizer.from_pretrained(model_directory)
model_new = AutoModelForTokenClassification.from_pretrained(model_directory, config=config)

nlp_new = pipeline("ner", model=model_new, tokenizer=tokenizer_new, aggregation_strategy="simple")




#koniec
max_entities_new =3 #None #parametr dla naszych nerów, None=bez ograniczeń, gdy chcemy mniej wpisujemy cyfre
nlp = spacy.load("pl_core_news_lg")
result_df['Viaf_AUTHOR'] = pd.NA
result_df['Chosen_Entity'] = pd.NA
result_df['VIAF_URL'] = pd.NA
result_df['Entity_Type'] = pd.NA

for index, row in tqdm(result_df[result_df['True/False'] == "True"].iterrows(),total=result_df[result_df['True/False'] == "True"].shape[0],desc="Processing Rows"):
    text = row['combined_text']
    autor = row['Autor']
    viaf_autor = check_viaf_with_fuzzy_match2(autor,entity_type='personalNames')
    tokens = tokenizer.tokenize(text)
    max_tokens = 512  # Przykładowe ograniczenie modelu
    token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]
    
    # Analiza każdego fragmentu osobno
    ner_results = []
    for fragment in fragments:
        ner_results.extend(nlp1(fragment))
    combined_entities = combine_tokens(ner_results)
    
    combined_entities_selected = [entity for entity in combined_entities if entity['score'] >= 0.92]
    entities = [(entity['word'], entity['type']) for entity in combined_entities_selected]
    
    doc = nlp(text.lower())
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    
    # Lematyzacja bytów i grupowanie
    lemmatized_entities = []
    entity_lemmatization_dict = {}
    for entity in entities:
        doc_entity = nlp(entity[0].lower())
        lemmatized_entity = " ".join([token.lemma_ for token in doc_entity])
        lemmatized_entities.append(lemmatized_entity)
        if lemmatized_entity not in entity_lemmatization_dict:
            entity_lemmatization_dict[lemmatized_entity] = {entity}
        else:
            entity_lemmatization_dict[lemmatized_entity].add(entity)
    
    entity_groups = group_similar_entities(lemmatized_entities, threshold)
    representatives = [sorted(group, key=lambda x: len(x))[0] for group in entity_groups]

    entity_to_representative_map = {}
    for group in entity_groups:
        representative = sorted(group, key=lambda x: (len(x), x))[0]
        for entity in group:
            entity_to_representative_map[entity] = representative
    
    updated_text = replace_entities_with_representatives(lemmatized_text, entity_to_representative_map)
    list_of_new_entities = list(set(entity_to_representative_map.values()))
    
    entity_counts = {entity: 0 for entity in list_of_new_entities}
    title_end_pos = updated_text.find("< /tytuł >")
    if title_end_pos == -1:
        title_end_pos = updated_text.find("< /tytuł>")
    
    for entity in list_of_new_entities:
        total_occurrences = updated_text.count(entity)
        entity_counts[entity] += total_occurrences
        if updated_text.find(entity) < title_end_pos:
            entity_counts[entity] += 50
    
    sorted_entity_counts = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    choosen_ents = [ent for ent in sorted_entity_counts if ent[1] > 5]
    
    # Dodawanie informacji o wybranym bycie do list
    if choosen_ents:
        first_entity_info = choosen_ents[0]
        
        original_entities = entity_lemmatization_dict.get(first_entity_info[0], [])
        
        result_df.at[index, 'Chosen_Entity'] = next(iter(original_entities))[0]
        viaf_url, entity_type = None, "Not found"
        if original_entities:
            chosen_entity = next(iter(original_entities))
            entity_name = chosen_entity[0]
            entity_type_code = chosen_entity[1]
            
            if entity_type_code == "PER":
                viaf_url = check_viaf_with_fuzzy_match2(entity_name, entity_type='personalNames')
            elif entity_type_code == "LOC":
                viaf_url = check_viaf_with_fuzzy_match2(entity_name, entity_type='geographicNames')
            elif entity_type_code == "ORG":
                viaf_url = check_viaf_with_fuzzy_match2(entity_name, entity_type='corporateNames')
            else:
                viaf_url = check_viaf_with_fuzzy_match2(entity_name)
            
            entity_type = entity_type_code
        
        result_df.at[index, 'VIAF_URL'] = viaf_url[0][0] if viaf_url else "Not found"
        result_df.at[index, 'Entity_Type'] = entity_type
    else:
        result_df.at[index, 'Chosen_Entity'] = pd.NA
        result_df.at[index, 'VIAF_URL'] = "Not found"
        result_df.at[index, 'Entity_Type'] = pd.NA
        
    if viaf_autor:
        result_df.at[index, 'Viaf_AUTHOR'] = ', '.join([ent[0] for ent in viaf_autor])
    else:
        result_df.at[index, 'Viaf_AUTHOR'] = "Not found"
        
    # ----- Przetwarzanie za pomocą Nowego Modelu NER (PLAY, BOOK, EVENT) -----
    max_tokens = 514  # Przykładowe ograniczenie modelu
    tokens = tokenizer_new.tokenize(text)
    token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    fragments = [tokenizer_new.convert_tokens_to_string(fragment) for fragment in token_fragments]
    ner_results_model2 = []
    for fragment in fragments:
        ner_results_model2.extend(nlp_new(fragment))
    
    
    # Filtracja encji z nowego modelu, biorąc tylko te z odpowiednich typów i powyżej progu
    filtered_entities_model2 = [
    (entity['word'], entity['entity_group'])
    for entity in ner_results_model2
    if entity['entity_group'] in ['PLAY', 'BOOK', 'EVENT'] and entity['score'] >= 0.80
]

    # Unikanie duplikatów
    unique_filtered_entities = []
    seen_entities = set()
    for entity_name, entity_type_code in filtered_entities_model2:
        entity_key = (entity_name.lower(), entity_type_code)
        if entity_key not in seen_entities:
            unique_filtered_entities.append((entity_name, entity_type_code))
            seen_entities.add(entity_key)
    
    # Przypisywanie encji Modelu 2 do dynamicznych kolumn
    for idx, (entity_name, entity_type_code) in enumerate(unique_filtered_entities, start=2):
        # Sprawdzenie, czy osiągnięto limit bytów
        if max_entities_new is not None and (idx - 1) > max_entities_new:
            break  # Zakończ pętlę, jeśli osiągnięto limit
        
        # Definicja nazw kolumn
        byt_col = f"byt{idx}"
        type_col = f"Type_{idx}"
        viaf_col = f"VIAF_{idx}"
        
        # Dodanie kolumn jeśli nie istnieją
        for col in [byt_col, type_col, viaf_col]:
            if col not in result_df.columns:
                result_df[col] = pd.NA
        
        # Przypisanie nazwy bytu
        result_df.at[index, byt_col] = entity_name
        
        # Przypisanie typu bytu
        result_df.at[index, type_col] = entity_type_code
        
        # Sprawdzanie VIAF dla encji
        viaf_result = None
        if entity_type_code == "PLAY":
            viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='dramaticWorks')  # Sprawdzenie w sztukach dramatycznych
            if not viaf_result:
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='artisticWorks')  # Sprawdzenie w pracach artystycznych
            if not viaf_result:
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='uniformTitles')  # Sprawdzenie w uniform titles
            if not viaf_result:
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='performances')
            # if not viaf_result:
            #     viaf_result = check_viaf_with_fuzzy_match2(entity_name)
        elif entity_type_code == "BOOK":
            # Najpierw sprawdzanie w 'publications'
            viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='publications')
            if not viaf_result:
                # Jeśli brak wyniku, sprawdza w 'uniformTitles'
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='uniformTitles')
            if not viaf_result:
                # Jeśli brak wyniku, sprawdza w 'UniformTitleExpression'
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='uniformTitleExpression')
            if not viaf_result:
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='works')
            if not viaf_result:
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='texts')
        elif entity_type_code == "EVENT":
            viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='eventNames')
            if not viaf_result:
                # Jeśli brak wyniku, sprawdza w 'corporateNames' tylko dla 'EVENT'
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='corporateNames')
            if not viaf_result:
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='conferenceNames')
            if not viaf_result:
                viaf_result = check_viaf_with_fuzzy_match2(entity_name, entity_type='performances')
        # Ustalanie URL-a wyniku lub "Not found"
        if viaf_result and len(viaf_result) > 0:
            # Sortowanie wyników według procentu dopasowania malejąco
            viaf_result_sorted = sorted(viaf_result, key=lambda x: x[1], reverse=True)
            best_viaf = viaf_result_sorted[0]  # Najlepsze dopasowanie
            viaf_url = best_viaf[0]  # URL VIAF
        else:
            viaf_url = "Not found"
    
        
        # Przypisanie URL-a VIAF do odpowiedniej kolumny
        result_df.at[index, viaf_col] = viaf_url
# Zapisanie wyników do pliku Excel
result_df.to_excel('nowe_przewidywania_with_byty.xlsx', index=False)

