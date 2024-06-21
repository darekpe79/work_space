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
sampled_df=df[:400]#[['combined_text','Tytuł artykułu','Tekst artykułu', 'Link', 'do PBL']]

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
        
        #if predicted_label_t_f == 'True':
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



result_df['comparison'] = np.where(result_df['do PBL'] == result_df['True/False'], 'Match', 'Mismatch')
result_df['comparison_gatunek'] = np.where(result_df['forma/gatunek'] == result_df['Gatunek'], 'Match', 'Mismatch')
result_df.to_excel('nowe_przewidywania21-06.xlsx', index=False)


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
    # Usuwanie dat z tekstu, np. "Emma Goldman, 1869-1940" staje się "Emma Goldman"
    return re.sub(r',?\s*\d{4}(-\d{4})?', '', text)

def check_viaf_with_fuzzy_match(entity_name, threshold=87):
    base_url = "http://viaf.org/viaf/AutoSuggest"
    query_params = {'query': entity_name}
    best_match = None
    best_score = 0
    
    try:
        response = requests.get(base_url, params=query_params)
        response.raise_for_status()
        data = response.json()

        # Dodatkowe sprawdzenie, czy 'result' jest w danych i czy nie jest None
        if data and data.get('result') is not None:
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

    except requests.RequestException as e:
        print(f"Error querying VIAF: {e}")
    
    if best_match:
        viaf_id = best_match.get('viafid')
        return f"http://viaf.org/viaf/{viaf_id}", best_score
    
    return None, None


nlp = spacy.load("pl_core_news_lg")
result_df['Chosen_Entity'] = pd.NA
result_df['VIAF_URL'] = pd.NA
result_df['Entity_Type'] = pd.NA

for index, row in tqdm(result_df[result_df['True/False'] == "True"].iterrows()):
    text = row['combined_text']
    tokens = tokenizer.tokenize(text)
    max_tokens = 514  # Przykładowe ograniczenie modelu
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
    #lemmatized_text=text.lower()

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
            viaf_url, _ = check_viaf_with_fuzzy_match(next(iter(original_entities))[0])  # Pobieranie pierwszego elementu z setu
            entity_type = next(iter(original_entities))[1]
        
        result_df.at[index, 'VIAF_URL'] = viaf_url if viaf_url else "Not found"
        result_df.at[index, 'Entity_Type'] = entity_type
    else:
        result_df.at[index, 'Chosen_Entity'] = pd.NA
        result_df.at[index, 'VIAF_URL'] = "Not found"
        result_df.at[index, 'Entity_Type'] = pd.NA
        
result_df.to_excel('nowe_przewidywania_with_byty.xlsx', index=False)