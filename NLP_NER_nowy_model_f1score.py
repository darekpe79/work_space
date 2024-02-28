# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:27:30 2024

@author: dariu
"""

import requests
import json
import pandas as pd
import json

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "byt 1", "zewnętrzny identyfikator bytu 1", "Tytuł spektaklu"]):
    # Wczytanie danych z pliku JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)

    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']]

    # Konwersja wartości w kolumnie 'Tekst artykułu' na stringi
    df_json['Tekst artykułu'] = df_json['Tekst artykułu'].astype(str)

    # Wczytanie danych z pliku Excel
    df_excel = pd.read_excel(excel_file_path)

    # Dodanie kolumny indeksowej do DataFrame'a z Excela
    df_excel['original_order'] = df_excel.index

    # Połączenie DataFrame'ów
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")

    # Sortowanie połączonego DataFrame według kolumny 'original_order'
    merged_df = merged_df.sort_values(by='original_order')

    # Konwersja wartości w kolumnach 'Tytuł artykułu' i 'Tekst artykułu' na stringi w połączonym DataFrame
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)

    # Znalezienie indeksu ostatniego wystąpienia 'zewnętrzny identyfikator bytu 1'
    if 'zewnętrzny identyfikator bytu 1' in merged_df.columns:
        last_id_index = merged_df[merged_df['zewnętrzny identyfikator bytu 1'].notna()].index[-1]
        merged_df = merged_df.loc[:last_id_index]
    else:
        print("Brak kolumny 'zewnętrzny identyfikator bytu 1' w DataFrame.")

    merged_df = merged_df.reset_index(drop=True)

    # Ograniczenie do wybranych kolumn
    if set(selected_columns_list).issubset(merged_df.columns):
        selected_columns = merged_df[selected_columns_list]
    else:
        print("Nie wszystkie wybrane kolumny są dostępne w DataFrame.")
        selected_columns = merged_df

    return selected_columns


json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
                
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'

# ... więcej plików w razie potrzeby

# Użycie funkcji

df2 = load_and_merge_data(json_file_path2, excel_file_path2)



# ... wczytanie kolejnych par plików

# Połączenie wszystkich DataFrame'ów



#%%proba oznaczania jeden df2
df2['combined_text'] = df2['Tytuł artykułu'] + " " + df2['Tekst artykułu']
import pandas as pd
import spacy
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from spacy.training import Example
from spacy.scorer import Scorer

from spacy.tokens import Span


# Załadowanie modelu języka polskiego
nlp = spacy.load("pl_core_news_lg")

# Funkcja do oznaczania słów z tytułów spektakli w tekście
def mark_titles(text, title):
    # Escapowanie specjalnych znaków w tytule
    title_pattern = re.escape(title) + r"(?![\w-])"  # Aby uniknąć dopasowania w środku słowa, dodajemy negative lookahead
    # Oznaczanie tytułu w tekście znacznikami
    marked_text = re.sub(title_pattern, r"[SPEKTAKL]\g<0>[/SPEKTAKL]", text, flags=re.IGNORECASE)
    return marked_text


df2['Tytuł spektaklu'] = df2['Tytuł spektaklu'].fillna('')
df2['marked_text'] = df2.apply(lambda row: mark_titles(row['combined_text'], row['Tytuł spektaklu']), axis=1)
def prepare_data_for_ner(text):
    pattern = r"\[SPEKTAKL\](.*?)\[/SPEKTAKL\]"
    entities = []
    current_pos = 0
    clean_text = ""
    last_end = 0

    for match in re.finditer(pattern, text):
        start, end = match.span()
        clean_text += text[last_end:start]  # Dodaj tekst przed znacznikiem
        start_entity = len(clean_text)
        entity_text = match.group(1)
        clean_text += entity_text  # Dodaj tekst encji bez znaczników
        end_entity = len(clean_text)
        entities.append((start_entity, end_entity, "SPEKTAKL"))
        last_end = end  # Zaktualizuj pozycję ostatniego znalezionego końca znacznika

    clean_text += text[last_end:]  # Dodaj pozostały tekst po ostatnim znaczniku

    return clean_text, {"entities": entities}

df2['spacy_marked'] = df2['marked_text'].apply(prepare_data_for_ner)

import spacy

# Załaduj istniejący model lub utwórz nowy
nlp = spacy.load("pl_core_news_lg")  # Załaduj istniejący model dla języka polskiego
# nlp = spacy.blank("pl")  # Lub utwórz nowy pusty model dla języka polskiego, jeśli wolisz

if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe("ner")

# Dodaj nową etykietę do pipeline'u NER
ner.add_label("SPEKTAKL")

# Przygotowanie danych treningowych z DataFrame
train_df, val_df = train_test_split(df2, test_size=0.15, random_state=42)

# Przygotowanie danych treningowych i walidacyjnych
TRAIN_DATA = [prepare_data_for_ner(row['marked_text']) for index, row in train_df.iterrows()]
VAL_DATA = [prepare_data_for_ner(row['marked_text']) for index, row in val_df.iterrows()]

def evaluate_model_with_metrics(nlp, val_data):
    scorer = Scorer(nlp)
    examples = []
    for text, annotations in val_data:
        pred_doc = nlp(text)
        gold_doc = nlp.make_doc(text)
        gold = Example.from_dict(gold_doc, annotations)
        examples.append(Example(pred_doc, gold.reference))
    scores = scorer.score(examples)
    return scores

# Trening i ewaluacja modelu

n_iter = 170
drop = 0.5
best_f1 = -1  # Użycie F1 jako kryterium
patience_counter = 0
patience = 5



 # Liczba epok bez poprawy po której trening zostanie zatrzymany
patience_counter = 0
output_dir = "model_spektakl"
optimizer = nlp.resume_training()
# W spaCy learn_rate nie jest bezpośrednio ustawiany w optimizer, ale możemy kontrolować szybkość uczenia się poprzez parametry w update()

for i in tqdm(range(n_iter), desc="Trenowanie"):
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=drop, sgd=optimizer, losses=losses)
    
    scores = evaluate_model_with_metrics(nlp, VAL_DATA)
    ner_f1 = scores['ents_f']  # Zakładając, że interesuje nas F1 dla encji
    print(f"Iteracja {i}, F1: {ner_f1}, Straty: {losses}")

    if ner_f1 > best_f1:
        best_f1 = ner_f1
        patience_counter = 0
        nlp.to_disk(output_dir)
        print(f"Nowe najlepsze F1: {best_f1}. Model zapisany.")
    else:
        patience_counter += 1
        print(f"Brak poprawy. Licznik cierpliwości: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print("Zatrzymanie treningu z powodu braku poprawy.")
        break

output_dir = "model_spektakl"

# Zapisanie modelu do dysku
nlp.to_disk(output_dir)

#%%
import random
import os
import json
from spacy.util import minibatch, compounding
from spacy.training.example import Example

df2['combined_text'] = df2['Tytuł artykułu'] + " " + df2['Tekst artykułu']
import pandas as pd
import spacy
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from spacy.training import Example
from spacy.scorer import Scorer

from spacy.tokens import Span


# Załadowanie modelu języka polskiego
nlp = spacy.load("pl_core_news_lg")

# Funkcja do oznaczania słów z tytułów spektakli w tekście
def mark_titles(text, title):
    # Escapowanie specjalnych znaków w tytule
    title_pattern = re.escape(title) + r"(?![\w-])"  # Aby uniknąć dopasowania w środku słowa, dodajemy negative lookahead
    # Oznaczanie tytułu w tekście znacznikami
    marked_text = re.sub(title_pattern, r"[SPEKTAKL]\g<0>[/SPEKTAKL]", text, flags=re.IGNORECASE)
    return marked_text


df2['Tytuł spektaklu'] = df2['Tytuł spektaklu'].fillna('')
df2['marked_text'] = df2.apply(lambda row: mark_titles(row['combined_text'], row['Tytuł spektaklu']), axis=1)
def prepare_data_for_ner(text):
    pattern = r"\[SPEKTAKL\](.*?)\[/SPEKTAKL\]"
    entities = []
    current_pos = 0
    clean_text = ""
    last_end = 0

    for match in re.finditer(pattern, text):
        start, end = match.span()
        clean_text += text[last_end:start]  # Dodaj tekst przed znacznikiem
        start_entity = len(clean_text)
        entity_text = match.group(1)
        clean_text += entity_text  # Dodaj tekst encji bez znaczników
        end_entity = len(clean_text)
        entities.append((start_entity, end_entity, "SPEKTAKL"))
        last_end = end  # Zaktualizuj pozycję ostatniego znalezionego końca znacznika

    clean_text += text[last_end:]  # Dodaj pozostały tekst po ostatnim znaczniku

    return clean_text, {"entities": entities}

df2['spacy_marked'] = df2['marked_text'].apply(prepare_data_for_ner)
nlp = spacy.load("pl_core_news_lg")
ner = nlp.get_pipe("ner")

# Add new entity labels to the NER model, if necessary


# Function to convert your DataFrame's row to spaCy's Example format


# Convert the DataFrame column to a list of (text, entities) tuples
#data = [(text, entity_info['entities']) for text, entity_info in df2['spacy_marked']]
transformed_data=df2['spacy_marked'].to_list()
json_files_dir = 'D:/Nowa_praca/dane_model_jezykowy/jsony_spektakl/'

# List all JSON files in the directory
json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]

# Iterate over each JSON file
for json_file in json_files:
    file_path = os.path.join(json_files_dir, json_file)
    
    # Load the JSON data from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
        # Extract annotations from the JSON data
        for item in json_data['annotations']:
            text = item[0]  # Extract text
            text = text.replace("[/tytuł]", "")
            entities = item[1]['entities']  # Assuming this directly gives a list of tuples [(start, end, label), ...]
            tuples_list = [tuple(item) for item in item[1]['entities']]
            # Append to the existing dataset
            transformed_data.append((text, {'entities':tuples_list}))
# Assuming `data` is your current dataset with texts and corresponding entities
# And each item in `data` looks like (text, [[start, end, label], [start, end, label], ...])

# transformed_data = []

# for text, entities in data:
#     # Transform list of lists into the required dictionary format
#     entity_annotations = {'entities': entities}
#     # Append the tuple of text and entity annotations dictionary to the transformed data
#     transformed_data.append((text, entity_annotations))

# `transformed_data` is now in the correct format for spaCy

train_data, val_data = train_test_split(transformed_data, test_size=0.2, random_state=42)

# Adding new labels to the NER model
for _, annotations in train_data:
    for ent in annotations['entities']:  # Use the 'entities' key to access the entity list
        ner.add_label(ent[2])

# Function to convert data into spaCy's Example format
def convert_to_spacy_format(text, entity_annotations):
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, entity_annotations)
    return example

# Convert the training and validation data to spaCy's format
train_examples = [convert_to_spacy_format(text, annotations) for text, annotations in train_data]
val_examples = [convert_to_spacy_format(text, annotations) for text, annotations in val_data]

# List of pipeline components other than NER to be disabled during training
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

# Training loop
with nlp.disable_pipes(*unaffected_pipes):  # Disable non-NER components
    optimizer = nlp.resume_training()
    for iteration in range(100):  # Adjust the number of iterations as needed
        random.shuffle(train_examples)
        losses = {}
        
        # Batch the examples and iterate over them
        for batch in minibatch(train_examples, size=compounding(4., 32., 1.001)):
            for example in batch:
                # Update the model
                nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)
        print(f"Losses at iteration {iteration}: {losses}")

# Evaluation function
def evaluate(ner_model, examples):
    scorer = Scorer(ner_model)
    scores = scorer.score(examples)
    return scores

# Evaluate the model
evaluation_scores = evaluate(nlp, val_examples)
print(f"Evaluation Scores: {evaluation_scores}")
print(f"Evaluation Scores: {evaluation_scores}")
text = "Wczoraj oglądałem spektakl Płatonow w Teatrze Rampa."
doc = nlp(text)
output_dir = "model_spektakl"

# Zapisanie modelu do dysku
nlp.to_disk(output_dir)
for ent in doc.ents:
    print(ent.text)