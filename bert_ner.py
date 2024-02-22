# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:13:24 2024

@author: dariu
"""

import re
from transformers import AutoTokenizer
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

json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'
json_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afront_2022-09-08.json'
excel_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afront_2022-09-08.xlsx'
json_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/artpapier_2022-10-05.json'
excel_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/artpapier_2022-10-05.xlsx'
json_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/audycjekulturalne_2022-10-11.json'
excel_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/audycjekulturalne_2022-10-11.xlsx'
json_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/bylam_widzialam_2023-02-21.json'
excel_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/bylam_widzialam_2023-02-21.xlsx'
json_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/czas_kultury_2023-03-24.json'
excel_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/czas_kultury_2023-03-24.xlsx'
json_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_dziennik_2023-10-23.json'
excel_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_dziennik_2023-10-23.xlsx'
json_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/intimathule_2022-09-09.json'
excel_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/intimathule_2022-09-09.xlsx'
json_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/jerzy_sosnowski_2022-09-09.json'
excel_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jerzy_sosnowski_2022-09-09.xlsx'
json_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/komnen_kastamonu_2022-09-12.json'
excel_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/komnen_kastamonu_2022-09-12.xlsx'
json_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/krzysztof_jaworski_2022-12-08.json'
excel_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/krzysztof_jaworski_2022-12-08.xlsx'
json_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pgajda_2022-09-13.json'
excel_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pgajda_2022-09-13.xlsx'
json_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/poeci_po_godzinach_2022-09-14.json'
excel_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/poeci_po_godzinach_2022-09-14.xlsx'
# ... więcej plików w razie potrzeby

# Użycie funkcji
df1 = load_and_merge_data(json_file_path, excel_file_path)
df2 = load_and_merge_data(json_file_path2, excel_file_path2)
df3 = load_and_merge_data(json_file_path3, excel_file_path3)
df4 = load_and_merge_data(json_file_path4, excel_file_path4)
df5 = load_and_merge_data(json_file_path5, excel_file_path5)
df6 = load_and_merge_data(json_file_path6, excel_file_path6)
df7 = load_and_merge_data(json_file_path7, excel_file_path7)
df8 = load_and_merge_data(json_file_path8, excel_file_path8)
df9 = load_and_merge_data(json_file_path9, excel_file_path9)
df10 = load_and_merge_data(json_file_path10, excel_file_path10)
df11 = load_and_merge_data(json_file_path11, excel_file_path11)
df12 = load_and_merge_data(json_file_path12, excel_file_path12)
df13 = load_and_merge_data(json_file_path13, excel_file_path13)
df14 = load_and_merge_data(json_file_path14, excel_file_path14)
df15 = load_and_merge_data(json_file_path15, excel_file_path15)

# ... wczytanie kolejnych par plików

# Połączenie wszystkich DataFrame'ów
combined_df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], ignore_index=True)
combined_df = pd.concat([df1, df2], ignore_index=True)


#%%proba oznaczania jeden df2
df2['combined_text'] = df2['Tytuł artykułu'] + " " + df2['Tekst artykułu']
import pandas as pd
import spacy
import re
import torch

# Załadowanie modelu języka polskiego
#nlp = spacy.load("pl_core_news_lg")

# Funkcja do oznaczania słów z tytułów spektakli w tekście
def mark_titles(text, title):
    # Escapowanie specjalnych znaków w tytule
    title_pattern = re.escape(title) + r"(?![\w-])"  # Aby uniknąć dopasowania w środku słowa, dodajemy negative lookahead
    # Oznaczanie tytułu w tekście znacznikami
    marked_text = re.sub(title_pattern, r"[SPEKTAKL]\g<0>[/SPEKTAKL]", text, flags=re.IGNORECASE)
    return marked_text


df2['Tytuł spektaklu'] = df2['Tytuł spektaklu'].fillna('')
df2['marked_text'] = df2.apply(lambda row: mark_titles(row['combined_text'], row['Tytuł spektaklu']), axis=1)

# Inicjalizacja tokennizera
model_checkpoint = "allegro/herbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def prepare_ner_data(text):
    # Znajdź wszystkie anotacje i teksty pomiędzy nimi
    pattern = r"\[SPEKTAKL](.*?)\[/SPEKTAKL]"
    matches = re.finditer(pattern, text, re.DOTALL)
    
    processed_tokens = []
    processed_labels = []
    
    # Przetwarzaj tekst przed, pomiędzy i po anotacjach
    last_end = 0
    for match in matches:
        # Tekst przed anotacją
        before_text = text[last_end:match.start()]
        before_tokens = tokenizer.tokenize(before_text)
        processed_tokens.extend(before_tokens)
        processed_labels.extend(['O'] * len(before_tokens))
        
        # Tekst anotacji
        entity_text = match.group(1)
        entity_tokens = tokenizer.tokenize(entity_text)
        processed_tokens.extend(entity_tokens)
        processed_labels.extend(['B-SPEKTAKL'] + ['I-SPEKTAKL'] * (len(entity_tokens) - 1))
        
        last_end = match.end()
    
    # Tekst po ostatniej anotacji
    after_text = text[last_end:]
    after_tokens = tokenizer.tokenize(after_text)
    processed_tokens.extend(after_tokens)
    processed_labels.extend(['O'] * len(after_tokens))
    
    return processed_tokens, processed_labels

# Przykład użycia



# Przetwarzanie wszystkich tekstów
processed_data = df2['marked_text'].apply(lambda x: prepare_ner_data(x))

# Rozpakowanie danych do list tokenów i etykiet
#tokens = [item[0] for item in processed_data]
#labels = [item[1] for item in processed_data]
tokens_list, labels_list = zip(*processed_data)



def adjust_labels_for_sequence(labels_indexed, max_length=512, padding_value=-100):
    adjusted_labels = []
    for labels in labels_indexed:
        # Ustawienie etykiet dla [CLS] i [SEP]
        adjusted = [padding_value] + labels[:max_length-2] + [padding_value]
        # Dodanie paddingu, jeśli jest to konieczne
        adjusted += [padding_value] * (max_length - len(adjusted))
        adjusted_labels.append(adjusted[:max_length])
    return adjusted_labels
label_map = {"O": 0, "B-SPEKTAKL": 1, "I-SPEKTAKL": 2}
labels_indexed = [[label_map[label] for label in single_labels_list] for single_labels_list in labels_list]

# max_length = max(len(labels_seq) for labels_seq in labels_indexed)


input_ids = []
attention_masks = []
for text in df2['combined_text']:
    encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=512,  # Ustaw maksymalną długość na podstawie modelu lub danych
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
print("Długość sekwencji input_ids przed dostosowaniem: ", input_ids.size(1))
print("Długość sekwencji input_ids przed dostosowaniem: ", input_ids.size(1))
for i, label in enumerate(labels_indexed):
    print(f"Długość sekwencji {i} w input_ids: 512, długość etykiet: {len(label)}")
    # Tu możesz dodać więcej logiki debugowania, jeśli potrzebujesz

# Tutaj dostosowujemy labels do długości input_ids
adjusted_labels = adjust_labels_for_sequence(labels_indexed)
labels_tensors = torch.tensor(adjusted_labels, dtype=torch.long)
from torch.utils.data import TensorDataset
# Tworzenie datasetu
dataset = TensorDataset(input_ids, attention_masks, labels_tensors)






# Podział na zbiory treningowy i walidacyjny.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32  # Możesz dostosować w zależności od możliwości obliczeniowych

train_dataloader = DataLoader(
            train_dataset,  # Dane treningowe.
            sampler = RandomSampler(train_dataset), # Wybierz dane losowo do treningu.
            batch_size = batch_size # Trenuj na batchach o rozmiarze 'batch_size'.
        )

validation_dataloader = DataLoader(
            val_dataset, # Dane walidacyjne.
            sampler = SequentialSampler(val_dataset), # Pobieraj dane sekwencyjnie
            batch_size = batch_size # Ewaluuj na batchach o rozmiarze 'batch_size'.
        )
from transformers import AutoModelForTokenClassification, AdamW

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_map),  # liczba etykiet klasyfikacji (np. dla NER)
    output_attentions = False,  # model zwraca attention weights, jeśli True
    output_hidden_states = False, # model zwraca wszystkie ukryte stany, jeśli True
)
optimizer = AdamW(model.parameters(),
                  lr = 5e-5,  # domyślna szybkość uczenia
                  eps = 1e-8  # domyślna wartość epsilon
                 )
from transformers import get_linear_schedule_with_warmup

epochs = 4  # Liczba epok

# Całkowita liczba kroków treningowych to liczba batchy razy liczba epok.
total_steps = len(train_dataloader) * epochs

# Stwórz harmonogram zmian szybkości uczenia
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # domyślna wartość dla warmup
                                            num_training_steps = total_steps)

