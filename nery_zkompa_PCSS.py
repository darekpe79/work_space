# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:08:27 2024

@author: dariu
"""

from transformers import BertTokenizer
import numpy as np
from transformers import HerbertTokenizerFast

# Initialize the tokenizer with the Polish model
tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-large-cased')
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


json_file_path2 = 'C:/Users/User/Desktop/materiał_do_treningu/drive-download-20240125T115916Z-001/jsony/afisz_teatralny_2022-09-08.json'
                
excel_file_path2 = 'C:/Users/User/Desktop/materiał_do_treningu/drive-download-20240125T115916Z-001/afisz_teatralny_2022-09-08.xlsx'
# ... więcej plików w razie potrzeby

# Użycie funkcji

df2 = load_and_merge_data(json_file_path2, excel_file_path2)
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




# Funkcja do oznaczania słów z tytułów spektakli w tekście
def mark_titles(text, title):
    # Escapowanie specjalnych znaków w tytule
    title_pattern = re.escape(title) + r"(?![\w-])"  # Aby uniknąć dopasowania w środku słowa, dodajemy negative lookahead
    # Oznaczanie tytułu w tekście znacznikami
    marked_text = re.sub(title_pattern, r"[PLAY]\g<0>[/PLAY]", text, flags=re.IGNORECASE)
    return marked_text

df2.dropna(subset=['Tytuł spektaklu'], inplace=True)

df2['Tytuł spektaklu'] = df2['Tytuł spektaklu'].fillna('')
df2['marked_text'] = df2.apply(lambda row: mark_titles(row['combined_text'], row['Tytuł spektaklu']), axis=1)
def prepare_data_for_ner(text):
    pattern = r"\[PLAY\](.*?)\[/PLAY\]"
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
        entities.append((start_entity, end_entity, "PLAY"))
        last_end = end  # Zaktualizuj pozycję ostatniego znalezionego końca znacznika

    clean_text += text[last_end:]  # Dodaj pozostały tekst po ostatnim znaczniku

    return clean_text, {"entities": entities}

df2['spacy_marked'] = df2['marked_text'].apply(prepare_data_for_ner)

import json
import os
transformed_data=df2['spacy_marked'].to_list()
def find_nearest_acceptable_split_point(pos, text, total_length):
    """
    Finds the nearest split point that does not split words or sentences,
    preferring to split at punctuation followed by space or at natural sentence boundaries.
    """
    if pos <= 0:
        return 0
    if pos >= total_length:
        return total_length
    
    for offset in range(1, min(50, pos, total_length - pos) + 1):
        left = pos - offset
        right = pos + offset

        if left > 0 and text[left - 1] in '.?!' and text[left] == ' ':
            return left + 1

        if right < total_length and text[right - 1] in '.?!' and text[right] == ' ':
            return right + 1
    
    for offset in range(1, min(50, pos, total_length - pos) + 1):
        left = pos - offset
        right = pos + offset

        if left > 0 and text[left] == ' ':
            return left + 1

        if right < total_length and text[right] == ' ':
            return right + 1

    return pos

def split_text_around_entities_adjusted_for_four_parts(data_list):
    split_data = []
    
    for text, annotation in data_list:
        entities = sorted(annotation['entities'], key=lambda e: e[0])
        total_length = len(text)
        ideal_part_length = total_length // 5  # Adjusted for four parts
        
        split_points = [0]
        current_split = 0
        
        for _ in range(4):  # Adjusted to perform three splits for four parts
            proposed_split = current_split + ideal_part_length
            if proposed_split >= total_length:
                break
            
            adjusted_split = find_nearest_acceptable_split_point(proposed_split, text, total_length)
            
            for start, end, _ in entities:
                if adjusted_split > start and adjusted_split < end:
                    adjusted_split = end
                    break
            
            if adjusted_split != current_split:
                split_points.append(adjusted_split)
                current_split = adjusted_split
        
        split_points.append(total_length)
        
        last_split = 0
        for split in split_points[1:]:
            part_text = text[last_split:split].strip()
            part_entities = [(start - last_split, end - last_split, label) for start, end, label in entities if start >= last_split and end <= split]
            split_data.append((part_text, {'entities': part_entities}))
            last_split = split

    return split_data
transformed_data = split_text_around_entities_adjusted_for_four_parts(transformed_data)

transformed_data = [data for data in transformed_data if data[1]['entities']]

from sklearn.model_selection import train_test_split




transformed_data_train, transformed_data_eval = train_test_split(transformed_data, test_size=0.1, random_state=42)
from transformers import BertTokenizerFast
import torch

# Przygotowanie tokenizera
#tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-base-cased')

def prepare_data(data, tokenizer, tag2id, max_length=512):
    input_ids = []
    attention_masks = []
    labels = []
    
    for text, annotation in data:
        tokenized_input = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        offset_mapping = tokenized_input["offset_mapping"].squeeze().tolist()[1:-1]  # Usunięcie mapowania dla [CLS] i [SEP]
        sequence_labels = ['O'] * len(offset_mapping)  # Inicjalizacja etykiet jako 'O' dla tokenów (bez [CLS] i [SEP])
        
        for start, end, label in annotation['entities']:
            entity_start_index = None
            entity_end_index = None
            
            for idx, (offset_start, offset_end) in enumerate(offset_mapping):
                if start == offset_start or (start > offset_start and start < offset_end):
                    entity_start_index = idx
                if (end > offset_start and end <= offset_end):
                    entity_end_index = idx
                    break

            if entity_start_index is not None and entity_end_index is not None:
                sequence_labels[entity_start_index] = f'B-{label}'  # Begin label
                for i in range(entity_start_index + 1, entity_end_index + 1):
                    sequence_labels[i] = f'I-{label}'  # Inside label
        
        # Dodajemy 'O' dla [CLS] i [SEP] oraz dopasowujemy długość etykiet do max_length
        full_sequence_labels = ['O'] + sequence_labels + ['O'] * (max_length - len(sequence_labels) - 1)
        label_ids = [tag2id.get(label, tag2id['O']) for label in full_sequence_labels]
        
        input_ids.append(tokenized_input['input_ids'].squeeze().tolist())
        attention_masks.append(tokenized_input['attention_mask'].squeeze().tolist())
        labels.append(label_ids)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_masks, labels



# Mapowanie etykiet
tag2id = {'O': 0, 'B-PLAY': 1, 'I-PLAY': 2}

# Przygotowanie danych
input_ids, attention_masks, labels = prepare_data(transformed_data_train, tokenizer, tag2id)
# Przygotowanie danych ewaluacyjnych
input_ids_eval, attention_masks_eval, labels_eval = prepare_data(transformed_data_eval, tokenizer, tag2id)


# Weryfikacja wyników
print(input_ids.shape, attention_masks.shape, labels.shape)

example_idx = 0  # indeks przykładu, który chcemy wydrukować

# Konwersja input_ids do tokenów
tokens = tokenizer.convert_ids_to_tokens(input_ids[example_idx])

print(f"Tokens:\n{tokens}\n")
print(f"Input IDs:\n{input_ids[example_idx]}\n")
print(f"Attention Masks:\n{attention_masks[example_idx]}\n")
print(f"Tag IDs:\n{labels[example_idx]}\n")

# Wydrukuj skojarzone z tokenami etykiety (dla lepszej czytelności)
tags = [list(tag2id.keys())[list(tag2id.values()).index(tag_id)] if tag_id in tag2id.values() else 'PAD' for tag_id in labels[example_idx]]
print(f"Tags:\n{tags}\n")
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    'allegro/herbert-large-cased',
    num_labels=len(tag2id)
)
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
eval_dataset = TensorDataset(input_ids_eval, attention_masks_eval, labels_eval)

# DataLoader dla danych ewaluacyjnych
eval_loader = DataLoader(
    eval_dataset,
    batch_size=4,  # Dostosuj zgodnie z potrzebami
    shuffle=False  # Nie ma potrzeby mieszać danych ewaluacyjnych
)
# Przygotowanie TensorDataset
train_dataset = TensorDataset(input_ids, attention_masks, labels)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # Możesz dostosować w zależności od zasobów
    sampler=RandomSampler(train_dataset)  # Mieszanie danych
)
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# Przenieś model na odpowiednie urządzenie (GPU, jeśli dostępne)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Pętla treningowa
from transformers import logging
logging.set_verbosity_info()

print_loss_every = 20  # Drukuj loss co 50 kroków
step = 0

for epoch in range(3):  # Liczba epok
    total_loss = 0
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attention_mask, b_labels = batch
        
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % print_loss_every == 0:
            print(f"Step {step + 1}, Loss: {total_loss / print_loss_every}")
            total_loss = 0
        
        step += 1
        
        
save_directory = "C:/Users/User/Desktop/model_NER"

# Zapisanie modelu
model.save_pretrained(save_directory)

# Zapisanie tokennizera
tokenizer.save_pretrained(save_directory)
import json

# Ścieżka, gdzie chcesz zapisać mapowanie tag2id
tag2id_path = "C:/Users/User/Desktop/model_NER/tag2id.json"

# Zapisanie tag2id do pliku JSON
with open(tag2id_path, 'w') as f:
    json.dump(tag2id, f)
    
    
    
    
    
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Ustawienie modelu w tryb ewaluacji
model.eval()

eval_loss = 0
predictions , true_labels = [], []

# Iteracja przez DataLoader danych ewaluacyjnych
for batch in eval_loader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_attention_mask, b_labels = batch
    
    # Wyłączenie obliczania gradientów
    with torch.no_grad():
        # Przewidywanie etykiet przez model
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
    
    # Akumulacja straty
    eval_loss += outputs.loss.item()
    
    # Przechwytywanie przewidywań i prawdziwych etykiet
    logits = outputs.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    predictions.append(logits)
    true_labels.append(label_ids)

# Obliczenie średniej straty
eval_loss = eval_loss / len(eval_loader)
print(f"Evaluation loss: {eval_loss}")

# Obliczenie metryk, np. dokładności, precyzji, pełności i miary F1
predictions = np.argmax(np.concatenate(predictions, axis=0), axis=2)
true_labels = np.concatenate(true_labels, axis=0)

# Dla uproszczenia: obliczanie dokładności
accuracy = accuracy_score(true_labels.flatten(), predictions.flatten())
print(f"Accuracy: {accuracy}")

# Dla bardziej zaawansowanych metryk, możesz dostosować poniższe linijki
precision, recall, f1, _ = precision_recall_fscore_support(true_labels.flatten(), predictions.flatten(), average='weighted')
print(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}")



def predict_ner(text, model, tokenizer, tag2id):
    # Tokenizacja tekstu
    tokenized_sentence = tokenizer.encode(text, return_tensors="pt")
    
    # Predykcja modelu
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    with torch.no_grad():
        output = model(tokenized_sentence)
    
    # Dekodowanie etykiet
    label_indices = np.argmax(output.logits.to('cpu').numpy(), axis=2)
    
    # Pobranie tokenów i odpowiadających im etykiet
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(list(tag2id.keys())[list(tag2id.values()).index(label_idx)])
            new_tokens.append(token)
    
    # Wyświetlenie tokenów z przewidzianymi etykietami
    for token, label in zip(new_tokens, new_labels):
        print(f"{token}: {label}")

# Przykładowy tekst
text = '''23 października klasa 7 wybrała się do Teatru Cracovia na spektakl „Balladyna”. Czy warto obejrzeć tę sztukę? Przeczytajcie opinie siódmoklasistów:

Spektakl Balladyna

Dnia 23 października 2019 roku w Teatrze Cracovia na terenie Centrum Kultury Solvay odbył się spektakl pod tytułem Balladyna na podstawie dramatu Juliusz Słowackiego o takim samym tytule. Jest on prezentowany od 2008 roku. Ma obecną formę dzięki reżyserii i scenografii Annę Kasprzyk.'''

# Użycie funkcji
predict_ner(text, model, tokenizer, tag2id)