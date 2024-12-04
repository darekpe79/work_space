# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:16:30 2024

@author: User
"""

from transformers import HerbertTokenizerFast, AutoModelForTokenClassification
import numpy as np
import requests
import json
import random
import os
import pandas as pd
import spacy
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import torch

# Definicja urządzenia (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ścieżka do zapisanego modelu i tokenizera
save_directory = "C:/Users/dariu/model_ner_3/"

# Ładowanie tokenizera z katalogu zapisanego modelu
tokenizer = HerbertTokenizerFast.from_pretrained(save_directory)

# Ładowanie wytrenowanego modelu
model = AutoModelForTokenClassification.from_pretrained(save_directory)
model.to(device)  # Przeniesienie modelu na GPU, jeśli dostępne

# Ścieżka do danych JSON
json_files_dir = 'D:/Nowa_praca/model QA i inne/dokumenty po anotacji/'
json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]

# Iteracja przez każdy plik JSON i ekstrakcja danych
transformed_data = []
for json_file in json_files:
    file_path = os.path.join(json_files_dir, json_file)
    
    # Ładowanie danych JSON z pliku
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
        # Ekstrakcja adnotacji z danych JSON
        for item in json_data['annotations']:
            text = item[0]  # Ekstrakcja tekstu
            #text = text.replace("[/tytuł]", "")
            entities = item[1]['entities']  # Lista krotek [(start, end, label), ...]
            tuples_list = [tuple(entity) for entity in entities]
            # Dodanie do zbioru danych
            transformed_data.append((text, {'entities': tuples_list}))

# Liczenie typów encji
entity_counts = defaultdict(int)
for text, annotation in transformed_data:
    entities = annotation['entities']
    for start, end, label in entities:
        entity_counts[label] += 1
entity_stats = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
print("Entity Statistics:", entity_stats)

# Funkcja do znajdowania najbliższego punktu podziału tekstu
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

# Funkcja do dzielenia tekstu na części
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

# Dzielenie tekstu na części
transformed_data = split_text_around_entities_adjusted_for_four_parts(transformed_data)

# Usunięcie części bez encji
transformed_data = [data for data in transformed_data if data[1]['entities']]

# Podział danych na treningowe i ewaluacyjne
transformed_data_train, transformed_data_eval = train_test_split(transformed_data, test_size=0.1, random_state=42)

# Mapowanie etykiet na ID
tag2id = {
    'O': 0, 
    'B-PLAY': 1, 
    'I-PLAY': 2,
    'B-EVENT': 3,  # Nowa etykieta dla początku encji EVENT
    'I-EVENT': 4,  # Nowa etykieta dla wnętrza encji EVENT
    'B-BOOK': 5,   # Nowa etykieta dla początku encji BOOK
    'I-BOOK': 6    # Nowa etykieta dla wnętrza encji BOOK
}
id2tag = {v: k for k, v in tag2id.items()}

# Funkcja do przygotowania danych
def prepare_data(data, tokenizer, tag2id, max_length=514):
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
        sequence_labels = ['O'] * len(offset_mapping)  # Inicjalizacja etykiet jako 'O' dla tokenów
        
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
                sequence_labels[entity_start_index] = f'B-{label}'  # Etykieta początku
                for i in range(entity_start_index + 1, entity_end_index + 1):
                    sequence_labels[i] = f'I-{label}'  # Etykieta wnętrza
        
        # Dodanie 'O' dla [CLS] i [SEP] oraz dopasowanie długości etykiet do max_length
        full_sequence_labels = ['O'] + sequence_labels + ['O'] * (max_length - len(sequence_labels) - 1)
        label_ids = [tag2id.get(label, tag2id['O']) for label in full_sequence_labels]
        
        input_ids.append(tokenized_input['input_ids'].squeeze().tolist())
        attention_masks.append(tokenized_input['attention_mask'].squeeze().tolist())
        labels.append(label_ids)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_masks, labels

# Przygotowanie danych treningowych i ewaluacyjnych
input_ids_train, attention_masks_train, labels_train = prepare_data(transformed_data_train, tokenizer, tag2id)
input_ids_eval, attention_masks_eval, labels_eval = prepare_data(transformed_data_eval, tokenizer, tag2id)

# Weryfikacja kształtów danych
print("Training Data Shapes:", input_ids_train.shape, attention_masks_train.shape, labels_train.shape)
print("Evaluation Data Shapes:", input_ids_eval.shape, attention_masks_eval.shape, labels_eval.shape)

# Przykład tokenów i etykiet
example_idx = 10  # Indeks przykładu do wydrukowania
text, annotation = transformed_data_train[example_idx]
tokens = tokenizer.convert_ids_to_tokens(input_ids_train[example_idx])
tags = [list(tag2id.keys())[list(tag2id.values()).index(tag_id)] if tag_id in tag2id.values() else 'O' for tag_id in labels_train[example_idx]]
print(f"Tags:\n{tags}\n")
print(f"Tokens:\n{tokens}\n")
print(f"Input IDs:\n{input_ids_train[example_idx]}\n")
print(f"Attention Masks:\n{attention_masks_train[example_idx]}\n")
print(f"Tag IDs:\n{labels_train[example_idx]}\n")

# Wydrukuj tokeny wraz z etykietami dla lepszej czytelności
tags_readable = [list(tag2id.keys())[list(tag2id.values()).index(tag_id)] if tag_id in tag2id.values() else 'PAD' for tag_id in labels_train[example_idx]]
print(f"Tags:\n{tags_readable}\n")
for token, label in zip(tokens, tags_readable):
    print(f"{token}\t{label}")

# Tworzenie TensorDataset i DataLoader dla ewaluacji
eval_dataset = TensorDataset(input_ids_eval, attention_masks_eval, labels_eval)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=4,  # Możesz dostosować rozmiar wsadu
    shuffle=False
)

# Definicja funkcji ewaluacyjnej
def evaluate_model(model, data_loader, tag2id, id2tag, tokenizer, device):
    model.eval()
    true_labels = []
    pred_labels = []
    total_loss = 0  # Zmienna do sumowania strat

    # Konwersja specjalnych tokenów na ID
    special_token_ids = tokenizer.convert_tokens_to_ids([
        tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token
    ])
    special_token_ids = [tid for tid in special_token_ids if tid is not None]

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids, b_attention_mask, b_labels = tuple(t.to(device) for t in batch)

            # Przekazanie etykiet do modelu
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss  # Pobranie straty
            total_loss += loss.item()  # Sumowanie straty
            logits = outputs.logits

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            input_ids = b_input_ids.cpu().numpy()
            attention_masks = b_attention_mask.cpu().numpy()

            for i in range(len(label_ids)):
                true_labels_seq = []
                pred_labels_seq = []
                for j in range(len(label_ids[i])):
                    if attention_masks[i][j] == 0:
                        continue
                    if input_ids[i][j] in special_token_ids:
                        continue

                    true_label_id = label_ids[i][j]
                    pred_label_id = logits[i][j].argmax()

                    true_label = id2tag.get(true_label_id, 'O')
                    pred_label = id2tag.get(pred_label_id, 'O')

                    true_labels_seq.append(true_label)
                    pred_labels_seq.append(pred_label)

                if true_labels_seq:
                    true_labels.append(true_labels_seq)
                    pred_labels.append(pred_labels_seq)

    # Obliczenie średniej straty walidacyjnej
    avg_val_loss = total_loss / len(data_loader)

    # Obliczanie metryk
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print(f"\nValidation Loss: {avg_val_loss:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1-score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))

    return avg_val_loss  # Zwracamy średnią stratę walidacyjną

# Ocena modelu na zbiorze ewaluacyjnym
print("\nEvaluating on Evaluation Set...")
avg_eval_loss = evaluate_model(model, eval_loader, tag2id, id2tag, tokenizer, device)
