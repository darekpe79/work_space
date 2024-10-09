# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:33:30 2024

@author: dariu
"""


from transformers import BertTokenizer
import numpy as np
from transformers import HerbertTokenizerFast
import requests
import json
import random
import os
import json
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import pandas as pd
import spacy
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from spacy.training import Example
from spacy.scorer import Scorer

from spacy.tokens import Span
# Initialize the tokenizer with the Polish model
tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-large-cased')
##LADOWANIE danych JSON
json_files_dir = 'D:/Nowa_praca/dane_model_jezykowy/dokumenty po anotacji-20240930T120225Z-001/dokumenty po anotacji/'
json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]

# Iterate over each JSON file
transformed_data=[]
for json_file in json_files:
    file_path = os.path.join(json_files_dir, json_file)
    
    # Load the JSON data from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
        # Extract annotations from the JSON data
        for item in json_data['annotations']:
            text = item[0]  # Extract text
            #text = text.replace("[/tytuł]", "")
            entities = item[1]['entities']  # Assuming this directly gives a list of tuples [(start, end, label), ...]
            tuples_list = [tuple(item) for item in item[1]['entities']]
            # Append to the existing dataset
            transformed_data.append((text, {'entities':tuples_list}))   
            
from collections import defaultdict

# Initialize a dictionary to hold the count of each entity type
entity_counts = defaultdict(int)

# Iterate over the transformed data
for text, annotation in transformed_data:
    entities = annotation['entities']
    
    # Count each entity label
    for start, end, label in entities:
        
        entity_counts[label] += 1     
entity_stats = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)

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
# Existing tag2id mapping
tag2id = {
    'O': 0, 
    'B-PLAY': 1, 
    'I-PLAY': 2,
    'B-EVENT': 3,  # New label for the beginning of an EVENT entity
    'I-EVENT': 4,  # New label for the inside of an EVENT entity
    'B-BOOK': 5,   # New label for the beginning of a BOOK entity
    'I-BOOK': 6    # New label for the inside of a BOOK entity
}


# Przygotowanie danych
input_ids, attention_masks, labels = prepare_data(transformed_data_train, tokenizer, tag2id)
# Przygotowanie danych ewaluacyjnych
input_ids_eval, attention_masks_eval, labels_eval = prepare_data(transformed_data_eval, tokenizer, tag2id)


# Weryfikacja wyników
print(input_ids.shape, attention_masks.shape, labels.shape)

example_idx = 10  # indeks przykładu, który chcemy wydrukować
text, annotation = transformed_data_train[example_idx]
# Konwersja input_ids do tokenów
tokens = tokenizer.convert_ids_to_tokens(input_ids[example_idx])
tags = [list(tag2id.keys())[list(tag2id.values()).index(tag_id)] if tag_id in tag2id.values() else 'O' for tag_id in labels[example_idx]]
print(f"Tags:\n{tags}\n")
print(f"Tokens:\n{tokens}\n")
print(f"Input IDs:\n{input_ids[example_idx]}\n")
print(f"Attention Masks:\n{attention_masks[example_idx]}\n")
print(f"Tag IDs:\n{labels[example_idx]}\n")

# Wydrukuj skojarzone z tokenami etykiety (dla lepszej czytelności)
tags = [list(tag2id.keys())[list(tag2id.values()).index(tag_id)] if tag_id in tag2id.values() else 'PAD' for tag_id in labels[example_idx]]
print(f"Tags:\n{tags}\n")
for token, label in zip(tokens, tags):
    print(f"{token}\t{label}")

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
#%% Pierwsze podejscie
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

for epoch in range(5):  # Liczba epok
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
        
#%%druga petla
from seqeval.metrics import classification_report, f1_score
from seqeval.metrics import precision_score, recall_score
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import json
from tqdm import tqdm
# Funkcja ewaluacyjna
def evaluate_model(model, data_loader, tag2id, id2tag, tokenizer, device):
    model.eval()
    true_labels = []
    pred_labels = []
    total_loss = 0  # Zmienna do sumowania strat

    special_token_ids = tokenizer.convert_tokens_to_ids([
        tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token])
    special_token_ids = [tid for tid in special_token_ids if tid is not None]

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids, b_attention_mask, b_labels = tuple(
                t.to(device) for t in batch)

            # Przekazanie etykiet do modelu
            outputs = model(b_input_ids, attention_mask=b_attention_mask,
                            labels=b_labels)
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

                    true_label = id2tag[true_label_id]
                    pred_label = id2tag[pred_label_id]

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

# Definicja liczby epok przed obliczeniem total_steps
num_epochs = 5  # Liczba epok

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

total_steps = len(train_loader) * num_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Przeniesienie modelu na odpowiednie urządzenie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inicjalizacja wczesnego zatrzymania
best_val_loss = float('inf')
patience = 2
epochs_no_improve = 0

print_loss_every = 20  # Drukuj stratę co 20 kroków
id2tag = {v: k for k, v in tag2id.items()}
for epoch in range(num_epochs):
    total_loss = 0
    total_train_loss = 0
    step = 0
    model.train()

    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attention_mask, b_labels = batch

        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask,
                        labels=b_labels)

        loss = outputs.loss
        total_loss += loss.item()
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % print_loss_every == 0:
            avg_loss = total_loss / print_loss_every
            print(f"  Step {step + 1}, Loss: {avg_loss:.4f}")
            total_loss = 0

        step += 1

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    # Ewaluacja po każdej epoce
    print(f"\nEvaluating after epoch {epoch + 1}...")
    avg_val_loss = evaluate_model(model, eval_loader, tag2id, id2tag,
                                  tokenizer, device)

    # Sprawdzenie wczesnego zatrzymania
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Zapisz najlepszy model
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break

# Po treningu wczytaj najlepszy model
model.load_state_dict(torch.load('best_model.pt'))

# Zapisanie modelu i tokenizera
save_directory = 'C:/Users/dariu/model_NER2/'

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Ścieżka do zapisu mapowania tag2id
tag2id_path = "C:/Users/dariu/model_NER2/tag2id.json"

# Zapisanie tag2id do pliku JSON
with open(tag2id_path, 'w') as f:
    json.dump(tag2id, f)
    json.dump(tag2id, f)