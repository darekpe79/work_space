# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:16:30 2024

@author: User
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
json_files_dir = 'D:/Nowa_praca/adnotacje_spubi/anotowane/'
json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]
labels_to_remove = {'TOM', 'WYDAWNICTWO'}
# Iterate over each JSON file

            
transformed_data = []

for json_file in json_files:
    file_path = os.path.join(json_files_dir, json_file)
    
    # Ładowanie danych JSON z pliku
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
        # Ekstrakcja adnotacji z danych JSON
        for item in json_data['annotations']:
            text = item[0]  # Wyodrębnienie tekstu
            entities = item[1]['entities']  # Lista encji
            
            # Filtracja encji, aby usunąć te, które nie są reprezentatywne
            filtered_entities = [
                (start, end, label) 
                for start, end, label in entities 
                if label not in labels_to_remove
            ]
            
            # Tylko dodaj dane, jeśli po filtracji są jakieś encje
            if filtered_entities:
                transformed_data.append((text, {'entities': filtered_entities}))
            
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


transformed_data = [data for data in transformed_data if data[1]['entities']]

from sklearn.model_selection import train_test_split




transformed_data_train, transformed_data_eval = train_test_split(transformed_data, test_size=0.1, random_state=42)

from transformers import BertTokenizerFast
import torch

# Przygotowanie tokenizera
#tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-base-cased')

def prepare_data(data, tokenizer, tag2id, max_length=256):
    input_ids = []
    attention_masks = []
    labels = []
    
    for text, annotation in data:
        # Tokenizacja z zachowaniem offsetów
        tokenized_input = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_id = tokenized_input['input_ids'].squeeze().tolist()
        attention_mask = tokenized_input['attention_mask'].squeeze().tolist()
        offset_mapping = tokenized_input['offset_mapping'].squeeze().tolist()  # Uwzględnij [CLS] i [SEP]
        
        sequence_labels = ['O'] * len(offset_mapping)  # Inicjalizacja etykiet jako 'O' dla tokenów
        
        for start, end, label in annotation['entities']:
            entity_start_index = None
            entity_end_index = None
            
            for idx, (offset_start, offset_end) in enumerate(offset_mapping):
                if idx == 0 or idx == len(offset_mapping) -1:
                    continue  # Ignoruj [CLS] i [SEP]
                
                if start == offset_start or (start > offset_start and start < offset_end):
                    entity_start_index = idx
                if (end > offset_start and end <= offset_end):
                    entity_end_index = idx
                    break

            if entity_start_index is not None and entity_end_index is not None:
                sequence_labels[entity_start_index] = f'B-{label}'  # Begin label
                for i in range(entity_start_index + 1, entity_end_index + 1):
                    sequence_labels[i] = f'I-{label}'  # Inside label
        
        # Ustawienie etykiet dla [CLS] i [SEP] na -100
        sequence_labels[0] = -100  # [CLS]
        sequence_labels[-1] = -100  # [SEP]
        
        # Ustawienie etykiet paddingu na -100
        for i in range(len(sequence_labels)):
            if attention_mask[i] == 0:
                sequence_labels[i] = -100  # Ignoruj padding
        
        # Konwersja etykiet na ID, ustawienie -100 dla paddingu
        label_ids = [
            tag2id.get(lbl, tag2id['O']) if lbl != -100 else -100 
            for lbl in sequence_labels
        ]
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label_ids)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_masks, labels

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import json
from datasets import Dataset

# Definicja tagów
tag2id = {
    'O': 0,
    
    'B-MIEJSCE WYDANIA': 1,
    'I-MIEJSCE WYDANIA': 2,
    
    'B-TYTUŁ': 3,
    'I-TYTUŁ': 4,
    
    'B-DATA': 5,
    'I-DATA': 6,
    
    'B-STRONY': 7,
    'I-STRONY': 8,
    
    'B-WSPÓŁTWÓRCA': 9,
    'I-WSPÓŁTWÓRCA': 10,
    
    'B-AUTOR': 11,
    'I-AUTOR': 12,
    
    'B-FUNKCJA WSPÓŁTWÓRSTWA': 13,
    'I-FUNKCJA WSPÓŁTWÓRSTWA': 14,
    
    'B-DOPISEK BIBLIOGRAFICZNY': 15,
    'I-DOPISEK BIBLIOGRAFICZNY': 16
}

id2tag = {v: k for k, v in tag2id.items()}

# Przygotowanie danych treningowych
input_ids, attention_masks, labels = prepare_data(transformed_data_train, tokenizer, tag2id)

# Przygotowanie danych ewaluacyjnych
input_ids_eval, attention_masks_eval, labels_eval = prepare_data(transformed_data_eval, tokenizer, tag2id)



# Przygotowanie listy tekstów
texts_train = [data[0] for data in transformed_data_train]
texts_eval = [data[0] for data in transformed_data_eval]

# Konwersja TensorDataset do słowników z tekstem
def convert_to_dict_with_text(input_ids, attention_masks, labels, texts):
    return {
        'input_ids': input_ids.tolist(),
        'attention_mask': attention_masks.tolist(),
        'labels': labels.tolist(),
        'text': texts
    }

train_dict = convert_to_dict_with_text(input_ids, attention_masks, labels, texts_train)
eval_dict = convert_to_dict_with_text(input_ids_eval, attention_masks_eval, labels_eval, texts_eval)

# Konwersja do Huggingface Dataset
train_dataset = Dataset.from_dict(train_dict)
eval_dataset = Dataset.from_dict(eval_dict)

# Przykładowa weryfikacja danych treningowych
for i in range(1):  # Sprawdź pierwsze 3 przykłady
    original_text = train_dataset[i]['text']  # Pobranie oryginalnego tekstu
    input_id = train_dataset[i]['input_ids']
    attention_mask = train_dataset[i]['attention_mask']
    label_id = train_dataset[i]['labels']
    
    tokens = tokenizer.convert_ids_to_tokens(input_id)
    labels_seq = [id2tag.get(l, 'PAD') if l != -100 else 'PAD' for l in label_id]
    
    # Zignoruj padding i specjalne tokeny
    tokens = tokens[:len(attention_mask)]
    labels_seq = labels_seq[:len(attention_mask)]
    
    # Drukowanie oryginalnego tekstu
    print(f"Tekst {i+1}: {original_text}")
    
    # Drukowanie tokenów z etykietami
    for token, label in zip(tokens, labels_seq):
        print(f"{token}\t{label}")
    print("\n" + "-"*30 + "\n")

# Konwersja entity_stats do słownika
counts_dict = dict(entity_stats)

# Obliczanie wag klas
class_weights = []
for tag in tag2id.keys():
    if tag == 'O':
        weight = 1.0
    else:
        # Wyodrębnienie nazwy klasy encji bez prefiksów 'B-' i 'I-'
        entity_type = tag.split('-', 1)[1]
        count = counts_dict.get(entity_type, 0)
        weight = 1.0 / (count + 1e-6)  # Dodanie epsilon, aby uniknąć dzielenia przez zero
    class_weights.append(weight)

# Normalizacja wag
class_weights = np.array(class_weights)
class_weights = class_weights / class_weights.sum() * len(tag2id)

# Konwersja do tensorów
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Definicja funkcji metryk
def compute_metrics(p):
    predictions, labels = p.predictions, p.label_ids
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for pred, label in zip(predictions, labels):
        temp_true = []
        temp_pred = []
        for p, l in zip(pred, label):
            if l != -100:
                temp_true.append(id2tag.get(l, 'O'))
                temp_pred.append(id2tag.get(p, 'O'))
        if temp_true:
            true_labels.append(temp_true)
            pred_labels.append(temp_pred)

    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Definicja Custom Trainer, aby uwzględnić class_weights
class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# Definicja modelu
model = AutoModelForTokenClassification.from_pretrained(
    'allegro/herbert-large-cased',
    num_labels=len(tag2id)
)
model.to(device)

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    save_total_limit=3,
)

# Inicjalizacja CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

# Trening modelu
trainer.train()

# Zapisanie najlepszego modelu
save_directory = "D:/Nowa_praca/adnotacje_spubi/model/"
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)

# Zapisanie tag2id do pliku JSON
tag2id_path = "D:/Nowa_praca/adnotacje_spubi/model/tag2id.json"
with open(tag2id_path, 'w') as f:
    json.dump(tag2id, f)