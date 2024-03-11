# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:56:22 2024

@author: dariu
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer
import json

# Ścieżka do katalogu, w którym zapisany jest model i tokenizator
model_directory = "C:/Users/dariu/model_NER/"

# Ścieżka do pliku JSON z mapowaniem tag2id
tag2id_path = "C:/Users/dariu/model_NER/tag2id.json"

# Ładowanie modelu
model = AutoModelForTokenClassification.from_pretrained(model_directory)

# Ładowanie tokenizatora
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Ładowanie mapowania tag2id
with open(tag2id_path, 'r') as f:
    tag2id = json.load(f)

# Odwrócenie mapowania tag2id na id2tag dla dekodowania predykcji
id2tag = {v: k for k, v in tag2id.items()}

import torch
import numpy as np

def predict_ner(text, model, tokenizer, id2tag):
    # Tokenizacja tekstu
    tokenized_input = tokenizer.encode_plus(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    
    input_ids = tokenized_input["input_ids"]
    
    # Predykcja modelu
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Dekodowanie etykiet
    predictions = torch.argmax(outputs.logits, dim=2)
    
    # Konwersja predykcji na listę etykiet
    predicted_labels = [id2tag[label_id.item()] for label_id in predictions[0]]
    
    # Pobranie tokenów
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Zwrócenie sparowanych tokenów i etykiet
    return [(token, label) for token, label in zip(tokens, predicted_labels) if token not in tokenizer.all_special_tokens]

# Przykładowy tekst do analizy
text = ' Spektakl "Makbet" na podstawie dramatu Szekspira "MAKBET"'

# Użycie funkcji
predicted_ner = predict_ner(text, model, tokenizer, id2tag)

# Wyświetlenie wyników
for token, label in predicted_ner:
    print(f"{token}: {label}")
