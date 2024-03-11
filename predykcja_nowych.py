# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:56:22 2024

@author: dariu
"""

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json

# Ścieżka do zapisanego modelu i tokenizera
model_directory = "C:/Users/User/Desktop/model_NER2"

# Załadowanie modelu
model = AutoModel.from_pretrained(model_directory)

# Załadowanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Ścieżka do pliku tag2id.json
tag2id_path = "C:/Users/User/Desktop/model_NER/tag2id.json"

# Załadowanie tag2id
with open(tag2id_path, 'r') as f:
    tag2id = json.load(f)

# Twoja funkcja predict_ner
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

# Użycie funkcji
text = '''23 października klasa 7 wybrała się do Teatru Cracovia na spektakl „Balladyna”. Czy warto obejrzeć tę sztukę? Przeczytajcie opinie siódmoklasistów:

Spektakl Balladyna

Dnia 23 października 2019 roku w Teatrze Cracovia na terenie Centrum Kultury Solvay odbył się spektakl pod tytułem Balladyna na podstawie dramatu Juliusz Słowackiego o takim samym tytule. Jest on prezentowany od 2008 roku. Ma obecną formę dzięki reżyserii i scenografii Annę Kasprzyk.'''

predict_ner(text, model, tokenizer, tag2id)_