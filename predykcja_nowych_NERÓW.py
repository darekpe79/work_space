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
    
    
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json
from tqdm import tqdm

# Ścieżka do modelu
model_path = "C:/Users/dariu/model_NER/"

# Załaduj model i tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Załaduj tag2id i utwórz id2tag
tag2id_path = f"{model_path}/tag2id.json"
with open(tag2id_path, 'r') as f:
    tag2id = json.load(f)
id2tag = {int(v): k for k, v in tag2id.items()}

def nlp1(text):
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True,
        is_split_into_words=False
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0]
    word_ids = encoding.word_ids()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    ner_results = []
    
    for idx, (pred_id, word_id) in enumerate(zip(predictions, word_ids)):
        if word_id is None:
            continue  # Pomijamy tokeny specjalne
        
        token = tokens[idx]
        tag_id = pred_id
        tag = id2tag.get(tag_id, 'O')
        start = int(offset_mapping[idx][0])
        end = int(offset_mapping[idx][1])
        
        ner_results.append({
            'word': token,
            'entity': tag,
            'start': start,
            'end': end
        })
    
    return ner_results

def combine_tokens(ner_results):
    combined_entities = []
    current_entity = None

    for token in ner_results:
        tag = token['entity']
        word = token['word']
        score = token['score']
        start = token['start']
        end = token['end']

        if tag.startswith('B-'):
            if current_entity:
                combined_entities.append(current_entity)
            current_entity = {
                'entity': tag[2:],
                'word': tokenizer.convert_tokens_to_string([word]),
                'start': start,
                'end': end,
                'score_sum': score,
                'token_count': 1
            }
        elif tag.startswith('I-') and current_entity and current_entity['entity'] == tag[2:]:
            current_entity['word'] += tokenizer.convert_tokens_to_string([word])
            current_entity['end'] = end
            current_entity['score_sum'] += score
            current_entity['token_count'] += 1
        else:
            if current_entity:
                combined_entities.append(current_entity)
                current_entity = None

    if current_entity:
        combined_entities.append(current_entity)

    for entity in combined_entities:
        entity['score'] = entity['score_sum'] / entity['token_count']
        del entity['score_sum']
        del entity['token_count']

    return combined_entities

def ner_on_long_text(text):
    max_length = 510
    tokens = tokenizer.tokenize(text)
    token_fragments = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]

    ner_results = []
    offset = 0
    for fragment in fragments:
        fragment_ner_results = nlp1(fragment)
        # Aktualizacja offsetów
        for entity in fragment_ner_results:
            entity['start'] += offset
            entity['end'] += offset
        ner_results.extend(fragment_ner_results)
        offset += len(fragment)

    combined_entities = combine_tokens(ner_results)
    return combined_entities

# Przykładowy tekst
sample_text = "Wczoraj czytałem książkę 'Pan Tadeusz' Adama Mickiewicza i oglądałem sztukę 'Dziady'."

entities = ner_on_long_text(sample_text)

print("Wyodrębnione jednostki:")
for entity in entities:
    print(f"Typ: {entity['entity']}, Tekst: {entity['word']}, Start: {entity['start']}, End: {entity['end']}, Score: {entity['score']:.4f}")

