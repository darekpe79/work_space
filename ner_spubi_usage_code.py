# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:58:04 2025

@author: darek
"""

import json
from transformers import AutoModelForTokenClassification, HerbertTokenizerFast, pipeline

# Ścieżki do modelu i pliku tag2id
model_path = "C:/Users/darek/model_output/best_model/"
tag2id_path = "C:/Users/darek/model_output/best_model/tag2id.json"

# Ładowanie modelu
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Ładowanie tokenizatora
tokenizer = HerbertTokenizerFast.from_pretrained(model_path)

# Załadowanie mapowania tagów z pliku JSON
with open(tag2id_path, "r", encoding="utf-8") as f:
    tag2id = json.load(f)

# Utworzenie odwróconego mapowania id2tag
id2tag = {v: k for k, v in tag2id.items()}

# Przypisanie mapowania do konfiguracji modelu
model.config.label2id = tag2id
model.config.id2label = id2tag

# Tworzenie pipeline NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Przykładowy tekst
text = '''
[w ks.:] Janusz Tazbir: Pożegnanie z XX wiekiem. Warszawa 1999, s. 152-156 [m.in. nt. roli powieści w recepcji \"Ogniem i mieczem\" Henryka Sienkiewicza
'''

# Analiza tekstu za pomocą pipeline
results = nlp(text)

# Wyświetlenie wyników
for entity in results:
    print(f"Tekst: {entity['word']}, Etykieta: {entity['entity_group']}, Skala pewności: {entity['score']:.2f}")