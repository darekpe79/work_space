# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:39:49 2025

@author: darek
"""


import json
from transformers import AutoModelForTokenClassification, HerbertTokenizerFast, pipeline

# Ścieżki do modelu i pliku tag2id
model_path = "C:/Users/darek/model_output/best_model/"
tag2id_path = "C:/Users/darek/model_output/best_model/tag2id.json"
model_path = "D:/Nowa_praca/adnotacje_spubi/best_model/"
tag2id_path = "D:/Nowa_praca/adnotacje_spubi/best_model/tag2id.json"

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
device = 0 
# Tworzenie pipeline NER
nlp = pipeline(
    "ner", 
    model=model, 
    tokenizer=tokenizer, 
    aggregation_strategy="simple", 
    device=device
)

# Przykładowy tekst
text = '''
[w ks.:] Tadeusz Sułkowski, Maria Dąbrowska: Listy 1943-1959. Oprac., wstępem i przypisami opatrzyła Ewa Głębicka. Skierniewice 2007 s. 67, 69-70 [do Marii Dąbrowskiej dot. śmierci Tadeusza Sułkowskiego (2 z 1960)
'''

# Analiza tekstu za pomocą pipeline
results = nlp(text)

# Wyświetlenie wyników
for entity in results:
    print(f"Tekst: {entity['word']}, Etykieta: {entity['entity_group']}, Skala pewności: {entity['score']:.2f}")