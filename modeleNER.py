# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:09:59 2024

@author: dariu
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Załaduj tokenizer i model
model_name = "Babelscape/wikineural-multilingual-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Ustawienia pipeline z agregacją
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Dłuższy tekst
text = "Dłuższy tekst, który chcesz przetworzyć... (tutaj wstaw pełny tekst)"

# Tokenizowanie tekstu
tokens = tokenizer.tokenize(text)
max_tokens = 514  # Przykładowy limit tokenów

# Dzielimy tekst na fragmenty, które mieszczą się w limicie tokenów
token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]

# Przetwarzanie każdego fragmentu osobno
for i, fragment in enumerate(fragments):
    results = ner_pipeline(fragment)
    print(f"--- Wyniki dla fragmentu {i+1} ---")
    for entity in results:
        print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Confidence: {entity['score']:.2f}")
#%%
import spacy

# Załaduj model spaCy pl_core_news_lg
nlp = spacy.load("pl_core_news_lg")

# Przykładowy tekst w języku polskim
text = "Jan Kowalski pracuje w OpenAI w Warszawie, 10 października 2021."
doc = nlp(text)

# Wyświetlenie rozpoznanych encji, w tym dat
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")


#%%
from deeppavlov import build_model, configs

# Załaduj model Slavic BERT NER
ner_model = build_model("deeppavlov/configs/ner/ner_bert_slav.json", download=True)

# Przykładowy tekst w języku polskim
text = ["To Bert z ulicy Sezamkowej"]
predictions = ner_model(text)

# Wyniki: tokeny i ich etykiety
tokens = predictions[0][0]
labels = predictions[1][0]

# Funkcja agregująca etykiety w całościowe encje
def aggregate_entities(tokens, labels):
    entities = []
    current_entity = []
    current_label = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):  # Begin of an entity
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
            current_entity = [token]
            current_label = label[2:]
        elif label.startswith("I-") and current_label == label[2:]:  # Inside the same entity
            current_entity.append(token)
        else:  # Outside any entity
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
            current_entity = []
            current_label = None

    # Dodaj ostatnią encję
    if current_entity:
        entities.append((" ".join(current_entity), current_label))
    
    return entities

# Wywołanie funkcji agregującej
aggregated_entities = aggregate_entities(tokens, labels)
for entity, label in aggregated_entities:
    print(f"Entity: {entity}, Label: {label}")
    
    
    
    
    
from transformers import AutoModel, AutoTokenizer

model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Przykładowy tekst w języku polskim
text = "Zażółcić gęślą jaźń."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state)  # Wynik modelu
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Ładowanie modelu XLM-RoBERTa dla NER
model_name = "xlm-roberta-large-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Tworzenie pipeline do NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Przykładowy tekst
text = "John Doe pracuje w OpenAI w San Francisco, 10 października 2021."
results = ner_pipeline(text)
print(results)
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Ładowanie tokenizera i modelu WikiNEuRal
tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

# Tworzenie pipeline NER z ustawioną strategią agregacji
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Przykładowy tekst w języku polskim
text = "Jan Kowalski pracuje w OpenAI w Warszawie, 10 października 2021."
results = ner_pipeline(text)

from flair.models import SequenceTagger
from flair.data import Sentence

# Ładowanie modelu wielojęzycznego do NER z datami
tagger = SequenceTagger.load("flair/ner-multi")

# Przykładowy tekst
sentence = Sentence("John Doe pracuje w OpenAI w San Francisco, 10 października 2021.")
tagger.predict(sentence)

# Wyświetlenie wyników
for entity in sentence.get_spans("ner"):
    print(f"Entity: {entity.text}, Label: {entity.tag}, Confidence: {entity.score:.2f}")