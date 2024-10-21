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