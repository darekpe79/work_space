# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:52:41 2024

@author: dariu
"""

import spacy
import gensim
from gensim import corpora
from pprint import pprint

# Wczytywanie modelu języka polskiego do spaCy
nlp = spacy.load("pl_core_news_sm")

# Przykładowe dokumenty tekstowe w języku polskim
documents = [
    "Uczenie maszynowe jest przyszłością sztucznej inteligencji",
    "Przetwarzanie języka naturalnego to gałąź sztucznej inteligencji",
    "Modelowanie tematów to technika wykorzystywana w przetwarzaniu języka naturalnego",
    "Python to popularny język programowania w dziedzinie nauki danych",
    "Nauka danych polega na analizie i interpretacji złożonych zbiorów danych"
]

# Tokenizacja dokumentów za pomocą spaCy
tokenized_documents = []
for document in documents:
    doc = nlp(document)
    tokenized_document = [token.text.lower() for token in doc]
    tokenized_documents.append(tokenized_document)

# Tworzenie słownika
dictionary = corpora.Dictionary(tokenized_documents)

# Tworzenie korpusu
corpus = [dictionary.doc2bow(document) for document in tokenized_documents]

# Wywołanie algorytmu LDA
lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary)

# Wyświetlenie tematów
print("Tematy:")
pprint(lda_model.print_topics())

# Przewidywanie tematów dla nowego dokumentu
new_document = "Python i uczenie maszynowe są ze sobą ściśle powiązane"
new_document_doc = nlp(new_document)
new_document_tokens = [token.text.lower() for token in new_document_doc]
new_document_bow = dictionary.doc2bow(new_document_tokens)
print("\nRozkład tematów dla nowego dokumentu:")
pprint(lda_model.get_document_topics(new_document_bow))
