import json
import numpy as np
from pymarc import MARCReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Ustawienia
MARC_FILE_PATH = "D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc"
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Funkcja ekstrakcji danych z MARC21
def extract_marc_data(file_path, limit=100):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Processing MARC records", unit="record")):
            if i >= limit:
                break
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            description = record.get_fields('520')
            description = description[0]['a'] if description and 'a' in description[0] else "Brak opisu"
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else "Nieznany autor"

            if title:
                records.append({
                    "text": f"Tytuł: {title}. Autor: {author}. Tematy: {', '.join(subjects) if subjects else 'Brak tematyki'}. Opis: {description}",
                    "title": title,
                    "author": author,
                    "subjects": subjects
                })
    return records

# Funkcja przetwarzania danych na wektory z podziałem na pola
def process_to_vectors(records, embedding_model):
    vector_data = []
    for record in records:
        full_text_vector = embedding_model.encode(record['text']).astype('float32')
        title_vector = embedding_model.encode(record['title']).astype('float32')
        author_vector = embedding_model.encode(record['author']).astype('float32')
        subjects_vector = embedding_model.encode(", ".join(record['subjects'])).astype('float32') if record['subjects'] else np.zeros_like(full_text_vector)

        vector_data.append({
            "full_text_vector": full_text_vector.tolist(),
            "title_vector": title_vector.tolist(),
            "author_vector": author_vector.tolist(),
            "subjects_vector": subjects_vector.tolist(),
            "metadata": record
        })
    return vector_data

# Budowa indeksu FAISS
def build_faiss_index(vector_data, vector_key):
    dimension = len(vector_data[0][vector_key])
    index = faiss.IndexFlatIP(dimension)
    embeddings = np.array([item[vector_key] for item in vector_data]).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalizacja
    index.add(embeddings)
    return index

# Funkcja wyszukiwania
def search(query, embedding_model, index, vector_data, top_k=3):
    query_vector = embedding_model.encode(query).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)  # Normalizacja zapytania

    # Wyszukiwanie w indeksie FAISS
    distances, indices = index.search(np.array([query_vector]), top_k)
    results = [
        {"metadata": vector_data[idx]["metadata"], "score": distances[0][i]}
        for i, idx in enumerate(indices[0])
    ]
    return results

# Główna logika
print("Wczytywanie danych z MARC21...")
marc_records = extract_marc_data(MARC_FILE_PATH, limit=100)
print("Generowanie wektorów...")
vector_data = process_to_vectors(marc_records, embedding_model)

# Tworzenie indeksów FAISS
print("Budowa indeksów FAISS...")
full_text_index = build_faiss_index(vector_data, "full_text_vector")
title_index = build_faiss_index(vector_data, "title_vector")
author_index = build_faiss_index(vector_data, "author_vector")
subjects_index = build_faiss_index(vector_data, "subjects_vector")

# Test wyszukiwania
query = "Gry komputerowe"
results = search(query, embedding_model, subjects_index, vector_data)

print("Wyniki wyszukiwania:")
for result in results:
    metadata = result['metadata']
    print(f"Tytuł: {metadata['title']}, Autor: {metadata['author']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")

# Opcjonalna ewaluacja z większą liczbą zapytań
test_queries = [
    {"query": "Gry komputerowe", "focus": "subjects"},
    {"query": "Checkpointy", "focus": "title"},
    {"query": "Rowling, J. K.", "focus": "author"},
    {"query": "II wojna światowa", "focus": "subjects"},
    {"query": "Literatura polska", "focus": "subjects"}
]

print("\nEwaluacja zapytań:")
for test in test_queries:
    query = test["query"]
    focus = test["focus"]
    index = {
        "full_text": full_text_index,
        "title": title_index,
        "author": author_index,
        "subjects": subjects_index
    }.get(focus, full_text_index)
    results = search(query, embedding_model, index, vector_data)
    print(f"Pytanie: {query} (Focus: {focus})")
    for result in results:
        metadata = result['metadata']
        print(f"Tytuł: {metadata['title']}, Autor: {metadata['author']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")
    print("-" * 50)

# Zapis do JSON (opcjonalny)
with open('vector_database.json', 'w', encoding='utf-8') as f:
    json.dump(vector_data, f, ensure_ascii=False, indent=4)





#%%
# RAG do wytrenowanego modelu


import json
import numpy as np
from pymarc import MARCReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Ustawienia
MARC_FILE_PATH = "D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc"
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Funkcja ekstrakcji danych z MARC21
def extract_marc_data(file_path, limit=100):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Processing MARC records", unit="record")):
            if i >= limit:
                break
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            description = record.get_fields('520')
            description = description[0]['a'] if description and 'a' in description[0] else "Brak opisu"
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else "Nieznany autor"

            if title:
                records.append({
                    "text": f"Tytuł: {title}. Autor: {author}. Tematy: {', '.join(subjects) if subjects else 'Brak tematyki'}. Opis: {description}",
                    "title": title,
                    "author": author,
                    "subjects": subjects
                })
    return records

# Funkcja przetwarzania danych na wektory
def process_to_vectors(records, embedding_model):
    vector_data = []
    for record in records:
        embedding = embedding_model.encode(record['text'])  # Zamiana tekstu na wektor
        vector_data.append({"vector": embedding.tolist(), "metadata": record})
    return vector_data

# Budowa indeksu FAISS
def build_faiss_index(vector_data):
    dimension = len(vector_data[0]['vector'])
    index = faiss.IndexFlatIP(dimension)
    embeddings = np.array([item['vector'] for item in vector_data]).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalizacja
    index.add(embeddings)
    return index

# Funkcja wyszukiwania
def search(query, embedding_model, index, vector_data, top_k=3, focus="text"):
    """
    Wyszukuje w bazie FAISS, koncentrując się na określonym polu.

    Args:
        query (str): Pytanie użytkownika.
        embedding_model (SentenceTransformer): Model do kodowania zapytań.
        index (faiss.IndexFlatIP): Indeks FAISS.
        vector_data (list): Dane wektorowe.
        top_k (int): Liczba wyników.
        focus (str): Pole do przeszukiwania ("text", "subjects", "title").

    Returns:
        list: Lista wyników z metadanymi i oceną podobieństwa.
    """
    # Przygotowanie zapytania
    query_text = ""
    if focus == "subjects":
        query_text = f"Tematy: {query}"
    elif focus == "title":
        query_text = f"Tytuł: {query}"
    else:
        query_text = query

    # Kodowanie zapytania
    query_vector = embedding_model.encode(query_text).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)  # Normalizacja zapytania

    # Wyszukiwanie w indeksie FAISS
    distances, indices = index.search(np.array([query_vector]), top_k)
    results = [
        {"metadata": vector_data[idx]["metadata"], "score": distances[0][i]}
        for i, idx in enumerate(indices[0])
    ]
    return results

# Główna logika
print("Wczytywanie danych z MARC21...")
marc_records = extract_marc_data(MARC_FILE_PATH, limit=100)
print("Generowanie wektorów...")
vector_data = process_to_vectors(marc_records, embedding_model)
print("Budowa indeksu FAISS...")
index = build_faiss_index(vector_data)

# Test wyszukiwania
query = "Jakie książki dotyczą gier komputerowych?"
results = search(query, embedding_model, index, vector_data, focus="subjects")

print("Wyniki wyszukiwania:")
for result in results:
    metadata = result['metadata']
    print(f"Tytuł: {metadata['title']}, Autor: {metadata['author']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")

# Opcjonalna ewaluacja z większą liczbą zapytań
test_queries = [
    {"query": "Gry komputerowe", "focus": "subjects"},
    {"query": "Checkpointy", "focus": "title"},
    {"query": "Rowling, J. K.", "focus": "author"},
    {"query": "II wojna światowa", "focus": "subjects"},
    {"query": "Literatura polska", "focus": "subjects"}
]

print("\nEwaluacja zapytań:")
for test in test_queries:
    query = test["query"]
    focus = test["focus"]
    results = search(query, embedding_model, index, vector_data, focus=focus)
    print(f"Pytanie: {query} (Focus: {focus})")
    for result in results:
        metadata = result['metadata']
        print(f"Tytuł: {metadata['title']}, Autor: {metadata['author']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")
    print("-" * 50)

# Zapis do JSON (opcjonalny)
with open('vector_database.json', 'w', encoding='utf-8') as f:
    json.dump(vector_data, f, ensure_ascii=False, indent=4)


#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 2024
@author: darek
"""
import json
import numpy as np
from pymarc import MARCReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Ustawienia
MARC_FILE_PATH = "D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc"
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Funkcja ekstrakcji danych z MARC21
def extract_marc_data(file_path, limit=100):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Processing MARC records", unit="record")):
            if i >= limit:
                break
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            description_field = record.get_fields('520')
            description = description_field[0]['a'] if description_field and 'a' in description_field[0] else "Brak opisu"
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else "Nieznany autor"
            
            if title:
                records.append({
                    "text": f"Tytuł: {title}. Autor: {author}. Tematy: {', '.join(subjects) if subjects else 'Brak tematyki'}. Opis: {description}",
                    "title": title,
                    "author": author,
                    "subjects": subjects
                })
    return records

# Funkcja przetwarzania danych na wektory
def process_to_vectors(records, embedding_model):
    vector_data = []
    for record in records:
        embedding = embedding_model.encode(record['text'])  # Zamiana tekstu na wektor
        vector_data.append({"vector": embedding.tolist(), "metadata": record})
    return vector_data

# Budowa indeksu FAISS
def build_faiss_index(vector_data):
    dimension = len(vector_data[0]['vector'])
    index = faiss.IndexFlatIP(dimension)
    embeddings = np.array([item['vector'] for item in vector_data]).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalizacja
    index.add(embeddings)
    return index

# Funkcja wyszukiwania
def search(query, embedding_model, index, vector_data, top_k=3):
    query_vector = embedding_model.encode(query).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)  # Normalizacja zapytania
    distances, indices = index.search(np.array([query_vector]), top_k)
    results = [
        {"metadata": vector_data[idx]["metadata"], "score": distances[0][i]}
        for i, idx in enumerate(indices[0])
    ]
    return results

# Główna logika
print("Wczytywanie danych z MARC21...")
marc_records = extract_marc_data(MARC_FILE_PATH, limit=100)
print("Generowanie wektorów...")
vector_data = process_to_vectors(marc_records, embedding_model)
print("Budowa indeksu FAISS...")
index = build_faiss_index(vector_data)

# Test wyszukiwania
query = "Jakie książki dotyczą gier komputerowych?"
results = search(query, embedding_model, index, vector_data)

print("Wyniki wyszukiwania:")
for result in results:
    metadata = result['metadata']
    print(f"Tytuł: {metadata['title']}, Autor: {metadata['author']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")

# Opcjonalna ewaluacja z większą liczbą zapytań
test_queries = [
    "Jakie książki dotyczą teatru polskiego?",
    "Książki o inscenizacjach i teatrze?",
    "Jakie książki opisują I wojnę światową?",
    "Książki o wojnach i polityce?",
    "Czy są książki o katolicyzmie i architekturze sakralnej?",
    "Literatura o II wojnie światowej?",
    "Książki o Żydach w okresie okupacji niemieckiej?",
    "Czy są książki o Papieżach i świętych?",
    "Książki o muzeach i zbiorach muzealnych?",
    "Jakie książki dotyczą literatury polskiej i insurekcji kościuszkowskiej?"
]

# Ewaluacja test queries
print("\nEwaluacja zapytań:")
for query in test_queries:
    results = search(query, embedding_model, index, vector_data)
    print(f"Pytanie: {query}")
    for result in results:
        metadata = result['metadata']
        print(f"Tytuł: {metadata['title']}, Autor: {metadata['author']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")
    print("-" * 50)

# Zapis do JSON (opcjonalny)
with open('vector_database.json', 'w', encoding='utf-8') as f:
    json.dump(vector_data, f, ensure_ascii=False, indent=4)
    
    
    
#%%
import json
import numpy as np
from pymarc import MARCReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Ustawienia
MARC_FILE_PATH = "D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc"
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Funkcja ekstrakcji danych z MARC21
def extract_marc_data(file_path, limit=100):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Processing MARC records", unit="record")):
            if i >= limit:
                break
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            description = record.get_fields('520')
            description = description[0]['a'] if description and 'a' in description[0] else "Brak opisu"
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else "Nieznany autor"

            if title:
                records.append({
                    "text": f"Tytuł: {title}. Autor: {author}. Tematy: {', '.join(subjects) if subjects else 'Brak tematyki'}. Opis: {description}",
                    "title": title,
                    "author": author,
                    "subjects": subjects
                })
    return records

# Funkcja przetwarzania danych na wektory
def process_to_vectors(records, embedding_model):
    vector_data = []
    for record in records:
        embedding = embedding_model.encode(record['text'])  # Zamiana tekstu na wektor
        vector_data.append({"vector": embedding.tolist(), "metadata": record})
    return vector_data

# Budowa indeksu FAISS
def build_faiss_index(vector_data):
    dimension = len(vector_data[0]['vector'])
    index = faiss.IndexFlatIP(dimension)
    embeddings = np.array([item['vector'] for item in vector_data]).astype('float32')
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalizacja
    index.add(embeddings)
    return index

# Funkcja wyszukiwania
def search(query, embedding_model, index, vector_data, top_k=3, focus="text"):
    """
    Wyszukuje w bazie FAISS, koncentrując się na określonym polu.

    Args:
        query (str): Pytanie użytkownika.
        embedding_model (SentenceTransformer): Model do kodowania zapytań.
        index (faiss.IndexFlatIP): Indeks FAISS.
        vector_data (list): Dane wektorowe.
        top_k (int): Liczba wyników.
        focus (str): Pole do przeszukiwania ("text", "subjects", "title").

    Returns:
        list: Lista wyników z metadanymi i oceną podobieństwa.
    """
    # Przygotowanie zapytania
    query_text = ""
    if focus == "subjects":
        query_text = f"Tematy: {query}"
    elif focus == "title":
        query_text = f"Tytuł: {query}"
    else:
        query_text = query

    # Kodowanie zapytania
    query_vector = embedding_model.encode(query_text).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)  # Normalizacja zapytania

    # Wyszukiwanie w indeksie FAISS
    distances, indices = index.search(np.array([query_vector]), top_k)
    results = [
        {"metadata": vector_data[idx]["metadata"], "score": distances[0][i]}
        for i, idx in enumerate(indices[0])
    ]
    return results

# Główna logika
print("Wczytywanie danych z MARC21...")
marc_records = extract_marc_data(MARC_FILE_PATH, limit=100)
print("Generowanie wektorów...")
vector_data = process_to_vectors(marc_records, embedding_model)
print("Budowa indeksu FAISS...")
index = build_faiss_index(vector_data)

# Test wyszukiwania
query = "Jakie książki dotyczą gier komputerowych?"
results = search(query, embedding_model, index, vector_data, focus="subjects")

print("Wyniki wyszukiwania:")
for result in results:
    metadata = result['metadata']
    print(f"Tytuł: {metadata['title']}, Autor: {metadata['author']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")

# Opcjonalna ewaluacja z większą liczbą zapytań
test_queries = [
    {"query": "Gry komputerowe", "focus": "subjects"},
    {"query": "Checkpointy", "focus": "title"},
    {"query": "Rowling, J. K.", "focus": "author"},
    {"query": "II wojna światowa", "focus": "subjects"},
    {"query": "Literatura polska", "focus": "subjects"}
]

print("\nEwaluacja zapytań:")
for test in test_queries:
    query = test["query"]
    focus = test["focus"]
    results = search(query, embedding_model, index, vector_data, focus=focus)
    print(f"Pytanie: {query} (Focus: {focus})")
    for result in results:
        metadata = result['metadata']
        print(f"Tytuł: {metadata['title']}, Autor: {metadata['author']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")
    print("-" * 50)

# Zapis do JSON (opcjonalny)
with open('vector_database.json', 'w', encoding='utf-8') as f:
    json.dump(vector_data, f, ensure_ascii=False, indent=4)


