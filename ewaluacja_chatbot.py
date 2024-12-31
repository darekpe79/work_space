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
            if title:
                records.append({
                    "text": f"{title}. Tematy: {', '.join(subjects) if subjects else 'Brak tematyki'}.",
                    "title": title,
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
    print(f"Tytuł: {metadata['title']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")

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
            
            # Pobranie opisu (jeśli istnieje)
            description_field = record.get_fields('520')
            description = description_field[0]['a'] if description_field and 'a' in description_field[0] else "Brak opisu"
            
            if title:
                records.append({
                    "text": f"{title}. Tematy: {', '.join(subjects) if subjects else 'Brak tematyki'}. Opis: {description}",
                    "title": title,
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
    print(f"Tytuł: {metadata['title']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")

# Opcjonalna ewaluacja z większą liczbą zapytań
test_queries = [
    "Jakie książki dotyczą teatru polskiego?",  # Dopasowanie do "Historia polskiego kabaretu /"
    "Książki o inscenizacjach i teatrze?",  # Dopasowanie do "Hamlet.pl :"
    "Jakie książki opisują I wojnę światową?",  # Dopasowanie do "Konwent i szpital bonifratrów w Cieszynie w latach I wojny światowej :"
    "Książki o wojnach i polityce?",  # Dopasowanie do "Rajnolda Hejdensztejna sekretarza królewskiego, dzieje Polski od śmierci Zygmunta Augusta do roku 1594 :"
    "Czy są książki o katolicyzmie i architekturze sakralnej?",  # Dopasowanie do "Kronika ks. prałata Michała Winiarza 1957-1970 /"
    "Literatura o II wojnie światowej?",  # Dopasowanie do "Zaraz po wojnie /"
    "Książki o Żydach w okresie okupacji niemieckiej?",  # Dopasowanie do "Kronika zamordowanego świata :"
    "Czy są książki o Papieżach i świętych?",  # Dopasowanie do "Pontifex Maximus :"
    "Książki o muzeach i zbiorach muzealnych?",  # Dopasowanie do "Najjaśniejszym :"
    "Jakie książki dotyczą literatury polskiej i insurekcji kościuszkowskiej?"  # Dopasowanie do "Język publicystyki okresu powstania kościuszkowskiego /"
]

# Ewaluacja test queries
print("\nEwaluacja zapytań:")
for query in test_queries:
    results = search(query, embedding_model, index, vector_data)
    print(f"Pytanie: {query}")
    for result in results:
        metadata = result['metadata']
        print(f"Tytuł: {metadata['title']}, Tematy: {', '.join(metadata['subjects'])}, Podobieństwo: {result['score']:.2f}")
    print("-" * 50)

# Zapis do JSON (opcjonalny)
with open('vector_database.json', 'w', encoding='utf-8') as f:
    json.dump(vector_data, f, ensure_ascii=False, indent=4)

