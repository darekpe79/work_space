# -*- coding: utf-8 -*-
"""
Kompletny przykład RAG, w którym każdy 'subject' (fraza tematyczna)
otrzymuje oddzielny wektor w indeksie FAISS.
"""

import os
import re
import unicodedata
import numpy as np
import faiss
from tqdm import tqdm
from pymarc import MARCReader
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ===================================================================
#                       KONFIGURACJA / ŚCIEŻKI
# ===================================================================

MARC_FILE_PATH = "D:/Nowa_praca/marki_po_updatach 2025,2024/NEW-marc_bn_books_08-02-2024.mrc"
MODEL_EMBEDDING_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"  # lub inny SentenceTransformer
MODEL_BIELIK_NAME = "speakleash/Bielik-11B-v2.3-Instruct"


# ===================================================================
#             1. Normalizacja tekstu (lower-case, bez diakrytyków)
# ===================================================================

def normalize_text(text: str) -> str:
    """
    Zmienia tekst na lower-case, usuwa polskie znaki diakrytyczne (ą,ę,ł,ś itp.)
    przy pomocy unicodedata, i przycina spacje.
    """
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.strip()


# ===================================================================
#             2. Ładowanie modelu do embeddingów
# ===================================================================

print("Ładowanie modelu do embeddingów...")
embedding_model = SentenceTransformer(MODEL_EMBEDDING_NAME)


# ===================================================================
#      3. Wczytywanie danych z MARC i tworzenie listy rekordów
# ===================================================================

def extract_marc_data(file_path, limit=100):
    """
    Wczytuje plik MARC, zwraca listę rekordów (dict).
    limit=100 to limit wczytywanych rekordów dla przykładu.
    """
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Processing MARC records", unit="record")):
            if i >= limit:
                break

            # Przykładowe pobranie wybranych pól:
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            description = record.get_fields('520')
            description = description[0]['a'] if description and 'a' in description[0] else "Brak opisu"
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else "Nieznany autor"

            # Zabezpieczenie przed brakiem tytułu:
            if title:
                record_dict = {
                    "title": title,
                    "author": author,
                    "subjects": subjects,   # lista fraz
                    "description": description
                }
                records.append(record_dict)
    return records


# ===================================================================
#   4. Tworzenie embeddingów dla (full_text, title, author) – bez subjects
# ===================================================================

def process_to_vectors(records, embedding_model):
    """
    Dla każdego rekordu tworzy wektory:
      - full_text_vector (łączymy tytuł, autora, opis, itd.)
      - title_vector
      - author_vector
    Nie tworzymy tu subjects_vector, bo każdy subject będzie osobno.
    """
    vector_data = []

    for i, record in enumerate(records):
        # Oryginalne dane
        title = record["title"]
        author = record["author"]
        description = record["description"]
        # subjectów nie ruszamy tutaj

        # Normalizujemy do embeddingu
        title_norm = normalize_text(title)
        author_norm = normalize_text(author)
        description_norm = normalize_text(description)

        full_text = (
            f"tytuł: {title_norm}. "
            f"autor: {author_norm}. "
            f"opis: {description_norm}"
        )
        full_text_vector = embedding_model.encode(full_text).astype('float32')
        title_vector = embedding_model.encode(title_norm).astype('float32')
        author_vector = embedding_model.encode(author_norm).astype('float32')

        vector_data.append({
            "id": i,
            "metadata": record,  # oryginalne dane do wyświetlania
            "full_text_vector": full_text_vector,
            "title_vector": title_vector,
            "author_vector": author_vector,
        })
    return vector_data


# ===================================================================
#   5. Budowa osobnego indeksu FAISS dla KAŻDEJ frazy subjects
# ===================================================================

def build_subjects_index(records, embedding_model):
    """
    Dla każdego rekordu i dla każdej frazy 'subject' tworzymy osobny embedding.
    Zwracamy:
      - index_subjects: indeks FAISS
      - subjects_vectors: macierz wektorów
      - subjects_metadata: lista słowników { "record_id": X, "subject": "..." }
    """
    subjects_vectors_list = []
    subjects_metadata = []

    for i, record in enumerate(records):
        subj_list = record.get("subjects", [])
        for subj in subj_list:
            subj_norm = normalize_text(subj)

            vec = embedding_model.encode(subj_norm).astype('float32')
            # Normalizujemy wektor
            vec = vec / np.linalg.norm(vec)

            subjects_vectors_list.append(vec)
            subjects_metadata.append({
                "record_id": i,
                "subject": subj  # oryginał lub znormalizowany, w zależności od potrzeb
            })

    if len(subjects_vectors_list) == 0:
        print("Brak jakichkolwiek fraz subject do zindeksowania!")
        return None, None, []

    subjects_vectors = np.array(subjects_vectors_list, dtype='float32')
    dim = subjects_vectors.shape[1]

    index_subjects = faiss.IndexFlatIP(dim)
    index_subjects.add(subjects_vectors)

    return index_subjects, subjects_vectors, subjects_metadata


# ===================================================================
#       6. Budowa indeksu FAISS dla full_text/title/author
# ===================================================================

def build_faiss_index(vector_data, vector_key="full_text_vector"):
    """
    Tworzy płaski indeks FAISS na bazie wybranego klucza w vector_data.
    Normalizuje wektory L2 przed dodaniem do indeksu.
    Zwraca (index, embeddings).
    """
    dimension = len(vector_data[0][vector_key])
    index = faiss.IndexFlatIP(dimension)

    embeddings = []
    for item in vector_data:
        v = item[vector_key]
        v = v / np.linalg.norm(v)  # normalizacja L2
        embeddings.append(v)

    embeddings = np.array(embeddings, dtype='float32')
    index.add(embeddings)
    return index, embeddings


# ===================================================================
#   7. Funkcje wyszukiwania w indeksach (subject i inne)
# ===================================================================

def search_subjects(query, embedding_model, index_subjects, subjects_metadata, records, top_k=3):
    """
    Wyszukiwanie zapytania 'query' w indeksie fraz tematycznych.
    Zwraca listę pasujących rekordów:
    [
      {
        "score": float,
        "metadata": ... (dane oryginalne rekordu),
        "subject_matched": "konkretna fraza, która się dopasowała"
      },
      ...
    ]
    """
    if index_subjects is None:
        return []

    # Normalizacja zapytania
    query_norm = normalize_text(query)
    q_vec = embedding_model.encode(query_norm).astype('float32')
    q_vec = q_vec / np.linalg.norm(q_vec)

    distances, indices = index_subjects.search(np.array([q_vec]), top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        score = distances[0][rank]

        if idx >= 0 and idx < len(subjects_metadata):
            record_id = subjects_metadata[idx]["record_id"]
            matched_subj = subjects_metadata[idx]["subject"]

            # Oryginalny rekord
            record_meta = records[record_id]
            results.append({
                "score": score,
                "metadata": record_meta,
                "subject_matched": matched_subj
            })

    return results

def search_faiss(query, embedding_model, index, vector_data, vector_key="full_text_vector", top_k=3):
    """
    Wyszukiwanie w indeksie 'full_text', 'title' lub 'author'.
    """
    query_norm = normalize_text(query)
    query_vector = embedding_model.encode(query_norm).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)

    distances, indices = index.search(np.array([query_vector]), top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        score = distances[0][rank]
        record_meta = vector_data[idx]["metadata"]
        results.append({
            "score": score,
            "metadata": record_meta
        })
    return results


# ===================================================================
#                 8. Ładowanie modelu Bielik
# ===================================================================

print("Ładowanie modelu Bielik...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_BIELIK_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_BIELIK_NAME,
    device_map="auto",        # wymaga accelerate do automatycznego mapowania GPU
    torch_dtype=torch.float16,
    trust_remote_code=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def ask_bielik(prompt: str, max_new_tokens=512) -> str:
    """
    Funkcja pomocnicza do generowania tekstu przez Bielika.
    """
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.2,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]


# ===================================================================
#  9. Prompt do interpretacji zapytania (subject/author/title)
# ===================================================================

interpret_prompt = """Jesteś asystentem bibliotecznym, który odbiera pytania użytkowników i potrafi przygotować zapytania do systemu wyszukiwania RAG.

Twoim zadaniem jest:
1. Zrozumieć, o co pyta użytkownik.
2. Określić, czy szuka książek po tytule, autorze, czy tematyce.
3. Wyodrębnij kluczowe hasło (np. jeśli to tematyka – wyodrębnij temat; jeśli to autor – nazwisko; jeśli to tytuł – tytuł).
4. Wypisać to w formacie:
   [subject: "coś"], [author: "nazwisko"], albo [title: "tytuł"]
   Możesz wypisać kilka, jeśli pytanie tego wymaga.
5. Dodaj krótkie wyjaśnienie w języku naturalnym.

Przykład:
Pytanie: "Czy znajdę książki o grach komputerowych?"
Odpowiedź:
[subject: "gry komputerowe"]
Użytkownik szuka książek związanych z grami komputerowymi.

Zachowaj tę konwencję w odpowiedzi.
"""

def interpret_query(user_query: str) -> str:
    """
    Zwraca surowy tekst od Bielika np.:
    [subject: "gry komputerowe"]
    Użytkownik szuka książek o tematyce gier komputerowych.
    """
    prompt = interpret_prompt + f"\nPytanie użytkownika: {user_query}\n"
    return ask_bielik(prompt, max_new_tokens=300)


# ===================================================================
#  10. Funkcja do tworzenia finalnej odpowiedzi (RAG) z wyników
# ===================================================================

def create_final_answer(search_results, user_query=""):
    """
    Tworzy spójną odpowiedź w języku naturalnym, uwzględniając znalezione pozycje.
    Używamy Bielika, by ładnie sformułował tekst.
    """
    if not search_results:
        results_text = "Brak wyników."
    else:
        results_text = "Znalezione pozycje:\n"
        for i, r in enumerate(search_results, start=1):
            md = r['metadata']
            title = md['title']
            author = md['author']
            subjects = md['subjects'] if md['subjects'] else []
            subjects_str = ", ".join(subjects) if subjects else "brak"

            # Jeśli mamy "subject_matched", to fajnie pokazać, która fraza
            subject_matched = r.get("subject_matched", None)
            if subject_matched:
                results_text += f"{i}) Tytuł: {title}, Autor: {author}, Tematy: {subjects_str} (Pasująca fraza: {subject_matched})\n"
            else:
                results_text += f"{i}) Tytuł: {title}, Autor: {author}, Tematy: {subjects_str}\n"

    final_prompt = f"""Użytkownik pytał: {user_query}

Oto wyniki wyszukiwania z bazy biblioteki:
{results_text}

Napisz zwięzłą i przyjazną odpowiedź dla użytkownika, uwzględniając powyższe pozycje.
"""

    return ask_bielik(final_prompt, max_new_tokens=300)


# ===================================================================
#  11. Główna funkcja RAG: interpret -> wybór indeksu -> wyszukaj -> generuj
# ===================================================================

def rag_query(user_query, 
              embedding_model,
              records,
              vector_data,
              full_text_index,
              title_index,
              author_index,
              index_subjects,
              subjects_metadata,
              top_k=3):
    """
    1) interpretacja zapytania (Bielik -> [subject: "..."] / [author: "..."] / [title: "..."])
    2) wybór indeksu FAISS
    3) wyszukiwanie
    4) generowanie finalnej odpowiedzi
    """
    # KROK 1: Interpretacja
    interpretation = interpret_query(user_query)
    print(">>> Interpretacja zapytania:\n", interpretation)

    match = re.search(r'\[(subject|author|title):\s*"([^"]+)"\]', interpretation, re.IGNORECASE)
    if not match:
        # fallback: przeszukujemy full_text
        print("Nie rozpoznano rodzaju zapytania, używam full_text_index (full_text_vector).")
        query_type = "full_text"
        query_value = user_query
    else:
        query_type = match.group(1).lower()
        query_value = match.group(2)

    # KROK 2: Wybór i wykonanie wyszukiwania
    if query_type == "subject":
        # Szukamy we frazach tematycznych
        results = search_subjects(
            query_value,
            embedding_model,
            index_subjects,
            subjects_metadata,
            records,
            top_k=top_k
        )
    elif query_type == "author":
        # Szukamy w author_index
        results = search_faiss(
            query_value,
            embedding_model,
            author_index,
            vector_data,
            vector_key="author_vector",
            top_k=top_k
        )
    elif query_type == "title":
        # Szukamy w title_index
        results = search_faiss(
            query_value,
            embedding_model,
            title_index,
            vector_data,
            vector_key="title_vector",
            top_k=top_k
        )
    else:
        # "full_text" lub fallback
        results = search_faiss(
            query_value,
            embedding_model,
            full_text_index,
            vector_data,
            vector_key="full_text_vector",
            top_k=top_k
        )

    # KROK 3: Generowanie finalnej odpowiedzi
    final_answer = create_final_answer(results, user_query=user_query)
    return final_answer


# ===================================================================
#                12. MAIN - DEMO
# ===================================================================

if __name__ == "__main__":

    # -- 1) Wczytujemy rekordy z pliku MARC --
    print("Wczytywanie danych z MARC21...")
    marc_records = extract_marc_data(MARC_FILE_PATH, limit=100)

    # -- 2) Tworzymy embeddingi (tylko full_text, title, author) --
    print("Generowanie wektorów (embeddingi) dla tytułu, autora, full_text...")
    vector_data = process_to_vectors(marc_records, embedding_model)

    # -- 3) Budujemy indeksy FAISS (full_text, title, author) --
    print("Budowa indeksów FAISS...")
    full_text_index, _ = build_faiss_index(vector_data, "full_text_vector")
    title_index, _ = build_faiss_index(vector_data, "title_vector")
    author_index, _ = build_faiss_index(vector_data, "author_vector")

    # -- 4) Budujemy indeks FRAZ TEMATYCZNYCH (osobny) --
    print("Budowa indeksu FAISS dla subjects (każda fraza osobno)...")
    index_subjects, subjects_vectors, subjects_metadata = build_subjects_index(marc_records, embedding_model)

    # -- 5) Testowe zapytanie użytkownika --
    user_question = "Czy znajdę książki o grach komputerowych?"

    # -- 6) RAG: interpretacja -> wyszukanie -> wygenerowanie odpowiedzi --
    final_answer = rag_query(
        user_question,
        embedding_model=embedding_model,
        records=marc_records,
        vector_data=vector_data,
        full_text_index=full_text_index,
        title_index=title_index,
        author_index=author_index,
        index_subjects=index_subjects,
        subjects_metadata=subjects_metadata,
        top_k=3
    )

    print("\n>>> Odpowiedź RAG:\n", final_answer)

#%%
import torch

# Sprawdzenie, czy GPU i CUDA są dostępne
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Urządzenie CUDA: {torch.cuda.get_device_name(device)}")

    # Tworzenie tensora FP16
    try:
        tensor_fp16 = torch.randn(1000, 1000, dtype=torch.float16, device=device)
        result = tensor_fp16 @ tensor_fp16  # Mnożenie macierzy w FP16
        print("FP16 operacje działają poprawnie na GPU!")
    except Exception as e:
        print(f"Problem z FP16: {e}")
else:
    print("CUDA jest niedostępne. Sprawdź konfigurację.")




#%%
# -*- coding: utf-8 -*-
"""
Przykład kompletnego kodu RAG z normalizacją tekstu, wieloma indeksami FAISS
i integracją z modelem Bielik.
"""

import os
import re
import unicodedata
import numpy as np
import faiss
from tqdm import tqdm
from pymarc import MARCReader
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ===================================================================
#                       KONFIGURACJA / ŚCIEŻKI
# ===================================================================

MARC_FILE_PATH = "D:/Nowa_praca/marki_po_updatach 2025,2024/NEW-marc_bn_books_08-02-2024.mrc"
MODEL_EMBEDDING_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"  # lub inny SentenceTransformer
MODEL_BIELIK_NAME = "speakleash/Bielik-11B-v2.3-Instruct"


# ===================================================================
#             1. Normalizacja tekstu (lower-case, bez diakrytyków)
# ===================================================================

def normalize_text(text: str) -> str:
    """
    Zmienia tekst na lower-case, usuwa polskie znaki diakrytyczne
    (np. ą,ę,ł,ś itp.) i przycina spacje.
    """
    text = text.lower()
    # Usunięcie diakrytyków za pomocą unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.strip()


# ===================================================================
#             2. Ładowanie modelu do embeddingów
# ===================================================================

print("Ładowanie modelu do embeddingów...")
embedding_model = SentenceTransformer(MODEL_EMBEDDING_NAME)


# ===================================================================
#      3. Wczytywanie danych z MARC i tworzenie listy rekordów
# ===================================================================

def extract_marc_data(file_path, limit=100):
    """
    Wczytuje plik MARC, zwraca listę rekordów (dict).
    limit=100 to limit wczytywanych rekordów dla przykładu.
    """
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Processing MARC records", unit="record")):
            if i >= limit:
                break
            # Przykładowe pobranie wybranych pól:
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            description = record.get_fields('520')
            description = description[0]['a'] if description and 'a' in description[0] else "Brak opisu"
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else "Nieznany autor"

            # Zabezpieczenie przed brakiem tytułu:
            if title:
                record_dict = {
                    "title": title,
                    "author": author,
                    "subjects": subjects,
                    "description": description
                }
                records.append(record_dict)
    return records


# ===================================================================
#         4. Tworzenie embeddingów z normalizacją
# ===================================================================

def process_to_vectors(records, embedding_model):
    """
    Dla każdego rekordu tworzy wektory:
      - full_text_vector (łączymy tytuł, autora, opis, itd.)
      - title_vector
      - author_vector
      - subjects_vector (łączenie wszystkich fraz w jeden string)
    Stosujemy normalizację do embeddingu, natomiast
    oryginały przechowujemy w 'metadata' do wyświetlania.
    """
    vector_data = []

    for i, record in enumerate(records):
        # Pobieramy oryginalne pola
        title = record["title"]
        author = record["author"]
        subjects = record["subjects"] if record["subjects"] else []
        description = record["description"]

        # Normalizujemy do embeddingu
        title_norm = normalize_text(title)
        author_norm = normalize_text(author)
        subjects_norm = [normalize_text(s) for s in subjects]
        description_norm = normalize_text(description)

        # Tworzymy "full_text" do embeddings
        full_text = (
            f"tytuł: {title_norm}. "
            f"autor: {author_norm}. "
            f"tematy: {', '.join(subjects_norm) if subjects_norm else 'brak'}. "
            f"opis: {description_norm}"
        )

        # Embeddingi
        full_text_vector = embedding_model.encode(full_text).astype('float32')
        title_vector = embedding_model.encode(title_norm).astype('float32')
        author_vector = embedding_model.encode(author_norm).astype('float32')

        # Łączymy frazy w jeden string
        if subjects_norm:
            subjects_text = ", ".join(subjects_norm)
            subjects_vector = embedding_model.encode(subjects_text).astype('float32')
        else:
            subjects_vector = np.zeros_like(full_text_vector)

        vector_data.append({
            "id": i,  
            "metadata": record,  # oryginalne dane
            "full_text_vector": full_text_vector,
            "title_vector": title_vector,
            "author_vector": author_vector,
            "subjects_vector": subjects_vector
        })
    return vector_data


# ===================================================================
#       5. Budowa indeksu FAISS (płaski, IP -> kosinus z normalizacją)
# ===================================================================

def build_faiss_index(vector_data, vector_key="full_text_vector"):
    """
    Tworzy płaski indeks FAISS na bazie wybranego klucza (np. 'full_text_vector').
    Zwraca obiekt index oraz macierz embeddings (dla ewentualnego zapisu).
    """
    dimension = len(vector_data[0][vector_key])
    index = faiss.IndexFlatIP(dimension)

    embeddings = np.array([item[vector_key] for item in vector_data]).astype('float32')
    # Normalizacja L2
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index.add(embeddings)
    return index, embeddings


# ===================================================================
#               6. Funkcja wyszukiwania w indeksie
# ===================================================================

def search_faiss(query, embedding_model, index, vector_data, vector_key="full_text_vector", top_k=3):
    """
    Zwraca listę 'top_k' pasujących rekordów z FAISS, z normalizacją zapytania.
    """
    query_norm = normalize_text(query)  # normalizacja
    query_vector = embedding_model.encode(query_norm).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)

    distances, indices = index.search(np.array([query_vector]), top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        score = distances[0][rank]
        record_meta = vector_data[idx]["metadata"]
        results.append({
            "score": score,
            "metadata": record_meta
        })
    return results


# ===================================================================
#                 7. Ładowanie modelu Bielik
# ===================================================================

print("Ładowanie modelu Bielik...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_BIELIK_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_BIELIK_NAME,
    device_map="auto",        # wymaga accelerate do automatycznego mapowania GPU
    torch_dtype=torch.float16,
    trust_remote_code=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def ask_bielik(prompt: str, max_new_tokens=512) -> str:
    """
    Funkcja pomocnicza do generowania tekstu przez Bielika.
    """
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.2,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]


# ===================================================================
#    8. Prompt do interpretacji zapytania (subject/author/title)
# ===================================================================

interpret_prompt = """Jesteś asystentem bibliotecznym, który odbiera pytania użytkowników i potrafi przygotować zapytania do systemu wyszukiwania RAG.

Twoim zadaniem jest:
1. Zrozumieć, o co pyta użytkownik.
2. Określić, czy szuka książek po tytule, autorze, czy tematyce.
3. Wyodrębnij kluczowe hasło (np. jeśli to tematyka – wyodrębnij temat; jeśli to autor – nazwisko; jeśli to tytuł – tytuł).
4. Wypisz to w formacie:
   [subject: "coś"], [author: "nazwisko"], albo [title: "tytuł"]
   Możesz wypisać kilka, jeśli pytanie tego wymaga.
5. Dodaj krótkie wyjaśnienie w języku naturalnym.

Przykład:
Pytanie: "Czy znajdę książki o grach komputerowych?"
Odpowiedź:
[subject: "gry komputerowe"]
Użytkownik szuka książek związanych z grami komputerowymi.

Zachowaj tę konwencję w odpowiedzi.
"""

def interpret_query(user_query: str) -> str:
    """
    Zwraca surowy tekst od Bielika np.:
    [subject: "gry komputerowe"]
    Użytkownik szuka książek o tematyce gier komputerowych.
    """
    prompt = interpret_prompt + f"\nPytanie użytkownika: {user_query}\n"
    return ask_bielik(prompt, max_new_tokens=300)


# ===================================================================
#  9. Funkcja tworząca finalną odpowiedź na podstawie wyników
# ===================================================================

def create_final_answer(search_results, user_query=""):
    """
    Tworzy spójną odpowiedź w języku naturalnym, z uwzględnieniem wyników.
    Tu korzystamy ponownie z Bielika, aby ładnie sformułował odpowiedź.
    """
    results_text = "Znalezione pozycje:\n"
    for i, r in enumerate(search_results, start=1):
        md = r['metadata']
        title = md['title']
        author = md['author']
        subjects = md['subjects'] if md['subjects'] else []
        subjects_str = ", ".join(subjects) if subjects else "brak"
        results_text += f"{i}) Tytuł: {title}, Autor: {author}, Tematy: {subjects_str}\n"

    final_prompt = f"""Użytkownik pytał: {user_query}

Oto wyniki wyszukiwania z bazy biblioteki:
{results_text}

Napisz zwięzłą i przyjazną odpowiedź dla użytkownika, uwzględniając powyższe pozycje.
"""

    return ask_bielik(final_prompt, max_new_tokens=300)


# ===================================================================
#          10. Główna funkcja RAG: interpret -> wyszukaj -> generuj
# ===================================================================

def rag_query(user_query, 
              embedding_model,
              vector_data,
              full_text_index,
              title_index,
              author_index,
              subjects_index,
              top_k=3):
    """
    1) interpretacja zapytania (Bielik -> [subject: "..."] / [author: "..."] / [title: "..."])
    2) wybór indeksu FAISS
    3) wyszukiwanie w FAISS
    4) generowanie finalnej odpowiedzi
    """
    # Interpretacja
    interpretation = interpret_query(user_query)
    print(">>> Interpretacja zapytania:\n", interpretation)

    # Wyłuskanie typu i frazy z format [subject: "..."]
    match = re.search(r'\[(subject|author|title):\s*"([^"]+)"\]', interpretation, re.IGNORECASE)
    if not match:
        # fallback: przeszukujemy full_text
        print("Nie rozpoznano rodzaju zapytania, używam full_text_index.")
        query_value = user_query
        index_to_use = full_text_index
        vector_key = "full_text_vector"
    else:
        query_type = match.group(1).lower()
        query_value = match.group(2)

        if query_type == "subject":
            index_to_use = subjects_index
            vector_key = "subjects_vector"
        elif query_type == "author":
            index_to_use = author_index
            vector_key = "author_vector"
        elif query_type == "title":
            index_to_use = title_index
            vector_key = "title_vector"
        else:
            index_to_use = full_text_index
            vector_key = "full_text_vector"

    # Wyszukiwanie w FAISS
    results = search_faiss(query_value, embedding_model, index_to_use, vector_data, vector_key=vector_key, top_k=top_k)
    
    # Stworzenie końcowej odpowiedzi
    final_answer = create_final_answer(results, user_query=user_query)
    return final_answer


# ===================================================================
#                     11. MAIN - DEMO
# ===================================================================
if __name__ == "__main__":
    # --- 1) Wczytujemy rekordy MARC ---
    print("Wczytywanie danych z MARC21...")
    marc_records = extract_marc_data(MARC_FILE_PATH, limit=100)

    # --- 2) Tworzymy embeddingi (z normalizacją) ---
    print("Generowanie wektorów (embeddingi)...")
    vector_data = process_to_vectors(marc_records, embedding_model)

    # --- 3) Budujemy indeksy FAISS ---
    print("Budowa indeksów FAISS...")
    full_text_index, _ = build_faiss_index(vector_data, "full_text_vector")
    title_index, _ = build_faiss_index(vector_data, "title_vector")
    author_index, _ = build_faiss_index(vector_data, "author_vector")
    subjects_index, _ = build_faiss_index(vector_data, "subjects_vector")

    # --- 4) Przykładowe zapytanie użytkownika ---
    user_question = "Czy znajdę książki o grach komputerowych?"

    # --- 5) RAG: interpretacja -> wyszukanie -> wygenerowanie odpowiedzi ---
    final_answer = rag_query(
        user_question,
        embedding_model=embedding_model,
        vector_data=vector_data,
        full_text_index=full_text_index,
        title_index=title_index,
        author_index=author_index,
        subjects_index=subjects_index,
        top_k=3
    )

    print("\n>>> Odpowiedź RAG:\n", final_answer)



#%% ALways last version:


import os
import re
import unicodedata
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from pymarc import MARCReader
from sentence_transformers import SentenceTransformer

# --------------------------
# Konfiguracja
# --------------------------
MARC_FILE_PATH = "D:/Nowa_praca/marki_po_updatach 2025,2024/NEW-marc_bn_books_08-02-2024.mrc"
EMBED_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"  # lub inny
LIMIT = 100  # limit rekordów do wczytania (dla testów)

# Nazwy plików, do których zapiszemy indeksy FAISS i dane
FULL_TEXT_INDEX_PATH = "full_text_index.faiss"
TITLE_INDEX_PATH = "title_index.faiss"
AUTHOR_INDEX_PATH = "author_index.faiss"
SUBJECTS_INDEX_PATH = "subjects_index.faiss"

VECTOR_DATA_PICKLE = "vector_data.pkl"
MARC_RECORDS_PICKLE = "marc_records.pkl"
SUBJECTS_METADATA_PICKLE = "subjects_metadata.pkl"

# --------------------------
# Normalizacja tekstu
# --------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.strip()

# --------------------------
# Wczytywanie danych MARC
# --------------------------
def extract_marc_data(file_path, limit=100):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Reading MARC", unit="record")):
            if i >= limit:
                break

            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            description = record.get_fields('520')
            description = description[0]['a'] if description and 'a' in description[0] else "Brak opisu"
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else "Nieznany autor"

            if title:
                rec = {
                    "title": title,
                    "author": author,
                    "subjects": subjects,
                    "description": description
                }
                records.append(rec)
    return records

# --------------------------
# Embedding: full_text, title, author
# --------------------------
def build_vectors(records, model):
    vector_data = []
    for i, r in enumerate(records):
        title_norm = normalize_text(r["title"])
        author_norm = normalize_text(r["author"])
        desc_norm = normalize_text(r["description"])

        # Full_text
        full_text = f"tytuł: {title_norm}. autor: {author_norm}. opis: {desc_norm}"
        full_text_vec = model.encode(full_text).astype('float32')
        title_vec = model.encode(title_norm).astype('float32')
        author_vec = model.encode(author_norm).astype('float32')

        vector_data.append({
            "id": i,
            "metadata": r,
            "full_text_vector": full_text_vec,
            "title_vector": title_vec,
            "author_vector": author_vec,
        })
    return vector_data

# --------------------------
# Budowa płaskiego indeksu FAISS
# --------------------------
def build_faiss_index(vector_data, vector_key="full_text_vector"):
    dim = len(vector_data[0][vector_key])
    index = faiss.IndexFlatIP(dim)
    embeddings_list = []
    for item in vector_data:
        vec = item[vector_key]
        vec = vec / np.linalg.norm(vec)  # normalizacja L2
        embeddings_list.append(vec)
    embeddings = np.array(embeddings_list, dtype='float32')
    index.add(embeddings)
    return index, embeddings

# --------------------------
# Budowa indeksu tematycznego (każda fraza osobno)
# --------------------------
def build_subjects_index(records, model):
    """ 
    Tworzy osobny indeks FAISS, w którym każda fraza z 'subjects' jest osobnym wektorem.
    Zwraca: (index_subjects, subjects_vectors, subjects_metadata)
    Gdzie subjects_metadata to lista dict z polami 'record_id', 'subject'
    """
    subjects_vectors_list = []
    subjects_metadata = []

    for i, rec in enumerate(records):
        subj_list = rec["subjects"] if rec["subjects"] else []
        for subj in subj_list:
            subj_norm = normalize_text(subj)
            vec = model.encode(subj_norm).astype('float32')
            vec = vec / np.linalg.norm(vec)  # normalizacja
            subjects_vectors_list.append(vec)
            subjects_metadata.append({
                "record_id": i,
                "subject": subj  # oryginalna fraza
            })

    if not subjects_vectors_list:
        print("Brak fraz w polach 650.")
        return None, None, []

    subjects_vectors = np.array(subjects_vectors_list, dtype='float32')
    dim = subjects_vectors.shape[1]
    index_subjects = faiss.IndexFlatIP(dim)
    index_subjects.add(subjects_vectors)

    return index_subjects, subjects_vectors, subjects_metadata

def search_subjects(query, model, index_subjects, subjects_metadata, records, top_k=3, score_threshold=0.8):
    # Jeśli indeks nie został utworzony, zwróć pustą listę
    if index_subjects is None:
        return []
    
    # Normalizujemy zapytanie
    q_norm = normalize_text(query)
    # Generujemy embedding zapytania
    q_vec = model.encode(q_norm).astype('float32')
    # Normalizujemy embedding zapytania (L2)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    # Wyszukujemy top_k podobnych wektorów w indeksie
    distances, indices = index_subjects.search(np.array([q_vec]), top_k)
    
    # Użyjemy słownika, aby dla danego record_id zapisać wynik o najwyższym score
    unique_results = {}
    
    for rank, idx in enumerate(indices[0]):
        score = distances[0][rank]
        if score < score_threshold:
            continue
        # Z subjects_metadata pobieramy, do którego rekordu należy dana fraza
        meta = subjects_metadata[idx]
        record_id = meta["record_id"]
        matched_subj = meta["subject"]
        
        # Jeśli ten rekord już mamy, zachowujemy wynik z wyższym score
        if record_id in unique_results:
            if score > unique_results[record_id]["score"]:
                unique_results[record_id] = {
                    "score": score,
                    "metadata": records[record_id],
                    "subject_matched": matched_subj
                }
        else:
            unique_results[record_id] = {
                "score": score,
                "metadata": records[record_id],
                "subject_matched": matched_subj
            }
    
    # Zwracamy listę unikalnych wyników
    return list(unique_results.values())

def search_faiss(query, model, index, vector_data, vector_key="full_text_vector", top_k=3, score_threshold=0.8):
    """
    Wyszukuje zapytanie 'query' w indeksie FAISS na podstawie wektorów.
    
    :param query: Tekst zapytania (np. tytuł lub autor)
    :param model: Model embeddingów (np. SentenceTransformer)
    :param index: Indeks FAISS (np. title_index lub author_index)
    :param vector_data: Lista słowników zawierających embeddingi oraz metadane
    :param vector_key: Klucz w vector_data, którego embeddingi mają być użyte (np. "title_vector" lub "author_vector")
    :param top_k: Liczba wyników do zwrócenia
    :param score_threshold: Minimalna wartość podobieństwa (score), aby wynik został zaakceptowany (domyślnie 0.8)
    :return: Lista wyników (każdy wynik to słownik z kluczami: "score", "metadata")
    """
    # Normalizujemy zapytanie
    q_norm = normalize_text(query)
    # Generujemy embedding zapytania
    q_vec = model.encode(q_norm).astype('float32')
    # Normalizujemy wektor zapytania (L2)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    # Wyszukujemy top_k podobnych wektorów w indeksie
    distances, indices = index.search(np.array([q_vec]), top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        score = distances[0][rank]
        if score < score_threshold:
            continue  # pomijamy wyniki, których score jest mniejsze niż threshold
        record_meta = vector_data[idx]["metadata"]
        results.append({
            "score": score,
            "metadata": record_meta
        })
    return results


# --------------------------
# MAIN (budujemy i zapisujemy)
# --------------------------
if __name__ == "__main__":
    print("Ładowanie modelu do embeddingów...")
    embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Wczytywanie danych MARC...")
    marc_records = extract_marc_data(MARC_FILE_PATH, LIMIT)

    print("Tworzenie wektorów (full_text/title/author)...")
    vector_data = build_vectors(marc_records, embedding_model)

    print("Budowa indeksów FAISS (full_text, title, author)...")
    full_text_index, _ = build_faiss_index(vector_data, "full_text_vector")
    title_index, _ = build_faiss_index(vector_data, "title_vector")
    author_index, _ = build_faiss_index(vector_data, "author_vector")

    print("Budowa indeksu FAISS dla subjects (każda fraza osobno)...")
    index_subjects, subjects_vectors, subjects_metadata = build_subjects_index(marc_records, embedding_model)

    # ---------------------------
    # ZAPISUJEMY indeksy FAISS
    # ---------------------------
    # print("\nZapisujemy indeksy FAISS do plików .faiss...")
    # faiss.write_index(full_text_index, FULL_TEXT_INDEX_PATH)
    # faiss.write_index(title_index, TITLE_INDEX_PATH)
    # faiss.write_index(author_index, AUTHOR_INDEX_PATH)
    # if index_subjects is not None:
    #     faiss.write_index(index_subjects, SUBJECTS_INDEX_PATH)

    # # ---------------------------
    # # ZAPISUJEMY vector_data, marc_records, subjects_metadata (pickle)
    # # ---------------------------
    # print("Zapis do pickle: vector_data, marc_records, subjects_metadata...")

    # with open(VECTOR_DATA_PICKLE, "wb") as f:
    #     pickle.dump(vector_data, f)

    # with open(MARC_RECORDS_PICKLE, "wb") as f:
    #     pickle.dump(marc_records, f)

    # with open(SUBJECTS_METADATA_PICKLE, "wb") as f:
    #     pickle.dump(subjects_metadata, f)

    # print("\nGotowe! Indeksy i dane zostały zapisane.")

    test_query = "gry komputerowe"
    results = search_subjects(test_query, embedding_model,
                              index_subjects, subjects_metadata,
                              marc_records, top_k=3)
    print(f"\nWyniki wyszukiwania tematycznego dla zapytania: {test_query}")

    for r in results:
        md = r["metadata"]
        print(f"Score={r['score']:.4f}")
        print(f"  Tytuł: {md.get('title')}")
        print(f"  Autor: {md.get('author')}")
        print(f"  Tematy: {md.get('subjects')}")
        print(f"  Opis: {md.get('description')}")
        if "subject_matched" in r:
            print(f"  Dopasowana fraza: {r['subject_matched']}")
        print("----")

    title_query = "Copernicon"
    results_title = search_faiss(title_query, embedding_model, title_index, vector_data, vector_key="title_vector", top_k=3)
    
    print("=== Wyniki wyszukiwania po tytule ===")
    for r in results_title:
        md = r["metadata"]
        print(f"Score: {r['score']:.4f}")
        print(f"  Tytuł: {md.get('title')}")
        print(f"  Autor: {md.get('author')}")
        print(f"  Tematy: {md.get('subjects')}")
        print(f"  Opis: {md.get('description')}")
        print("----")
    
    # Przykład wyszukiwania po autorze:
    author_query = "Garkowski, Patryk Daniel"
    results_author = search_faiss(author_query, embedding_model, author_index, vector_data, vector_key="author_vector", top_k=3)
    
    print("\n=== Wyniki wyszukiwania po autorze ===")
    for r in results_author:
        md = r["metadata"]
        print(f"Score: {r['score']:.4f}")
        print(f"  Tytuł: {md.get('title')}")
        print(f"  Autor: {md.get('author')}")
        print(f"  Tematy: {md.get('subjects')}")
        print(f"  Opis: {md.get('description')}")
        print("----")
    
    # Przykład wyszukiwania po temacie (subjects)
    subject_query = "gry komputerowe"
    results_subject = search_subjects(subject_query, embedding_model, subjects_index, subjects_metadata, marc_records, top_k=3, score_threshold=0.8)
    
    print("\n=== Wyniki wyszukiwania tematycznego ===")
    for r in results_subject:
        md = r["metadata"]
        print(f"Score: {r['score']:.4f}")
        print(f"  Tytuł: {md.get('title')}")
        print(f"  Autor: {md.get('author')}")
        print(f"  Tematy: {md.get('subjects')}")
        print(f"  Opis: {md.get('description')}")
        if "subject_matched" in r:
            print(f"  Dopasowana fraza: {r['subject_matched']}")
        print("----")
# use_bielik.py

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import faiss, pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# 1) Wczytujemy model do embeddingów (ten sam, co w build_index)
EMBED_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"
embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

# 2) Wczytujemy indeksy FAISS
full_text_index = faiss.read_index("full_text_index.faiss")
title_index = faiss.read_index("title_index.faiss")
author_index = faiss.read_index("author_index.faiss")
subjects_index = faiss.read_index("subjects_index.faiss")

# 3) Wczytujemy metadane
with open("vector_data.pkl", "rb") as f:
    vector_data = pickle.load(f)
with open("marc_records.pkl", "rb") as f:
    marc_records = pickle.load(f)
with open("subjects_metadata.pkl", "rb") as f:
    subjects_metadata = pickle.load(f)

#####################################################
# 1. Ładowanie Bielika z device=0 (GPU)
#####################################################
model_name = "speakleash/Bielik-11B-v2.3-Instruct"

print("Ładowanie tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Ładowanie modelu Bielik na GPU (device=0) w float16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()  # wgrywamy cały model na GPU

# Tworzymy pipeline z device=0 (GPU)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # wymusza GPU
)

def ask_bielik(prompt: str, max_new_tokens=512) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.2,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]

#####################################################
# 2. Prompt do interpretacji zapytań
#####################################################
interpret_prompt = """Jesteś asystentem bibliotecznym. 
Nie powtarzaj pytań użytkownika ani powyższych instrukcji. 
Twoim zadaniem jest wyłącznie:
1) Ustalić, czy zapytanie dotyczy tytułu, autora czy tematyki,
2) Wyodrębnić kluczowe hasło,
3) Wyświetlić wynik w formacie:
   [subject: "Temat"] lub [author: "Nazwisko"] lub [title: "Tytuł"]

Dodatkowo napisz krótko, czego poszukuje użytkownik.

Przykład:
Pytanie: "Czy znajdę książki o grach komputerowych?"
Odpowiedź:
[subject: "gry komputerowe"]
Użytkownik szuka książek związanych z grami komputerowymi.

Zachowaj tę konwencję. 

Pytanie użytkownika: {user_question}

Odpowiedź:
"""

def interpret_query(user_query: str) -> str:
    prompt = interpret_prompt + f"\nPytanie użytkownika: {user_query}\n"
    return ask_bielik(prompt, max_new_tokens=300)

#####################################################
# 3. Generowanie finalnej odpowiedzi (RAG)
#####################################################
def create_final_answer(results, user_query=""):
    if not results:
        results_text = "Brak wyników."
    else:
        results_text = "Znalezione pozycje:\n"
        for i, r in enumerate(results, start=1):
            md = r["metadata"]
            title = md["title"]
            author = md["author"]
            subjects = md["subjects"]
            subjects_str = ", ".join(subjects) if subjects else "brak"

            # Jeśli mamy klucz "subject_matched"
            matched_subj = r.get("subject_matched", None)
            if matched_subj:
                results_text += f"{i}) Tytuł: {title}, Autor: {author}, Tematy: {subjects_str} (Pasująca fraza: {matched_subj})\n"
            else:
                results_text += f"{i}) Tytuł: {title}, Autor: {author}, Tematy: {subjects_str}\n"

    final_prompt = f"""Użytkownik pytał: {user_query}

Oto wyniki wyszukiwania z bazy biblioteki:
{results_text}

Napisz zwięzłą i przyjazną odpowiedź dla użytkownika, uwzględniając powyższe pozycje.
"""
    return ask_bielik(final_prompt, max_new_tokens=300)

#####################################################
# 4. Pętla RAG
#####################################################
def rag_query(user_query, embedding_model,
              records,
              vector_data,
              full_text_index,
              title_index,
              author_index,
              index_subjects,
              subjects_metadata,
              top_k=3):
    
    # KROK 1: Interpretacja
    interpretation = interpret_query(user_query)
    print("\n[Interpretacja Bielika]:")
    print(interpretation)

    # Wyłuskanie z interpretacji
    match = re.search(r'\[(subject|author|title):\s*"([^"]+)"\]', interpretation, re.IGNORECASE)
    if not match:
        # fallback
        print("Nie rozpoznano typu zapytania, używam full_text.")
        query_value = user_query
        results = search_faiss(query_value, embedding_model, full_text_index, vector_data, "full_text_vector", top_k)
    else:
        qtype = match.group(1).lower()
        query_value = match.group(2)

        if qtype == "subject":
            results = search_subjects(query_value, embedding_model, index_subjects, subjects_metadata, records, top_k)
        elif qtype == "author":
            results = search_faiss(query_value, embedding_model, author_index, vector_data, "author_vector", top_k)
        elif qtype == "title":
            results = search_faiss(query_value, embedding_model, title_index, vector_data, "title_vector", top_k)
        else:
            results = search_faiss(query_value, embedding_model, full_text_index, vector_data, "full_text_vector", top_k)

    # KROK 2: Tworzenie finalnej odpowiedzi
    final_text = create_final_answer(results, user_query)
    return final_text

#####################################################
# 5. Główna pętla / DEMO
#####################################################
if __name__ == "__main__":
    while True:
        user_inp = input("\nUżytkownik pyta (lub 'exit'): ")
        if user_inp.lower() in ["exit", "quit"]:
            break

        answer = rag_query(
            user_inp,
            embedding_model,  # Tego użyłeś globalnie
            marc_records,     # wczytane z pickle
            vector_data,      # wczytane z pickle
            full_text_index,  # wczytane z faiss.read_index
            title_index,
            author_index,
            subjects_index,
            subjects_metadata,
            top_k=3
        )
        print("\n[RAG Odpowiedź]:\n", answer)
