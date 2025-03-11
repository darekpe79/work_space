# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:44:42 2025

@author: darek
"""

embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
embedding_model = SentenceTransformer('all-mpnet-base-v2')

embedding_model = SentenceTransformer('all-distilroberta-v1')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "speakleash/Bielik-11B-v2.3-Instruct"

# Ładujemy tokenizer i model Bielika w float16 na GPU:
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# ).cuda()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
    #device=0  # GPU
)

def ask_bielik(prompt: str, max_new_tokens=300) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,   # możesz zmienić
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]

# --------------------
# 1) Prompt interpretacyjny
# --------------------
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




interpret_prompt = """Jesteś asystentem bibliotecznym.

Masz wyłącznie ustalić, czy w pytaniu użytkownika występuje:
- tytuł,
- autor,
- tematyka,
- zakres dat (year range),
- pojedyncza data (year),
- miejsce wydania (place),
lub jakiekolwiek kombinacje powyższych.

Wynik przedstaw **wyłącznie** w formacie:
[author: "nazwisko"], [subject: "temat"], [title: "tytuł"], [year: "RRRR"], [year_range: "RRRR-RRRR"], [place: "miejsce wydania"]

Jeśli pytanie zawiera wiele aspektów, zwróć je osobno, np.:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"]

Nie powtarzaj pytania użytkownika, nie dodawaj komentarzy, nie zmieniaj nazwisk, tytułów, lat ani miejsca wydania.
Nie wpisuj kluczy, których nie ma w pytaniu.

Przykład:
Pytanie użytkownika: "Czy znajdę książki Mickiewicza o przyrodzie z lat 1830-1840 wydane w Warszawie?"
Odpowiedź:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"]

Pytanie użytkownika, interesuje Ci tylko ono, nic od siebie nie dodawaj, żadnych pytań, jeśli użytkownik używa imienia i nazwiska używaj ich obu, wszystko zawsze w mianowniku: {user_question}
Odpowiedź:"""

interpret_prompt = """Jesteś asystentem bibliotecznym.

Masz wyłącznie ustalić, czy w pytaniu użytkownika występuje:
- tytuł,
- autor,
- tematyka,
- zakres dat (year range),
- pojedyncza data (year),
- miejsce wydania (place),
- wydawca (publisher),
- język (language),
- typ dokumentu (document_type),
lub jakiekolwiek kombinacje powyższych.

Wynik przedstaw **wyłącznie** w formacie:
[author: "nazwisko"], [subject: "temat"], [title: "tytuł"], [year: "RRRR"], [year_range: "RRRR-RRRR"], [place: "miejsce wydania"], [publisher: "wydawca"], [language: "język"], [document_type: "typ dokumentu"]

Jeśli pytanie zawiera wiele aspektów, zwróć je osobno, np.:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"], [publisher: "Wydawnictwo Naukowe PWN"], [language: "polski"], [document_type: "książka"]

Nie powtarzaj pytania użytkownika, nie dodawaj komentarzy, nie zmieniaj nazwisk, tytułów, lat ani miejsca wydania.
Nie wpisuj kluczy, których nie ma w pytaniu.

Przykład:
Pytanie użytkownika: "Czy znajdę książki Mickiewicza o przyrodzie z lat 1830-1840 wydane w Warszawie?"
Odpowiedź:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"]

Pytanie użytkownika, interesuje Ci tylko ono, nic od siebie nie dodawaj, żadnych pytań, jeśli użytkownik używa imienia i nazwiska używaj ich obu, wszystko zawsze w mianowniku: {user_question}
Odpowiedź:"""
interpret_prompt = """Jesteś asystentem bibliotecznym.

Masz wyłącznie ustalić, czy w pytaniu użytkownika występuje:
- tytuł,
- autor,
- tematyka,
- zakres dat (year range),
- pojedyncza data (year),
- miejsce wydania (place),
- wydawca (publisher),
- język (language),
- typ dokumentu (document_type),
lub jakiekolwiek kombinacje powyższych.

Wynik przedstaw **wyłącznie** w formacie:
[author: "nazwisko"], [subject: "temat"], [title: "tytuł"], [year: "RRRR"], [year_range: "RRRR-RRRR"], [place: "miejsce wydania"], [publisher: "wydawca"], [language: "język"], [document_type: "typ dokumentu"]

Jeśli pytanie zawiera wiele aspektów, zwróć je osobno, np.:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"], [publisher: "Wydawnictwo Naukowe PWN"], [language: "polski"], [document_type: "książka"]

Nie powtarzaj pytania użytkownika, nie dodawaj komentarzy, nie zmieniaj nazwisk, tytułów, lat ani miejsca wydania.
Nie wpisuj kluczy, których nie ma w pytaniu.

Przykłady:
1. Pytanie użytkownika: "Czy znajdę książki Mickiewicza o przyrodzie z lat 1830-1840 wydane w Warszawie?"
Odpowiedź:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"]

2. Pytanie użytkownika: "Szukam czasopism o astronomii wydanych w Krakowie."
Odpowiedź:
[subject: "astronomia"], [place: "Kraków"], [document_type: "czasopismo"]

3. Pytanie użytkownika: "Czy macie artykuły naukowe o sztucznej inteligencji z lat 2010-2020?"
Odpowiedź:
[subject: "sztuczna inteligencja"], [year_range: "2010-2020"], [document_type: "artykuł naukowy"]

4. Pytanie użytkownika: "Gdzie mogę znaleźć ebooki Stephena Kinga w języku angielskim?"
Odpowiedź:
[author: "Stephen King"], [language: "angielski"], [document_type: "ebook"]

5. Pytanie użytkownika: "Czy macie raporty o zmianach klimatycznych wydane przez ONZ?"
Odpowiedź:
[subject: "zmiany klimatyczne"], [publisher: "ONZ"], [document_type: "raport"]

6. Pytanie użytkownika: "Szukam audiobooków z powieściami kryminalnymi."
Odpowiedź:
[subject: "powieści kryminalne"], [document_type: "audiobook"]

Pytanie użytkownika, interesuje Ci tylko ono, nic od siebie nie dodawaj, żadnych pytań, jeśli użytkownik używa imienia i nazwiska używaj ich obu, wszystko zawsze w mianowniku: {user_question}
Odpowiedź:"""

interpret_prompt = """Jesteś asystentem bibliotecznym.

Masz wyłącznie ustalić, czy w pytaniu użytkownika występuje:
- tytuł,
- autor,
- tematyka,
- zakres dat (year range),
- pojedyncza data (year),
- miejsce wydania (place),
- wydawca (publisher),
- język (language),
- typ dokumentu (document_type),
lub jakiekolwiek kombinacje powyższych.

Wynik przedstaw **wyłącznie** w formacie:
[author: "nazwisko"], [subject: "temat"], [title: "tytuł"], [year: "RRRR"], [year_range: "RRRR-RRRR"], [place: "miejsce wydania"], [publisher: "wydawca"], [language: "język"], [document_type: "typ dokumentu"]

Jeśli pytanie zawiera wiele aspektów, zwróć je osobno, np.:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"], [publisher: "Wydawnictwo Naukowe PWN"], [language: "polski"], [document_type: "książka"]

Nie powtarzaj pytania użytkownika, nie dodawaj komentarzy, nie zmieniaj nazwisk, tytułów, lat ani miejsca wydania.
Nie wpisuj kluczy, których nie ma w pytaniu.

Przykłady:
1. Pytanie użytkownika: "Czy znajdę książki Mickiewicza o przyrodzie z lat 1830-1840 wydane w Warszawie?"
Odpowiedź:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"]

2. Pytanie użytkownika: "Szukam czasopism o astronomii wydanych w Krakowie."
Odpowiedź:
[subject: "astronomia"], [place: "Kraków"], [document_type: "czasopismo"]

3. Pytanie użytkownika: "Czy macie artykuły naukowe o sztucznej inteligencji z lat 2010-2020?"
Odpowiedź:
[subject: "sztuczna inteligencja"], [year_range: "2010-2020"], [document_type: "artykuł naukowy"]

4. Pytanie użytkownika: "Gdzie mogę znaleźć ebooki Stephena Kinga w języku angielskim?"
Odpowiedź:
[author: "Stephen King"], [language: "angielski"], [document_type: "ebook"]

5. Pytanie użytkownika: "Czy macie raporty o zmianach klimatycznych wydane przez ONZ?"
Odpowiedź:
[subject: "zmiany klimatyczne"], [publisher: "ONZ"], [document_type: "raport"]

6. Pytanie użytkownika: "Szukam audiobooków z powieściami kryminalnymi."
Odpowiedź:
[subject: "powieści kryminalne"], [document_type: "audiobook"]

7. Pytanie użytkownika: "Czy macie raporty ekonomiczne wydane przez Bank Światowy oraz Forum Ekonomiczne?"
Odpowiedź:
[subject: "ekonomia"], [publisher: "Bank Światowy"], [publisher: "Forum Ekonomiczne"], [document_type: "raport"]

8. Pytanie użytkownika: "Szukam książek o historii Polski lub historii Europy wydanych w latach 2000-2010."
Odpowiedź:
[subject: "historia Polski"], [subject: "historia Europy"], [year_range: "2000-2010"], [document_type: "książka"]

9. Pytanie użytkownika: "Czy macie artykuły naukowe o medycynie oraz raporty o zdrowiu publicznym?"
Odpowiedź:
[subject: "medycyna"], [document_type: "artykuł naukowy"], [subject: "zdrowie publiczne"], [document_type: "raport"]

Pytanie użytkownika, interesuje Ci tylko ono, nic od siebie nie dodawaj, żadnych pytań, jeśli użytkownik używa imienia i nazwiska używaj ich obu, wszystko zawsze w mianowniku: {user_question}
Odpowiedź:"""

import datetime

# Oblicz bieżący rok
current_year = datetime.datetime.now().year

# Dynamically insert the current year into the prompt:
interpret_prompt = f"""Jesteś asystentem bibliotecznym.

Masz wyłącznie ustalić, czy w pytaniu użytkownika występuje:
- tytuł,
- autor,
- tematyka,
- zakres dat (year range),
- pojedyncza data (year),
- miejsce wydania (place),
- wydawca (publisher),
- język (language),
- typ dokumentu (document_type),
lub jakiekolwiek kombinacje powyższych.

Wynik przedstaw **wyłącznie** w formacie:
[author: "nazwisko"], [subject: "temat"], [title: "tytuł"], [year: "RRRR"], [year_range: "RRRR-RRRR"], [place: "miejsce wydania"], [publisher: "wydawca"], [language: "język"], [document_type: "typ dokumentu"]

Jeśli pytanie zawiera wiele aspektów, zwróć je osobno, np.:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"], [publisher: "Wydawnictwo Naukowe PWN"], [language: "polski"], [document_type: "książka"]

Nie powtarzaj pytania użytkownika, nie dodawaj komentarzy, nie zmieniaj nazwisk, tytułów, lat, miejsc wydania, wydawcy, języka ani typu dokumentu.
Nie wpisuj kluczy, których nie ma w pytaniu.

Dodatkowe zasady interpretacji dat:
1. Jeśli w pytaniu jest „ostatnich X lat”, przyjmij, że obecny rok to {current_year}, więc zapisz [year_range: "{current_year} - X + 1-{current_year}"].
2. Jeśli w pytaniu jest „przed rokiem YYYY”, przyjmij [year_range: "0000-(YYYY-1)"].
3. Jeśli w pytaniu jest „po roku YYYY”, przyjmij [year_range: "(YYYY+1)-9999"].
4. Jeśli w pytaniu jest „w roku YYYY”, przyjmij [year: "YYYY"].
5. Jeśli w pytaniu jest „od roku YYYY do roku ZZZZ”, przyjmij [year_range: "YYYY-ZZZZ"].
6. Jeśli nie da się jednoznacznie ustalić, nie wpisuj year ani year_range.

Przykłady:

1. Pytanie użytkownika: "Czy znajdę książki Mickiewicza o przyrodzie z lat 1830-1840 wydane w Warszawie?"
Odpowiedź:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"]

2. Pytanie użytkownika: "Szukam czasopism o astronomii wydanych w Krakowie."
Odpowiedź:
[subject: "astronomia"], [place: "Kraków"], [document_type: "czasopismo"]

3. Pytanie użytkownika: "Czy macie artykuły naukowe o sztucznej inteligencji z lat 2010-2020?"
Odpowiedź:
[subject: "sztuczna inteligencja"], [year_range: "2010-2020"], [document_type: "artykuł naukowy"]

4. Pytanie użytkownika: "Gdzie mogę znaleźć ebooki Stephena Kinga w języku angielskim?"
Odpowiedź:
[author: "Stephen King"], [language: "angielski"], [document_type: "ebook"]

5. Pytanie użytkownika: "Czy macie raporty o zmianach klimatycznych wydane przez ONZ?"
Odpowiedź:
[subject: "zmiany klimatyczne"], [publisher: "ONZ"], [document_type: "raport"]

6. Pytanie użytkownika: "Szukam audiobooków z powieściami kryminalnymi."
Odpowiedź:
[subject: "powieści kryminalne"], [document_type: "audiobook"]

7. Pytanie użytkownika: "Czy macie raporty ekonomiczne wydane przez Bank Światowy oraz Forum Ekonomiczne?"
Odpowiedź:
[subject: "ekonomia"], [publisher: "Bank Światowy"], [publisher: "Forum Ekonomiczne"], [document_type: "raport"]

8. Pytanie użytkownika: "Szukam książek o historii Polski lub historii Europy wydanych w latach 2000-2010."
Odpowiedź:
[subject: "historia Polski"], [subject: "historia Europy"], [year_range: "2000-2010"], [document_type: "książka"]

9. Pytanie użytkownika: "Czy macie artykuły naukowe o medycynie oraz raporty o zdrowiu publicznym?"
Odpowiedź:
[subject: "medycyna"], [document_type: "artykuł naukowy"], [subject: "zdrowie publiczne"], [document_type: "raport"]

10. Pytanie użytkownika: "Czy macie książki Stephena Kinga wydane przed 2000 rokiem?"
Odpowiedź:
[author: "Stephen King"], [year_range: "0000-1999"], [document_type: "książka"]

11. Pytanie użytkownika: "Czy macie artykuły naukowe o psychologii z ostatnich 5 lat?"
Odpowiedź:
[subject: "psychologia"], [year_range: "{current_year - 4}-{current_year}"], [document_type: "artykuł naukowy"]

Pytanie użytkownika, interesuje Ci tylko ono, nic od siebie nie dodawaj, żadnych pytań, jeśli użytkownik używa imienia i nazwiska używaj ich obu, wszystko zawsze w mianowniku, pamiętaj o zasadach dotyczących dat i obecnym roku {current_year}!: {{user_question}}
Odpowiedź:
"""



final_prompt_template = """Oto wyniki wyszukiwania z bazy biblioteki:
{results_text}

Napisz zwięzłą, krótką, konkretną i formalną odpowiedź dla użytkownika, uwzględniając powyższe pozycje. Podziękuj za pytanie i wymień pozycje z bazy, z polami które otrzymałes, nic więcej nie dodawaj.
"""
def create_final_prompt(user_query, results_text):
    return f"""Użytkownik pytał: {user_query}

{final_prompt_template.format(results_text=results_text)}
"""

if __name__ == "__main__":
    #user_query = "Czy znajdę książki o grach komputerowych?"
    user_query = "„Czy macie książki napisane przez Bolesława Prusa, wydane w Warszawie w roku 1890?"
    user_query ="Szukam książek o historii Polski w języku angielskim, wydanych przez wydawnictwo Penguin."
    user_query ="Czy macie czasopisma o astronomii wydane w Krakowie?"
    user_query ="Czy macie artykuły naukowe o medycynie oraz raporty o zdrowiu publicznym?"
    user_query ="Czy macie raporty ekonomiczne wydane przez Bank Światowy oraz Forum Europejskie?"
    user_query ="Czy macie czasopisma o technologii oraz artykuły naukowe o sztucznej inteligencji?"
    user_query ="Czy macie książki Stephena Kinga wydane przed 2000 rokiem oraz artykuły naukowe o psychologii z ostatnich 5 lat?"
    user_query ="Czy znajdę ebooki w języku angielskim autorstwa J.K. Rowling?"
    interp = interpret_prompt.format(user_question=user_query)

    print(">>> interpret_prompt:")
    print(interp)

    interpretation = ask_bielik(interp, max_new_tokens=150)

    print("=== INTERPRETACJA ===")
    print(interpretation)
    
    # 2) Załóżmy, że "wyniki" to 2 przykładowe książki (normalnie tu byłoby z FAISS):
#     mock_results_text = """Znalezione pozycje:
# 1) Tytuł: "Historia gier komputerowych", Autor: Jan Kowalski
# 2) Tytuł: "Gry komputerowe. Wpływ na kulturę masową", Autor: Anna Nowak
# """

    mock_results_text = """Znalezione pozycje:
    1)   Tytuł: Traktat o checkpointach /
      Autor: Garkowski, Patryk Daniel
      Tematy: ['Gry komputerowe']
    """

    
    # Teraz tworzymy prompt finalny
    final_prompt = create_final_prompt(user_query, mock_results_text)
    
    final_answer = ask_bielik(final_prompt, max_new_tokens=300)
    print("\n=== FINALNA ODPOWIEDŹ ===")
    print(final_answer)


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

def search_faiss(query, model, index, vector_data, vector_key, top_k=3, score_threshold=0.8):  #="full_text_vector"
    """
    Wyszukuje zapytanie 'query' w indeksie FAISS przy użyciu embeddingów określonych przez 'vector_key'.
    
    :param query: Tekst zapytania (np. tytuł lub autor).
    :param model: Model embeddingów (np. SentenceTransformer).
    :param index: Indeks FAISS (np. title_index lub author_index).
    :param vector_data: Lista słowników zawierających embeddingi oraz metadane.
                        Każdy element powinien zawierać klucz odpowiadający vector_key, np. "title_vector" lub "author_vector".
    :param vector_key: Klucz wskazujący, z którego embeddingu korzystamy (domyślnie "full_text_vector").
    :param top_k: Liczba wyników do zwrócenia.
    :param score_threshold: Minimalna wartość podobieństwa, aby wynik został uwzględniony (np. 0.8).
    :return: Lista wyników – każdy wynik to słownik zawierający:
             "score" (podobieństwo),
             "metadata" (dane oryginalnego rekordu),
             opcjonalnie "embedding" (embedding z vector_key, jeśli jest potrzebny).
    """
    # Upewnij się, że każdy element vector_data zawiera klucz vector_key
    for item in vector_data:
        if vector_key not in item:
            raise ValueError(f"Brak klucza '{vector_key}' w niektórych elementach vector_data.")
    
    # Normalizujemy zapytanie
    q_norm = normalize_text(query)
    # Generujemy embedding zapytania
    q_vec = model.encode(q_norm).astype('float32')
    # Normalizujemy wektor zapytania (L2)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    # Wyszukujemy top_k podobnych wektorów w przekazanym indeksie
    distances, indices = index.search(np.array([q_vec]), top_k)
    results = []
    for rank, idx in enumerate(indices[0]):
        score = distances[0][rank]
        if score < score_threshold:
            continue  # pomijamy wyniki, które nie spełniają progu
        record_meta = vector_data[idx]["metadata"]
        results.append({
            "score": score,
            "metadata": record_meta,
            "embedding": vector_data[idx].get(vector_key)  # opcjonalnie – może pomóc przy debugowaniu
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


    # test_query = "gry komputerowe"
    # results = search_subjects(test_query, embedding_model,
    #                           index_subjects, subjects_metadata,
    #                           marc_records, top_k=3)
    # print(f"\nWyniki wyszukiwania tematycznego dla zapytania: {test_query}")

    # for r in results:
    #     md = r["metadata"]
    #     print(f"Score={r['score']:.4f}")
    #     print(f"  Tytuł: {md.get('title')}")
    #     print(f"  Autor: {md.get('author')}")
    #     print(f"  Tematy: {md.get('subjects')}")
    #     print(f"  Opis: {md.get('description')}")
    #     if "subject_matched" in r:
    #         print(f"  Dopasowana fraza: {r['subject_matched']}")
    #     print("----")

    # title_query = "Copernicon"
    # results_title = search_faiss(title_query, embedding_model, title_index, vector_data, vector_key="title_vector", top_k=3)
    
    # print("=== Wyniki wyszukiwania po tytule ===")
    # for r in results_title:
    #     md = r["metadata"]
    #     print(f"Score: {r['score']:.4f}")
    #     print(f"  Tytuł: {md.get('title')}")
    #     print(f"  Autor: {md.get('author')}")
    #     print(f"  Tematy: {md.get('subjects')}")
    #     print(f"  Opis: {md.get('description')}")
    #     print("----")
    
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


#%% falcon            
from transformers import pipeline

model_name = "tiiuae/falcon-7b-instruct"
generator = pipeline(
    "text-generation",
    model=model_name,
    device=0,
    return_full_text=False
)

def ask_falcon(question: str, max_new_tokens=200) -> str:
    """
    Zadaj pytanie modelowi Falcon 7B i zwróć odpowiedź.
    """
    prompt = (
        "You are a helpful, knowledgeable assistant. "
        "Please answer in detail:\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"].strip()


q="What is the future of AI?"
response = ask_falcon(q, max_new_tokens=300)

if __name__ == "__main__":
    # Przykładowe pytania (po angielsku i po polsku)
    questions = [
        "What is the future of AI?",
                "How does reinforcement learning differ from supervised learning?"
    ]

    for i, q in enumerate(questions, 1):
        response = ask_falcon(q, max_new_tokens=200)
        print(f"\n--- [Pytanie {i}] ---")
        print("Question:", q)
        print("Answer:", response)
        
        
        
from transformers import pipeline

def translate_pl_to_en(text_pl: str) -> str:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")
    # translator może przetwarzać listy, więc w przypadku pojedynczego str:
    result = translator(text_pl, max_length=512)
    return result[0]["translation_text"]

if __name__ == "__main__":
    text_pl = '''skarpetki miały być wielokolorowe, a są białe'''
    translation = translate_pl_to_en(text_pl)
    print("ORYGINAŁ (PL):", text_pl)
    print("TŁUMACZENIE (EN):", translation)