# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:54:57 2025

@author: darek
"""

    
#%% Wszystkie id z prawdopodobieństwem w jednym wierszu 
import os
import json
import pandas as pd

def process_files_in_dir(input_dir: str, output_excel: str):
    """
    Przetwarza wszystkie pliki w katalogu `input_dir` i zapisuje
    zbiorcze wyniki do pliku Excel `output_excel`.
    Dla każdego spana NER zbiera wszystkie pary concept–score (z "results").
    """
    all_rows = []
    
    # Przechodzimy przez wszystkie pliki w katalogu
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        
        # Pomijamy katalogi
        if not os.path.isfile(filepath):
            continue
        
        # Wczytujemy plik jako JSON
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARN] Nie można wczytać pliku {filename} jako JSON: {e}")
            continue
        
        # Pobierz 'id' tekstu, jeśli jest
        text_id = data.get("id", None)
        
        # Pełny tekst (do wycinków spana)
        full_text = data.get("text", "")
        
        # Spany NER
        spans_ner = data.get("spans", {}).get("ner", [])
        
        # Szukamy sekcji z linkowaniem
        linking = data.get("records", {}).get("linking", {})
        clalink = linking.get("clalink", {})
        link_info_list = clalink.get("links", [])
        
        # Mapa: obj_id -> list of {concept, score}
        obj_id_to_results = {}
        for link_info in link_info_list:
            obj_id = link_info["obj-id"]
            results = link_info.get("results", [])
            # Zapisujemy wszystkie wyniki (każdy to {concept, score})
            obj_id_to_results[obj_id] = results
        
        # Iterujemy po spanach NER
        for span in spans_ner:
            obj_id = span["id"]
            start = span["start"]
            stop = span["stop"]
            ner_type = span.get("type", None)
            
            # Fragment tekstu
            entity_text = full_text[start:stop]
            
            # Wszystkie wyniki concept–score z linkowania
            results = obj_id_to_results.get(obj_id, [])
            
            # Tworzymy czytelny string z listy results
            # np. "Q1 (-29.79); Q221392 (-38.83); Q3327819 (-39.95)"
            concepts_scores_str = "; ".join(
                f"{r['concept']} ({r['score']})"
                for r in results
            )
            
            # Dodajemy wiersz do tabeli
            row = {
                "file_name": filename,
                "text_id": text_id,
                "obj_id": obj_id,
                "entity_text": entity_text,
                "start": start,
                "stop": stop,
                "ner_type": ner_type,
                # Kolumna z wszystkimi parami concept–score:
                "wikidata_concepts_scores": concepts_scores_str
            }
            all_rows.append(row)
    
    # Tworzymy DataFrame
    df = pd.DataFrame(all_rows)
    
    # Zapisujemy do Excela
    df.to_excel(output_excel, index=False)
    
    if not df.empty:
        liczba_rekordow = len(df)
        liczba_plikow = df["file_name"].nunique()
        print(f"Zapisano {liczba_rekordow} rekordów z {liczba_plikow} plików do: {output_excel}")
    else:
        print("Brak danych – ramka danych jest pusta.")


if __name__ == "__main__":
    # Przykładowe wywołanie
    process_files_in_dir(
        input_dir="C:/Users/darek/Downloads/postagger_13.2_13_46/",
        output_excel="zbiory_ner.xlsx"
    )
#%% wszystkie id z prawdopodobieństwem w róznych wierszach
import os
import json
import pandas as pd

def process_files_in_dir_explode(input_dir: str, output_excel: str, context_size: int = 100):
    """
    - Przetwarza wszystkie pliki w `input_dir`.
    - Dla każdej jednostki (spana) zwraca wiersze, gdzie każda para concept–score
      jest zapisywana osobno.
    - Do każdego wiersza dodaje się 'context_snippet' – fragment tekstu z ± context_size znaków.
    - Jeśli są wyniki, sortuje je malejąco według 'score' (czyli najlepszy wynik jest pierwszy).
    """
    all_rows = []
    
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARN] Nie można wczytać pliku {filename} jako JSON: {e}")
            continue
        
        text_id = data.get("id", None)
        full_text = data.get("text", "")
        spans_ner = data.get("spans", {}).get("ner", [])
        
        # Linkowanie: mapujemy obj_id na listę wyników [{concept, score}, ...]
        linking = data.get("records", {}).get("linking", {})
        clalink = linking.get("clalink", {})
        link_info_list = clalink.get("links", [])
        
        obj_id_to_results = {}
        for link_info in link_info_list:
            obj_id = link_info["obj-id"]
            obj_id_to_results[obj_id] = link_info.get("results", [])
        
        # Iterujemy po rozpoznanych jednostkach (spanach)
        for span in spans_ner:
            obj_id = span["id"]
            start = span["start"]
            stop = span["stop"]
            ner_type = span.get("type", None)
            
            entity_text = full_text[start:stop]
            
            # Wyliczamy kontekst: ± context_size znaków, ale nie wykraczając poza granice full_text
            context_start = max(0, start - context_size)
            context_end = min(len(full_text), stop + context_size)
            context_snippet = full_text[context_start:context_end]
            
            results = obj_id_to_results.get(obj_id, [])
            
            # Jeśli brak wyników, dodajemy jeden wiersz z pustymi concept i score
            if not results:
                row = {
                    "file_name": filename,
                    "text_id": text_id,
                    "full_text": full_text,
                    "context_snippet": context_snippet,
                    "obj_id": obj_id,
                    "entity_text": entity_text,
                    "start": start,
                    "stop": stop,
                    "ner_type": ner_type,
                    "concept": None,
                    "score": None
                }
                all_rows.append(row)
            else:
                # Sortujemy wyniki malejąco po score (najwyższy score pierwszy)
                sorted_results = sorted(results, key=lambda r: r["score"], reverse=True)
                for r in sorted_results:
                    row = {
                        "file_name": filename,
                        "text_id": text_id,
                        "full_text": full_text,
                        "context_snippet": context_snippet,
                        "obj_id": obj_id,
                        "entity_text": entity_text,
                        "start": start,
                        "stop": stop,
                        "ner_type": ner_type,
                        "concept": r["concept"],
                        "score": r["score"]
                    }
                    all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    df.to_excel(output_excel, index=False)
    
    if not df.empty:
        print(f"Zapisano {len(df)} wierszy z {df['file_name'].nunique()} plików do: {output_excel}")
    else:
        print("Brak danych – ramka jest pusta.")

if __name__ == "__main__":
    process_files_in_dir_explode(
        input_dir="D:/Nowa_praca/KPO/postagger/fragmenty artykułów dla CLARIN-u-20250218T071948Z-001/jsony/",
        output_excel="D:/Nowa_praca/KPO/postagger/zbiory_ner_explode.xlsx"
    )


#%%tylko najlepszy score
import os
import json
import pandas as pd

def process_files_in_dir_best_score(input_dir: str, output_excel: str):
    """
    Przetwarza wszystkie pliki w katalogu `input_dir` i zapisuje
    zbiorcze wyniki do pliku Excel `output_excel`.
    
    Dla każdego spana NER pobiera z "results" w polu "links" to dopasowanie,
    które ma najWYŻSZY (najlepszy) wynik 'score'.
    """
    all_rows = []
    
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        
        # Pomijamy katalogi
        if not os.path.isfile(filepath):
            continue
        
        # Wczytujemy JSON
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARN] Nie można wczytać pliku {filename} jako JSON: {e}")
            continue
        
        text_id = data.get("id", None)
        full_text = data.get("text", "")
        spans_ner = data.get("spans", {}).get("ner", [])
        
        # Linkowanie
        linking = data.get("records", {}).get("linking", {})
        clalink = linking.get("clalink", {})
        link_info_list = clalink.get("links", [])
        
        # Przygotowujemy mapę: obj_id -> [ {concept, score}, {…} ]
        obj_id_to_results = {}
        for link_info in link_info_list:
            obj_id = link_info["obj-id"]
            all_results = link_info.get("results", [])
            obj_id_to_results[obj_id] = all_results
        
        # Teraz iterujemy po spanach NER
        for span in spans_ner:
            obj_id = span["id"]
            start = span["start"]
            stop = span["stop"]
            ner_type = span.get("type", None)
            
            entity_text = full_text[start:stop]
            
            # Wszystkie dopasowania do Wikidaty
            results = obj_id_to_results.get(obj_id, [])
            
            # Jeśli brak dopasowań, ustawiamy None
            if not results:
                best_concept = None
                best_score = None
            else:
                # Znajdujemy to o najwyższym score
                best_result = max(results, key=lambda r: r["score"])
                best_concept = best_result.get("concept")
                best_score = best_result.get("score")
            
            row = {
                "file_name": filename,
                "text_id": text_id,
                "obj_id": obj_id,
                "full_text": full_text,
                "entity_text": entity_text,
                "start": start,
                "stop": stop,
                "ner_type": ner_type,
                "best_concept": best_concept,
                "best_score": best_score
            }
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    df.to_excel(output_excel, index=False)
    
    if not df.empty:
        print(f"Zapisano {len(df)} rekordów z {df['file_name'].nunique()} plików do: {output_excel}")
    else:
        print("Brak danych do zapisania – DataFrame jest pusty.")

if __name__ == "__main__":
    # Przykładowe wywołanie
    process_files_in_dir_best_score(
        input_dir="C:/Users/darek/Downloads/postagger_13.2_13_46/",
        output_excel="zbiory_ner_best_score.xlsx"
    )




#%% nowy kod do plików 30.06.2025

import os
import json
import pandas as pd

# Ścieżki
input_file_path = "D:/Nowa_praca/KPO/dariah_new_model_deepseek-v3/dariah_new_model_deepseek-v3/Fragment 1.json"
output_excel_path = "D:/Nowa_praca/KPO/dariah_new_model_deepseek-v3/dariah_new_model_deepseek-v3/Fragment_1_extracted.xlsx"

# Parametry kontekstu
context_size = 100

# Wczytanie pliku JSON
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

text_id = data.get("id", None)
full_text = data.get("text", "")
spans_ner = data.get("spans", {}).get("ner", [])

linking = data.get("records", {}).get("linking", {})
clalink = linking.get("clalink", {})
link_info_list = clalink.get("links", [])

# Mapowanie obj-id do results
obj_id_to_results = {}
for link_info in link_info_list:
    obj_id = link_info["obj-id"]
    obj_id_to_results[obj_id] = link_info.get("results", [])

# Gromadzenie danych
all_rows = []

for span in spans_ner:
    obj_id = span["id"]
    start = span["start"]
    stop = span["stop"]
    ner_type = span.get("type", None)
    entity_text = full_text[start:stop]

    # Fragment kontekstu
    context_start = max(0, start - context_size)
    context_end = min(len(full_text), stop + context_size)
    context_snippet = full_text[context_start:context_end]

    results = obj_id_to_results.get(obj_id, [])

    if not results:
        all_rows.append({
            "text_id": text_id,
            "context_snippet": context_snippet,
            "obj_id": obj_id,
            "entity_text": entity_text,
            "ner_type": ner_type,
            "concept": None,
            "score": None,
            "true_false": None
        })
    else:
        for r in results:
            concept = r["concept"][0] if isinstance(r["concept"], list) and r["concept"] else None
            score = r["score"] if r.get("score") is not None else None
            tf = r.get("T/F", None)

            all_rows.append({
                "text_id": text_id,
                "context_snippet": context_snippet,
                "obj_id": obj_id,
                "entity_text": entity_text,
                "ner_type": ner_type,
                "concept": concept,
                "score": score,
                "true_false": tf
            })

# Zapis do Excela
df = pd.DataFrame(all_rows)
df.to_excel(output_excel_path, index=False)

# Wyświetlenie
import ace_tools as tools
tools.display_dataframe_to_user(name="Fragment 5 Extracted Data", dataframe=df)




#%% label from wikidata
import requests
import pandas as pd

def get_wikidata_label(qid: str, lang: str = "pl") -> str:
    """
    Pobiera etykietę (label) z Wikidaty dla danego QID i języka (domyślnie polski).
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        entity = data["entities"].get(qid, {})
        labels = entity.get("labels", {})
        label_info = labels.get(lang, {})
        label = label_info.get("value")
        return label
    except (requests.RequestException, KeyError, ValueError):
        return None

def get_wikidata_description(qid: str, lang: str = "pl") -> str:
    """
    Pobiera opis (description) z Wikidaty dla danego QID i języka (domyślnie polski).
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        entity = data["entities"].get(qid, {})
        descriptions = entity.get("descriptions", {})
        desc_info = descriptions.get(lang, {})
        description = desc_info.get("value")
        return description
    except (requests.RequestException, KeyError, ValueError):
        return None

def enrich_excel_with_wikidata_labels(input_excel: str, output_excel: str, qid_column: str = "wikidata_id"):
    """
    Wczytuje plik Excel `input_excel` (z kolumną QID, np. 'wikidata_id'),
    pobiera etykiety (label) i opisy (description) z Wikidaty,
    dodaje kolumny 'wikidata_label', 'wikidata_description' oraz 'wikidata_url'
    i zapisuje wynik do `output_excel`.
    """
    # 1. Wczytujemy dane z Excela
    df = pd.read_excel(input_excel)
    
    if qid_column not in df.columns:
        print(f"Brak kolumny '{qid_column}' w pliku. Przerywam.")
        return
    
    # 2. Zbieramy unikatowe QID-y
    unique_qids = df[qid_column].dropna().unique()
    
    # 3. Tworzymy mapy: QID -> label oraz QID -> description
    qid_to_label = {}
    qid_to_description = {}
    
    for qid in unique_qids:
        if isinstance(qid, str) and qid.startswith("Q"):
            label = get_wikidata_label(qid, lang="pl")
            description = get_wikidata_description(qid, lang="pl")
            qid_to_label[qid] = label
            qid_to_description[qid] = description
            print(f"Przetworzono {qid}: label = {label}, description = {description}")
        else:
            qid_to_label[qid] = None
            qid_to_description[qid] = None
    
    # 4. Funkcja tworząca link do Wikidaty
    def make_url(qid):
        if isinstance(qid, str) and qid.startswith("Q"):
            return f"https://www.wikidata.org/wiki/{qid}"
        return None
    
    # 5. Dodajemy nowe kolumny do DataFrame
    df["wikidata_label"] = df[qid_column].map(qid_to_label)
    df["wikidata_description"] = df[qid_column].map(qid_to_description)
    df["wikidata_url"] = df[qid_column].apply(make_url)
    
    # 6. Zapisujemy wynik do Excela
    df.to_excel(output_excel, index=False)
    print(f"Zapisano wzbogacony plik do: {output_excel}")

if __name__ == "__main__":
    enrich_excel_with_wikidata_labels(
        input_excel="D:/Nowa_praca/KPO/dariah_new_model_deepseek-v3/dariah_new_model_deepseek-v3/Fragment_5_extracted.xlsx",
        output_excel="D:/Nowa_praca/KPO/dariah_new_model_deepseek-v3/dariah_new_model_deepseek-v3/Fragment_5_extracted_labels.xlsx",
        qid_column="concept"  # Nazwa kolumny z QID
    )

import requests
import json

# Przykładowy QID:
qid = "Q36"  # Zmień na dowolny inny, np. "Q207272", "Q809"

# Endpoint Wikidaty do pobierania informacji o encji
url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    # Wypisujemy surowe dane JSON (np. żeby zobaczyć całą strukturę)
    print("===== Odpowiedź Wikidaty (surowe JSON) =====")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()

    # Teraz próbujemy wyciągnąć etykietę i opis w języku polskim
    entities = data.get("entities", {})
    entity_data = entities.get(qid, {})
    labels = entity_data.get("labels", {})
    descriptions = entity_data.get("descriptions", {})

    pl_label = labels.get("pl", {}).get("value")
    pl_desc = descriptions.get("pl", {}).get("value")

    print("===== Wyciągnięte informacje =====")
    print(f"Etykieta PL: {pl_label}")
    print(f"Opis PL:     {pl_desc}")
else:
    print(f"[BŁĄD] Otrzymano kod HTTP {response.status_code}")


#%% porownanie nowe wyniki, stare 08.07.2025

import pandas as pd

# Wczytaj plik Excela z dwiema zakładkami
file_path = 'D:/Nowa_praca/KPO/dariah_new_model_deepseek-v3/dariah_new_model_deepseek-v3/Fragment_2_extracted_labels.xlsx'  # <-- zmień jeśli plik nazywa się inaczej
df_new = pd.read_excel(file_path, sheet_name='Sheet1')
df_old = pd.read_excel(file_path, sheet_name='old_eval')

# Zostaw tylko pierwsze wystąpienie dla każdego obj_id w starej zakładce
df_old_first = df_old.drop_duplicates(subset=['obj_id'], keep='first')

# Zostaw interesujące kolumny
df_new_subset = df_new[['obj_id', 'entity_text', 'concept', 'wikidata_label', 'wikidata_description']]
df_old_subset = df_old_first[['obj_id', 'entity_text', 'concept', 'wikidata_label', 'wikidata_description']]


# Scal po obj_id
comparison = pd.merge(df_new_subset, df_old_subset, on='obj_id', suffixes=('_new', '_old'))

# Porównania
comparison['entity_text_match'] = comparison['entity_text_new'] == comparison['entity_text_old']
comparison['wikidata_label_match'] = comparison['wikidata_label_new'] == comparison['wikidata_label_old']
comparison['concept_match'] = comparison['concept_new'] == comparison['concept_old']
output_file = 'porownanie_label_concept.xlsx'
comparison.to_excel(output_file, index=False)
# Podsumowanie
summary = {
    'total_compared': len(comparison),
    'entity_text_matches': comparison['entity_text_match'].sum(),
    'wikidata_label_matches': comparison['wikidata_label_match'].sum(),
    'concept_matches': comparison['concept_match'].sum()
}

print("📊 Podsumowanie porównania:")
for k, v in summary.items():
    print(f"- {k.replace('_', ' ').capitalize()}: {v}")

# Opcjonalnie: wyciągnij tylko konkretne przypadki
label_match_concept_mismatch = comparison[
    (comparison['wikidata_label_match']) & (~comparison['concept_match'])
]

print("\n🟨 Zgodne wikidata_label, niezgodne concept:")
print(label_match_concept_mismatch[['obj_id', 'entity_text_new', 'concept_new', 'concept_old', 'wikidata_label_new']])



#%% Obsluga niestandardowego JSONA
import ndjson

plik = "C:/Users/darek/Downloads/postagger_13.2_12_36/zeromski_zamiec.json"
with open(plik, "r", encoding="utf-8") as f:
    data = ndjson.load(f)  # wczytuje listę obiektów - każdy wiersz to 1 obiekt JSON

for obj in data:
    print(obj)
    
    
import os
import json
import ndjson
import pandas as pd

def process_files_in_dir_ndjson(input_dir: str, output_excel: str):
    """
    Przetwarza wszystkie pliki w katalogu `input_dir` jako NDJSON:
      - Każda linia w pliku to osobny obiekt JSON.
    Zbiorcze wyniki zapisuje do pliku Excel `output_excel`.
    """
    all_rows = []
    
    # Przechodzimy przez wszystkie elementy w katalogu
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        
        # Pomijamy katalogi - przetwarzamy tylko zwykłe pliki
        if not os.path.isfile(filepath):
            continue
        
        # Spróbujemy wczytać każdy plik jako NDJSON (kilka JSON-ów w liniach)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # ndjson.load() zwraca listę obiektów,
                # jeśli każda linia jest poprawnym JSON-em
                data_list = ndjson.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARN] Nie można wczytać pliku {filename} jako NDJSON: {e}")
            continue
        
        # Każdy element w data_list to osobny słownik, czyli obiekt JSON z jednej linii
        for data in data_list:
            # Pobierz 'id' tekstu, jeśli jest
            text_id = data.get("id", None)
            
            # Pełny tekst potrzebny do wycinków
            full_text = data.get("text", "")
            
            # Spany NER
            spans_ner = data.get("spans", {}).get("ner", [])
            
            # Dane o linkowaniu
            linking = data.get("records", {}).get("linking", {})
            clalink = linking.get("clalink", {})
            link_info_list = clalink.get("links", [])
            
            # Mapowanie obj_id -> najlepsze ID Wikidaty
            obj_id_to_wikidata = {}
            for link_info in link_info_list:
                obj_id = link_info["obj-id"]
                results = link_info.get("results", [])
                if results:
                    wikidata_id = results[0]["concept"]
                else:
                    wikidata_id = None
                obj_id_to_wikidata[obj_id] = wikidata_id
            
            # Wyciągamy poszczególne spany
            for span in spans_ner:
                obj_id = span["id"]
                start = span["start"]
                stop = span["stop"]
                ner_type = span.get("type", None)
                
                # Fragment tekstu
                entity_text = full_text[start:stop]
                
                # ID Wikidaty
                wikidata_id = obj_id_to_wikidata.get(obj_id)
                
                row = {
                    "file_name": filename,       # Nazwa pliku
                    "text_id": text_id,         # ID tekstu z JSON
                    "obj_id": obj_id,           # ID spana
                    "entity_text": entity_text,
                    "start": start,
                    "stop": stop,
                    "wikidata_id": wikidata_id,
                    "ner_type": ner_type
                }
                all_rows.append(row)
    
    # Gdy mamy już wszystkie wiersze, tworzymy jedną ramkę danych
    df = pd.DataFrame(all_rows)
    
    # Zapisujemy do Excela
    df.to_excel(output_excel, index=False)
    
    if not df.empty:
        liczba_rekordow = len(df)
        liczba_plikow = df["file_name"].nunique()
        print(f"Zapisano {liczba_rekordow} rekordów z {liczba_plikow} plików do pliku: {output_excel}")
    else:
        print("Brak danych NDJSON do zapisania. Ramka danych jest pusta.")

if __name__ == "__main__":
    # Przykład użycia
    process_files_in_dir_ndjson(
        input_dir="C:/Users/darek/Downloads/postagger_13.2_12_36/",
        output_excel="zbiory_ner_ndjson.xlsx"
    )


import pandas as pd

# 1. Wczytanie pliku A (z kolumną "true/false")
dfA = pd.read_excel("C:/Users/darek/Downloads/KPO_postagger_ewaluacja.xlsx") 
# Załóżmy, że w dfA są kolumny: 
# [entity_text, wikidata_label, concept, true/false]
# (oraz ewentualnie inne)

# 2. Wczytanie pliku B (bez "true/false", ale z pełnymi danymi)
dfB = pd.read_excel("C:/Users/darek/Downloads/KPO_postagger_test_backup.xlsx") 
# Zakładamy, że w dfB są m.in.:
# [text_id, context_snippet, obj_id, entity_text, wikidata_label, concept, 
#  wiki_Click_URL, wikidata_url, itp.]
# Bez kolumny true/false

# 3. Tworzymy "subset" z dfA, by nie nadpisywać 
#    ewentualnie niepotrzebnych kolumn
dfA_sub = dfA[["entity_text", "wikidata_label", "concept", "true/false"]]

# 4. Łączymy (merge) dfB z dfA_sub:
#    - how="left" => chcemy zachować wszystkie wiersze z dfB
#    - on=... => łączymy po 3 kolumnach:
#      entity_text, wikidata_label, concept
df_merged = pd.merge(
    dfB,
    dfA_sub,
    on=["entity_text", "wikidata_label", "concept"],
    how="left"
)

# 5. Zapisujemy połączoną tabelę do nowego pliku
df_merged.to_excel("KPO_postagger_ewaluacja_full.xlsx", index=False)

print("Gotowe! Zapisano df_merged z dołączoną kolumną true/false.")


import os
import json
import pandas as pd

import os
import json
import pandas as pd
import shutil

# Ścieżki (dostosuj do swoich potrzeb)
input_dir = r"D:/Nowa_praca/KPO/postagger/fragmenty artykułów dla CLARIN-u-20250218T071948Z-001/jsony/"
output_dir = r"D:/Nowa_praca/KPO/postagger/fragmenty artykułów dla CLARIN-u-20250218T071948Z-001/Jsony_True_False/"
excel_file = r"KPO_postagger_ewaluacja_full.xlsx"

# Tworzymy katalog wynikowy (jeśli nie istnieje)
os.makedirs(output_dir, exist_ok=True)

# Wczytujemy Excela
df_excel = pd.read_excel(excel_file)
filtered_df = df_excel[df_excel['text_id'] == '6b3dc02d-4379-4a78-82b0-a77bc9193ad1']

# wyświetlenie wyniku
print(filtered_df)


# Upewnijmy się, że true/false są tekstem
df_excel['true/false'] = df_excel['true/false'].astype(str).str.upper()

# Tworzymy słownik ułatwiający wyszukiwanie wartości true/false
tf_dict = {}
for idx, row in df_excel.iterrows():
    key = (row['text_id'], row['obj_id'], row['concept'])
    
    if pd.isna(row['true/false']):
        tf_dict[key] = "?"
    elif row['true/false'] == '1.0':
        tf_dict[key] = "T"
    elif row['true/false'] == '0.0':
        tf_dict[key] = "F"
    else:
        tf_dict[key] = "?"
        

count = sum(1 for v in tf_dict.values() if v == "?")
print(f'Ilość znaków "?": {count}')

selected_text_id = "6b3dc02d-4379-4a78-82b0-a77bc9193ad1"

# filtrowanie słownika
filtered_dict = {key: val for key, val in tf_dict.items() if key[0] == selected_text_id}

# sprawdzenie wyników
for key, val in filtered_dict.items():
    print(key, val)

# Przetwarzanie każdego pliku JSON
for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    input_filepath = os.path.join(input_dir, filename)
    output_filepath = os.path.join(output_dir, filename)

    # Wczytujemy oryginalny JSON
    with open(input_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    text_id = data.get('id', None)
    if text_id is None:
        print(f"Brak text_id w pliku: {filename}")
        continue

    # Iterujemy po obj-id w JSON-ie i dodajemy T/F
    links = data.get('records', {}).get('linking', {}).get('clalink', {}).get('links', [])
    for link in links:
        obj_id = link['obj-id']
        print(obj_id)
        for result in link['results']:
            concept = result['concept']
            # Wyszukujemy wartość T/F
            key = (text_id, obj_id, concept)
            tf_value = tf_dict.get(key, None)

            if tf_value is not None:
                result["T/F"] = tf_value  # Dodajemy T/F
            else:
                result["T/F"] = '?'  # Jeśli nie ma dopasowania

    # Zapisujemy zmodyfikowany JSON do nowego katalogu
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)

print("Przetwarzanie zakończone.")




    


