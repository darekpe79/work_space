#%% CR -- kod własciwy

import os
import glob
import time
from typing import List
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    PartOfSpeech,
)
#%% właciwy kod -- 8.12.2025
# =========================
# KONFIGURACJA ŚCIEŻEK
# =========================
STOPWORDS_PATH = r"C:/Users/darek/Downloads/stopwords-pl_21112025.txt"
FOLDER_PATH = r"C:/Users/darek/Downloads/polonisci/"
OUTPUT_XLSX = r"C:/Users/darek/Downloads/polonisci/topiki_bertopic_po_spotkaniu_v3.xlsx"

# Maksymalna liczba plików do przetworzenia (None = wszystkie)
MAX_FILES = 500  # możesz zmienić na None, jeśli chcesz wszystkie
start_file, end_file = 0, 500


# =========================
# FUNKCJE POMOCNICZE
# =========================
def load_stopwords(file_path: str, encoding: str = "utf-8") -> List[str]:
    with open(file_path, "r", encoding=encoding) as file:
        return [line.strip() for line in file if line.strip()]


# =========================
# WSTĘPNA KONFIGURACJA MODELI
# =========================

# Stopwordy
POLISH_STOPWORDS = load_stopwords(STOPWORDS_PATH)

# Model embeddingów po polsku (ten sam, co u Ciebie)
polish_st = "sdadas/st-polish-paraphrase-from-distilroberta"
embedding_model = SentenceTransformer(polish_st)

# UMAP
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

# HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size=25,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# Vectorizer – jak w Twoim pierwotnym kodzie
vectorizer_model = CountVectorizer(
    stop_words=POLISH_STOPWORDS,
    min_df=2,
    ngram_range=(1, 3),
)

# Reprezentacje tematów
keybert_model = KeyBERTInspired()
pos_model = PartOfSpeech("pl_core_news_lg")
mmr_model = MaximalMarginalRelevance(diversity=0.3)

representation_model = {
    "KeyBERT": keybert_model,
    "MMR": mmr_model,
    "POS": pos_model,
}

#%%
# =========================
# WŁAŚCIWY PIPELINE
# =========================
def main():
    # --- Wczytywanie tekstów: KAŻDA NIEPUSTA LINIA = OSOBNY DOKUMENT ---
    start = time.time()

    txt_files = glob.glob(os.path.join(FOLDER_PATH, "*.txt"))
    txt_files = sorted(txt_files)

    if MAX_FILES is not None:
        txt_files = txt_files[start_file:end_file]

    processed_texts: List[str] = []      # linie (dokumenty)
    processed_texts_ids: List[str] = []  # id pliku dla każdej linii
    doc_ids_unique: List[str] = []       # unikalne ID plików (teksty)

    for file_path in tqdm(txt_files, desc="Wczytywanie tekstów (linie jako dokumenty)"):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            cleaned = [line.strip() for line in lines if line.strip()]
            text_id = os.path.basename(file_path).replace(".txt", "")

            # zapamiętujemy unikalne ID tekstu (raz na plik)
            doc_ids_unique.append(text_id)

            # każda linia to osobny dokument
            for line in cleaned:
                processed_texts.append(line)
                processed_texts_ids.append(text_id)

    print(f"Wczytano {len(processed_texts)} dokumentów (linii) z {len(txt_files)} plików.")

    if not processed_texts:
        raise ValueError("Brak tekstów do przetworzenia. Sprawdź ścieżkę FOLDER_PATH.")

    # --- Topic modeling (BERTopic) ---
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=20,           # więcej słów, żeby spokojnie mieć 15
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(processed_texts)

    end = time.time()
    print(
        f"Czas wykonania dla {len(processed_texts)} dokumentów (linii): "
        f"{end - start:.2f} sekundy."
    )

    # =========================
    # GENEROWANIE ARKUSZY XLSX
    # =========================

    topics_dict = topic_model.get_topics()  # {topic_id: [(word, weight), ...]}

    if not topics_dict:
        raise ValueError(
            "BERTopic nie wygenerował żadnych tematów (topics_dict jest puste). "
            "Spróbuj zmienić parametry HDBSCAN albo sprawdzić teksty."
        )

    all_topic_ids = sorted(topics_dict.keys())

    # Tematy "normalne" (bez -1) i z niepustą reprezentacją
    valid_topic_ids = sorted(
        t for t in all_topic_ids
        if t != -1 and topics_dict[t]
    )

    if not valid_topic_ids:
        valid_topic_ids = sorted(
            t for t in all_topic_ids
            if topics_dict[t]
        )
        print(
            "Uwaga: BERTopic nie wygenerował żadnych tematów poza -1 "
            "(wszystko trafiło do szumu). Eksportuję -1 jako Topic 1."
        )

    if not valid_topic_ids:
        raise ValueError(
            "BERTopic nie wygenerował żadnych niepustych tematów. "
            "Spróbuj zmienić parametry (np. min_cluster_size) "
            "lub sprawdzić, czy dokumenty nie są zbyt krótkie/puste."
        )

    # Mapowanie: wewnętrzny topic_id -> etykieta 1..K (jak w Excelu)
    topic_id_to_label = {topic_id: i + 1 for i, topic_id in enumerate(valid_topic_ids)}

    # =========================
    # PRZYGOTOWANIE MACIERZY PROB (THETA) NA POZIOMIE LINII
    # =========================
    n_docs_lines = len(processed_texts)
    n_valid_topics = len(valid_topic_ids)
    topic_id_to_idx = {tid: i for i, tid in enumerate(valid_topic_ids)}

    theta_lines = np.zeros((n_docs_lines, n_valid_topics), dtype=float)

    probs_mode = "none"
    if probs is not None:
        probs_array = np.asarray(probs, dtype=object)

        # 2D (pełna macierz)
        if probs_array.ndim == 2:
            probs_mode = "2d"
            if probs_array.dtype == object:
                probs_array = np.vstack(probs_array)

            # zakładamy: kolumny = tematy (bez -1), w kolejności rosnącej
            if probs_array.shape[1] >= n_valid_topics:
                theta_lines = probs_array[:, :n_valid_topics]
            else:
                m = probs_array.shape[1]
                theta_lines[:, :m] = probs_array

        # 1D – tylko P(przypisany_topik)
        elif probs_array.ndim == 1:
            probs_mode = "1d"
            assigned_probs = np.array(
                [float(x) for x in probs_array], dtype=float
            )
            for i, (t, p) in enumerate(zip(topics, assigned_probs)):
                if t in topic_id_to_idx:
                    theta_lines[i, topic_id_to_idx[t]] = p

    # jeśli probs = None albo nic nie weszło: twarde przypisanie
    if probs_mode == "none":
        for i, t in enumerate(topics):
            if t in topic_id_to_idx:
                theta_lines[i, topic_id_to_idx[t]] = 1.0

    # =========================
    # AGREGACJA: Z LINII DO POZIOMU TEKSTU
    # =========================
    doc_to_indices = defaultdict(list)
    for idx_line, doc_id in enumerate(processed_texts_ids):
        doc_to_indices[doc_id].append(idx_line)

    n_docs_texts = len(doc_ids_unique)
    theta_texts = np.zeros((n_docs_texts, n_valid_topics), dtype=float)

    for j, doc_id in enumerate(doc_ids_unique):
        idxs = doc_to_indices.get(doc_id, [])
        if not idxs:
            continue
        theta_texts[j] = theta_lines[idxs].mean(axis=0)

    # =========================
    # ARKUSZE
    # =========================

    # ---------- Arkusz 1: "top 15 słówtopik" ----------
    N_TOP_WORDS_WIDE = 15

    top_words_wide = {}
    for topic_id in valid_topic_ids:
        label_idx = topic_id_to_label[topic_id]
        # bierzemy pierwsze 15 słów z reprezentacji (a mamy top_n_words=20)
        words_weights = topics_dict[topic_id][:N_TOP_WORDS_WIDE]
        words = [w for (w, _) in words_weights]
        # gdyby jakimś cudem było mniej niż 15, dopadamy None
        if len(words) < N_TOP_WORDS_WIDE:
            words += [None] * (N_TOP_WORDS_WIDE - len(words))
        top_words_wide[f"Topic {label_idx}"] = words

    df_top15 = pd.DataFrame(top_words_wide)

    # ---------- Arkusz 2: "top 10 słów + beta" ----------
    N_TOP_WORDS_LONG = 10

    rows_long = []
    for topic_id in valid_topic_ids:
        label_idx = topic_id_to_label[topic_id]
        for word, weight in topics_dict[topic_id][:N_TOP_WORDS_LONG]:
            rows_long.append(
                {
                    "topic": label_idx,
                    "term": word,
                    "beta": float(weight),
                }
            )

    if rows_long:
        df_top10_beta = pd.DataFrame(rows_long)[["topic", "term", "beta"]]
    else:
        df_top10_beta = pd.DataFrame(columns=["topic", "term", "beta"])

    # ---------- Arkusz 3: "topikiteksty" ----------
    max_probs_text = theta_texts.max(axis=1)
    best_topic_idx_text = theta_texts.argmax(axis=1)

    top_topics_labels_text = []
    for i, max_p in enumerate(max_probs_text):
        if max_p > 0:
            label = topic_id_to_label[valid_topic_ids[best_topic_idx_text[i]]]
        else:
            label = None
        top_topics_labels_text.append(label)

    df_topiki_teksty = pd.DataFrame(
        {
            "dokument": doc_ids_unique,       # każdy tekst dokładnie raz
            "Top topik": top_topics_labels_text,
        }
    )

    # ---------- Arkusz 4: "theta" ----------
    theta_df = pd.DataFrame(
        theta_texts,
        columns=[topic_id_to_label[topic_id] for topic_id in valid_topic_ids],
    )
    theta_df = theta_df[sorted(theta_df.columns)]
    theta_df.insert(0, "dokument", doc_ids_unique)

    # ---------- Zapis do Excela ----------
    os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        df_top15.to_excel(writer, sheet_name="top 15 słówtopik", index=False)
        df_top10_beta.to_excel(writer, sheet_name="top 10 słów + beta", index=False)
        df_topiki_teksty.to_excel(writer, sheet_name="topikiteksty", index=False)
        theta_df.to_excel(writer, sheet_name="theta", index=False)

    print(f"Zapisano rezultaty do: {OUTPUT_XLSX}")


#%%
start = time.time()

if __name__ == "__main__":
    main()

end = time.time()

print(f"Czas wykonania dla {MAX_FILES} tekstów: {end - start:.4f} sekundy.")
