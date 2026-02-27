# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:51:06 2026

@author: darek
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:44:51 2024

@author: dariu
"""
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import logging
import pandas as pd
import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoConfig
import os

# Załaduj konfigurację modelu Herbert
config = AutoConfig.from_pretrained("allegro/herbert-base-cased")

# Wyświetl maksymalną liczbę tokenów
print("Maksymalna liczba tokenów dla modelu Herbert:", config.max_position_embeddings)

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "hasła przedmiotowe"]):
    # Wczytanie danych z pliku JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)

    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']]

    # Konwersja wartości w kolumnie 'Tekst artykułu' na stringi
    df_json['Tekst artykułu'] = df_json['Tekst artykułu'].astype(str)

    # Wczytanie danych z pliku Excel
    df_excel = pd.read_excel(excel_file_path)

    # Połączenie DataFrame'ów
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")

    # Konwersja wartości w kolumnach 'Tytuł artykułu' i 'Tekst artykułu' na stringi w połączonym DataFrame
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)
    if 'do PBL' in merged_df.columns and 'hasła przedmiotowe' in merged_df.columns:
        # Filtracja rekordów, gdzie 'do PBL' jest True
        merged_df = merged_df[merged_df['do PBL'] == True]
        
        # Ograniczenie do wybranych kolumn i usunięcie wierszy z pustymi wartościami w 'hasła przedmiotowe'
        selected_columns = merged_df[selected_columns_list]
        selected_columns = selected_columns.dropna(subset=['hasła przedmiotowe'])
    
        return selected_columns
    else:
        # Jeśli wymagane kolumny nie istnieją, zwróć None lub pusty DataFrame
        return None  # Lub: return pd.DataFrame(columns=selected_columns_list)


# json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
# excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
# json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
# excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
# json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
# excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'
# json_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afront_2022-09-08.json'
# excel_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afront_2022-09-08.xlsx'
# json_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/artpapier_2022-10-05.json'
# excel_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/artpapier_2022-10-05.xlsx'
# json_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/audycjekulturalne_2022-10-11.json'
# excel_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/audycjekulturalne_2022-10-11.xlsx'
# json_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/bylam_widzialam_2023-02-21.json'
# excel_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/bylam_widzialam_2023-02-21.xlsx'
# json_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/czas_kultury_2023-03-24.json'
# excel_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/czas_kultury_2023-03-24.xlsx'
# json_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_dziennik_2023-10-23.json'
# excel_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_dziennik_2023-10-23.xlsx'
# json_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/intimathule_2022-09-09.json'
# excel_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/intimathule_2022-09-09.xlsx'
# json_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/jerzy_sosnowski_2022-09-09.json'
# excel_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jerzy_sosnowski_2022-09-09.xlsx'
# json_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/komnen_kastamonu_2022-09-12.json'
# excel_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/komnen_kastamonu_2022-09-12.xlsx'
# json_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/krzysztof_jaworski_2022-12-08.json'
# excel_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/krzysztof_jaworski_2022-12-08.xlsx'
# json_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pgajda_2022-09-13.json'
# excel_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pgajda_2022-09-13.xlsx'
# json_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/poeci_po_godzinach_2022-09-14.json'
# excel_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/poeci_po_godzinach_2022-09-14.xlsx'
# json_file_path16 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/biuroliterackie_biblioteka_2022-11-08.json'
# excel_file_path16 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/biuroliterackie_2022-11-08.xlsx'
# json_file_path17 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/chalwazwyciezonym_2023-02-01.json'
# excel_file_path17 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/chalwazwyciezonym_2023-02-01.xlsx'
# json_file_path18 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/cultureave_2023-02-20.json'
# excel_file_path18 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/cultureave_2023-10-12.xlsx'
# json_file_path19 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/eteatr_2023-10-12.json'
# excel_file_path19 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/eteatr_2023-10-12.xlsx'
# json_file_path20 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_org_pl_2023-02-06.json'
# excel_file_path20 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_org_pl_2023-02-06.xlsx'
# json_file_path21 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/gazetakulturalnazelow_2023-10-26.json'
# excel_file_path21 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/gazetakulturalnazelow_2023-10-26.xlsx'
# json_file_path22 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/hiperrealizm_2023-11-07.json'
# excel_file_path22 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/hiperrealizm_2023-11-07.xlsx'
# json_file_path23 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/kempinsky_2023-11-06.json'
# excel_file_path23 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/kempinsky_2023-11-06.xlsx'
# json_file_path24 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/kochampolskiekino_2023-02-02.json'
# excel_file_path24 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/kochampolskiekino_2023-02-02.xlsx'
# json_file_path25 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/martafox_2023-10-06.json'
# excel_file_path25 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/martafox_2023-10-06.xlsx'
# # ... więcej plików w razie potrzeby

# # Użycie funkcji
# df1 = load_and_merge_data(json_file_path, excel_file_path)
# df2 = load_and_merge_data(json_file_path2, excel_file_path2)
# df3 = load_and_merge_data(json_file_path3, excel_file_path3)
# df4 = load_and_merge_data(json_file_path4, excel_file_path4)
# df5 = load_and_merge_data(json_file_path5, excel_file_path5)
# df6 = load_and_merge_data(json_file_path6, excel_file_path6)
# df7 = load_and_merge_data(json_file_path7, excel_file_path7)
# df8 = load_and_merge_data(json_file_path8, excel_file_path8)
# df9 = load_and_merge_data(json_file_path9, excel_file_path9)
# df10 = load_and_merge_data(json_file_path10, excel_file_path10)
# df11 = load_and_merge_data(json_file_path11, excel_file_path11)
# df12 = load_and_merge_data(json_file_path12, excel_file_path12)
# df13 = load_and_merge_data(json_file_path13, excel_file_path13)
# df14 = load_and_merge_data(json_file_path14, excel_file_path14)
# df15 = load_and_merge_data(json_file_path15, excel_file_path15)
# df16 = load_and_merge_data(json_file_path16, excel_file_path16)
# df17 = load_and_merge_data(json_file_path17, excel_file_path17)
# df18 = load_and_merge_data(json_file_path18, excel_file_path18)
# df19= load_and_merge_data(json_file_path19, excel_file_path19)
# df20= load_and_merge_data(json_file_path20, excel_file_path20)
# df21= load_and_merge_data(json_file_path21, excel_file_path21)
# df22= load_and_merge_data(json_file_path22, excel_file_path22)
# df23= load_and_merge_data(json_file_path23, excel_file_path23)
# df24= load_and_merge_data(json_file_path24, excel_file_path24)
# df25= load_and_merge_data(json_file_path25, excel_file_path25)
# # ... wczytanie kolejnych par plików

# # Połączenie wszystkich DataFrame'ów
# df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25], ignore_index=True)
#%%
base_dir = 'D:/Nowa_praca/dane_model_jezykowy/kopia_dla_UAM/'
json_dir = os.path.join(base_dir, 'Json')

# Pobranie list plików JSON i Excel
json_files = {os.path.splitext(f)[0]: os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')}
excel_files = {os.path.splitext(f)[0]: os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.xlsx')}

# Znalezienie wspólnych nazw plików
common_files = set(json_files.keys()).intersection(excel_files.keys())

# Automatyczne wczytywanie i łączenie danych
merged_dfs = []
for file_name in common_files:
    json_path = json_files[file_name]
    excel_path = excel_files[file_name]
    print(f"Przetwarzanie pary: JSON - {json_path}, Excel - {excel_path}")
    merged_df = load_and_merge_data(json_path, excel_path)
    if merged_df is not None:
        merged_dfs.append(merged_df)

# Połączenie wszystkich wynikowych DataFrame'ów w jeden
if merged_dfs:
    df = pd.concat(merged_dfs, ignore_index=True)
    
print(f"Liczba WSZYSTKICH wczytanych artykułów: {len(df)}")
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)
df_excel = pd.read_excel('C:/Users/darek/Downloads/Mapowanie działów.xlsx')
df_excel['połączony dział'] = df_excel['nr działu'].astype(str) + " " + df_excel['nazwa działu']

mapowanie = pd.Series(df_excel['string uproszczony'].values, index=df_excel['połączony dział']).to_dict()

# Użycie mapowania do stworzenia nowej kolumny w df
df['rozwiniete_haslo'] = df['hasła przedmiotowe'].map(mapowanie)
#%%rzeczy zle liczone
# wartosci = df['hasła przedmiotowe'].str.split(r'[;,]', expand=True).stack().str.strip()



# # Zlicz wystąpienia każdej wartości
# liczba_wystapien = wartosci.value_counts()
# liczba_wystapien_sum = wartosci.value_counts().sum()
# print (liczba_wystapien_sum)
# ilosc_gatunkow = liczba_wystapien.index.nunique()

# wartosci_rozwiniete = wartosci.map(mapowanie)
# wartosci_rozwiniete = wartosci_rozwiniete.dropna()
# liczba_wystapien = wartosci_rozwiniete.value_counts()
# liczba_wystapien_sum = liczba_wystapien.sum()
# ilosc_gatunkow = liczba_wystapien.index.nunique()
#%%
MIN_COUNT = 100

# --- SUROWE HASŁA ---
wartosci_surowe = (
    df['hasła przedmiotowe']
    .dropna()
    .str.split(r'[;,]')
    .explode()
    .str.strip()
)

counts_surowe = wartosci_surowe.value_counts()
surowe_sum=counts_surowe.value_counts().sum()
counts_surowe_100 = counts_surowe[counts_surowe >= MIN_COUNT]

print("\n=== SUROWE HASŁA ===")
print("Unikalne hasła (wszystkie):", counts_surowe.index.nunique())
print("Unikalne hasła (>=100):", counts_surowe_100.index.nunique())
print("Liczba przypisań (>=100):", counts_surowe_100.sum())



# --- HASŁA ROZWINIĘTE / DZIAŁY ---
wartosci_rozwiniete = wartosci_surowe.map(mapowanie)
wartosci_rozwiniete = wartosci_rozwiniete.dropna()

counts_rozwiniete = wartosci_rozwiniete.value_counts()
counts_rozwiniete_100 = counts_rozwiniete[counts_rozwiniete >= MIN_COUNT]

print("\n=== HASŁA ROZWINIĘTE (DZIAŁY) ===")
print("Unikalne działy (wszystkie):", counts_rozwiniete.index.nunique())
print("Unikalne działy (>=100):", counts_rozwiniete_100.index.nunique())
print("Liczba przypisań (>=100):", counts_rozwiniete_100.sum())
df = df.dropna(subset=['hasła przedmiotowe'])
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']


df = df.assign(hasła_przedmiotowe_split=df['hasła przedmiotowe'].str.split(';')).explode('hasła_przedmiotowe_split')

# Usuń zbędne białe znaki z hasła przedmiotowego
df['hasła_przedmiotowe_split'] = df['hasła_przedmiotowe_split'].str.strip()

# Opcjonalnie: Usuń wiersze z pustymi hasłami po stripowaniu
df = df[df['hasła_przedmiotowe_split'] != '']

# Zresetuj indeksy DataFrame
df = df.reset_index(drop=True)
label_counts = (
    df['hasła_przedmiotowe_split']
    .value_counts()
)
print("\n=== HASŁA PRZEDMIOTOWE – TOP 20 (PO CZYSZCZENIU) ===")
for haslo, liczba in label_counts.head(20).items():
    print(f"{haslo}: {liczba}")
    
wartosci_surowe = (
    df['hasła przedmiotowe']
    .dropna()
    .str.split(r'[;,]')
    .explode()
    .str.strip()
)

# usunięcie pustych
wartosci_surowe = wartosci_surowe[wartosci_surowe != ""]

counts_surowe = wartosci_surowe.value_counts()

print("\n=== SUROWE HASŁA PRZEDMIOTOWE (PRZED OGRANICZENIEM) ===")
print(f"Liczba przypisań (łącznie): {counts_surowe.sum()}")
print(f"Liczba unikalnych haseł: {counts_surowe.index.nunique()}")

print("\nTOP 20 haseł (wg liczby przypisań):")
for haslo, liczba in counts_surowe.head(20).items():
    print(f"{haslo}: {liczba}")
labels_ok = label_counts[label_counts >= MIN_COUNT].index

df = df[df['hasła_przedmiotowe_split'].isin(labels_ok)].copy()

print("Po odcięciu MIN_COUNT:")
print("  Liczba próbek:", len(df))
print("  Liczba klas:", df['hasła_przedmiotowe_split'].nunique())
# Kodowanie etykiet
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['hasła_przedmiotowe_split'])

# Przygotuj tokenizator i model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels=len(label_encoder.classes_),
#     problem_type="single_label_classification"
# )

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-base-cased",
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
)


# Funkcja do tokenizacji i kodowania danych
def tokenize_and_encode(examples):
    return tokenizer(examples['combined_text'], padding='max_length', truncation=True, max_length=512)

# Mapowanie funkcji tokenizującej do datasetu
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_encode, batched=True)

# Podział na zbiór treningowy i walidacyjny
tmp = dataset.train_test_split(test_size=0.2)
train_valid = tmp['train'].train_test_split(test_size=0.125)

dataset_dict = DatasetDict({
    'train': train_valid['train'],   # ~70%
    'eval':  train_valid['test'],    # ~10%
    'test':  tmp['test']              # ~20%
})
eval_dataset=dataset_dict['eval']
# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir="G:/hasla_model/checkpoints",
    num_train_epochs=20,
    per_device_train_batch_size=7,
    per_device_eval_batch_size=7,
    warmup_steps=500,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    logging_dir="G:/hasla_model/logs",
    save_total_limit=2   # <<< bardzo ważne
)


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
from transformers import EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['eval'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
print(f"Liczba próbek do TRENINGU: {len(dataset_dict['train'])}")
print(f"Liczba próbek do WALIDACJI: {len(dataset_dict['eval'])}")
print(f"Liczba próbek do TESTÓW: {len(dataset_dict['test'])}")
trainer.train()
# Ewaluacja modelu
results = trainer.evaluate()

# Wyniki
print(results)

from sklearn.metrics import classification_report, confusion_matrix

# ======================
# PREDYKCJE NA ZBIORZE TESTOWYM
# ======================
predictions = trainer.predict(dataset_dict["test"])

y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(axis=1)

print("\n=== CLASSIFICATION REPORT (TEST SET) ===")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )
)

df_preds = pd.DataFrame({
    "true_label": label_encoder.inverse_transform(y_true),
    "pred_label": label_encoder.inverse_transform(y_pred)
})





# ======================
# ZAPIS NAJLEPSZEGO MODELU (EARLY STOPPING)
# ======================
OUTPUT_DIR = "G:/hasla_model/"

predictions_path = os.path.join(OUTPUT_DIR, "predictions_test_set.csv")
df_preds.to_csv(predictions_path, index=False, encoding="utf-8")

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


import joblib
label_encoder_path = os.path.join(
    OUTPUT_DIR,
    "label_encoder_hasla_best_spyder.joblib"
)
joblib.dump(label_encoder, label_encoder_path)
