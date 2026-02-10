# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 16:25:26 2026

@author: darek
"""

import pandas as pd
import json
import torch
import logging
import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    BertTokenizer, 
    EarlyStoppingCallback, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    AutoModel,
    HerbertTokenizerFast
)
from datasets import Dataset, DatasetDict

# --- FUNKCJA ŁADUJĄCA ---
def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', 'forma/gatunek', 'do PBL']):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)
    df_json = df_json[['Link', 'Tekst artykułu']]
    df_json['Tekst artykułu'] = df_json['Tekst artykułu'].astype(str)

    df_excel = pd.read_excel(excel_file_path)
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)

    if 'do PBL' in merged_df.columns:
        filtered_df = merged_df[merged_df['do PBL'] == True]
        if 'forma/gatunek' in filtered_df.columns:
            selected_columns = filtered_df[selected_columns_list]
            selected_columns = selected_columns.dropna(subset=['forma/gatunek'])
            return selected_columns
    return pd.DataFrame(columns=selected_columns_list)


# --- KONFIGURACJA ŚCIEŻEK ---
base_dir = 'D:/Nowa_praca/dane_model_jezykowy/kopia_dla_UAM/'
json_dir = os.path.join(base_dir, 'Json')
json_files = {os.path.splitext(f)[0]: os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith('.json')}
excel_files = {os.path.splitext(f)[0]: os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.xlsx')}
common_files = set(json_files.keys()).intersection(excel_files.keys())

# --- WCZYTYWANIE ---
merged_dfs = []
for file_name in common_files:
    json_path = json_files[file_name]
    excel_path = excel_files[file_name]
    # print(f"Przetwarzanie: {file_name}") # Odkomentuj jeśli chcesz widzieć pliki
    merged_df = load_and_merge_data(json_path, excel_path)
    if merged_df is not None:
        merged_dfs.append(merged_df)

if merged_dfs:
    df = pd.concat(merged_dfs, ignore_index=True)
    
print(f"Liczba WSZYSTKICH wczytanych artykułów: {len(df)}")    
    
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

# --- CZYSZCZENIE I EXPLODE ---
df = df.dropna(subset=['forma/gatunek'])

# Statystyka przed explode
print(f"Liczba unikalnych artykułów z gatunkami (przed explode): {len(df)}")

# Rozbijanie gatunków po przecinku i średniku
df = df.assign(forma_gatunek_split=df['forma/gatunek'].str.split(r'[;,]')).explode('forma_gatunek_split')

# Czyszczenie białych znaków i pustych stringów
df['forma_gatunek_split'] = df['forma_gatunek_split'].str.strip()
df = df[df['forma_gatunek_split'] != '']
df = df.reset_index(drop=True)

# === TUTAJ DODAŁEM WYDRUK LISTY WSZYSTKICH GATUNKÓW ===
print("\n=== GATUNKI – WSZYSTKIE (PRZED FILTRACJĄ) ===")
counts_all = df['forma_gatunek_split'].value_counts()
print(f"Liczba próbek: {len(df)}")
print(f"Liczba unikalnych gatunków: {len(counts_all)}")
print("\n--- Lista wystąpień ---")
for gatunek, liczebnosc in counts_all.items():
    print(f"{gatunek}: {liczebnosc}")


# --- OGRANICZENIE MIN_COUNT ---
min_samples = 100
classes_to_keep = counts_all[counts_all >= min_samples].index

# Zapamiętaj stan przed
n_samples_before = len(df)
n_classes_before = df['forma_gatunek_split'].nunique()

# Filtracja
df_filtered = df[df['forma_gatunek_split'].isin(classes_to_keep)].copy()
df = df_filtered.reset_index(drop=True)

# === WYDRUK LISTY PO FILTRACJI ===
print(f"\n=== GATUNKI – PO OGRANICZENIU (MIN_COUNT = {min_samples}) ===")
counts_final = df['forma_gatunek_split'].value_counts()
print(f"Liczba próbek: {len(df)}")
print(f"Liczba gatunków: {len(counts_final)}")
print("\n--- Pozostałe gatunki ---")
for gatunek, liczebnosc in counts_final.items():
    print(f"{gatunek}: {liczebnosc}")

print("\nUsunięto przez MIN_COUNT:")
print(f"  • próbek: {n_samples_before - len(df)}")
print(f"  • gatunków: {n_classes_before - len(counts_final)}")


# --- PRZYGOTOWANIE DO TRENINGU ---
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['forma_gatunek_split'])

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-base-cased",
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
)

def tokenize_and_encode(examples):
    return tokenizer(
        examples['combined_text'],
        padding='max_length',
        truncation=True,
        max_length=514
    )

dataset = Dataset.from_pandas(df[['combined_text', 'labels']])
dataset = dataset.map(tokenize_and_encode, batched=True)
dataset = dataset.remove_columns(["combined_text"])
dataset = dataset.rename_column("labels", "label")
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Podział: Train / Eval / Test
tmp = dataset.train_test_split(test_size=0.2)
train_valid = tmp['train'].train_test_split(test_size=0.125)

dataset_dict = DatasetDict({
    'train': train_valid['train'],   
    'eval':  train_valid['test'],    
    'test':  tmp['test']             
})

print("\n=== LICZEBNOŚĆ ZBIORÓW ===")
print(f"Liczba próbek do TRENINGU: {len(dataset_dict['train'])}")
print(f"Liczba próbek do WALIDACJI: {len(dataset_dict['eval'])}")
print(f"Liczba próbek do TESTÓW: {len(dataset_dict['test'])}")

# --- TRENING ---
training_args = TrainingArguments(
    output_dir="G:/gatunki_model/",
    num_train_epochs=10,             
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,     
    metric_for_best_model="f1",      
    greater_is_better=True,
    save_total_limit=2               
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['eval'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# --- RAPORT KOŃCOWY ---
print("\n...Generowanie raportu na zbiorze TESTOWYM...")
predictions = trainer.predict(dataset_dict["test"])
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(axis=1)

print("\n=== CLASSIFICATION REPORT (TEST SET) ===")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4, zero_division=0))

# Zapisywanie
OUTPUT_DIR = "G:/gatunki_model/"
df_preds = pd.DataFrame({
    "true_label": label_encoder.inverse_transform(y_true),
    "pred_label": label_encoder.inverse_transform(y_pred)
})
df_preds.to_csv(os.path.join(OUTPUT_DIR, "wyniki_gatunki_test.csv"), index=False, encoding="utf-8")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder_gatunki.joblib"))