# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:31:56 2025

@author: darek
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
import joblib, torch

model_id = "darekpe79/true-false-pbl-herbert"

# 1️⃣  Model i tokenizer
model     = AutoModelForSequenceClassification.from_pretrained(model_id, use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2️⃣  LabelEncoder – pobieramy pojedynczy plik z repo
encoder_path  = hf_hub_download(repo_id=model_id, filename="label_encoder.joblib")
label_encoder = joblib.load(encoder_path)

# 3️⃣  Przykładowy tekst
text = '''SPRAWY RODZINNE, A NAWET RODOWE (DOROTHY L. SAYERS: 'ZASTĘPY ŚWIADKÓW')  '''
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

pred_id = outputs.logits.argmax(-1).item()
label   = label_encoder.inverse_transform([pred_id])[0]

print("Predykcja modelu:", label)      # → True / False




import os, json, joblib, torch, logging
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# 1.  Funkcja  load_and_merge_data  (bez zmian)
# --------------------------------------------------
def load_and_merge_data(
        json_file_path,
        excel_file_path,
        common_column='Link',
        selected_columns_list=['Tytuł artykułu',
                               'Tekst artykułu',
                               'do PBL',
                               'hasła przedmiotowe']):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    df_json = pd.DataFrame(json_data)[['Link', 'Tekst artykułu']].astype(str)

    df_excel = pd.read_excel(excel_file_path)
    df_excel['original_order'] = df_excel.index

    merged_df = (
        pd.merge(df_json, df_excel, on=common_column, how="inner")
          .sort_values('original_order')
          .astype({'Tytuł artykułu': str, 'Tekst artykułu': str})
    )

    filt = merged_df[(merged_df['do PBL'] == True) &
                     (merged_df['hasła przedmiotowe'].notna())]

    if filt.empty:
        return pd.DataFrame(columns=selected_columns_list)

    last_idx = filt.index[-1]
    merged_df = merged_df.loc[:last_idx]
    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df[merged_df['do PBL'].isin([True, False])]

    return merged_df[selected_columns_list]

# --------------------------------------------------
# 2.  Wczytanie i połączenie wszystkich plików
# --------------------------------------------------
base_dir  = 'D:/Nowa_praca/dane_model_jezykowy/kopia_dla_UAM/'
json_dir  = os.path.join(base_dir, 'Json')

json_files   = {os.path.splitext(f)[0]: os.path.join(json_dir, f)
                for f in os.listdir(json_dir) if f.endswith('.json')}
excel_files  = {os.path.splitext(f)[0]: os.path.join(base_dir, f)
                for f in os.listdir(base_dir) if f.endswith('.xlsx')}

common_files = set(json_files) & set(excel_files)
merged_dfs   = []
for name in common_files:
    print(f"Łączenie: {name}")
    merged = load_and_merge_data(json_files[name], excel_files[name])
    if not merged.empty:
        merged_dfs.append(merged)

if not merged_dfs:
    raise RuntimeError("Brak danych do predykcji!")

df = pd.concat(merged_dfs, ignore_index=True)

# --------------------------------------------------
# 3.  Przygotowanie danych do modelu
# --------------------------------------------------
df = df.dropna(subset=['do PBL']).copy()
df['do PBL'] = (df['do PBL'].astype(str)
                .replace({'0.0': 'False', '1.0': 'True'}))

df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

sample_df = df.head(100).reset_index(drop=True)
MODEL_ID = "darekpe79/true-false-pbl-herbert"
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID,
                                                               use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

enc_path  = hf_hub_download(repo_id=MODEL_ID,
                            filename="label_encoder.joblib")
label_enc = joblib.load(enc_path)

# --------------------------------------------------
# 5.  Funkcja predykcji w partiach
# --------------------------------------------------
def predict(texts, model, tokenizer, batch_size=8):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok   = tokenizer(batch, return_tensors="pt",
                              truncation=True, padding=True, max_length=512)
            ids   = model(**tok).logits.argmax(-1).tolist()
            preds.extend(ids)
    return label_enc.inverse_transform(preds)

# --------------------------------------------------
# 6.  Dodaj kolumnę 'predykcja' i zapisz
# --------------------------------------------------
sample_df['predykcja'] = predict(sample_df['combined_text'].tolist(),
                                 model, tokenizer)

print(sample_df[['Tytuł artykułu', 'do PBL', 'predykcja']].head())

sample_df.to_excel("df_predykcje.xlsx", index=False)