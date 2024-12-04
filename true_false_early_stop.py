# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:36:55 2024

@author: dariu
"""

import pandas as pd
import json

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "do PBL"]):
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

    # Dodanie kolumny indeksowej do DataFrame'a z Excela
    df_excel['original_order'] = df_excel.index

    # Połączenie DataFrame'ów
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")

    # Sortowanie połączonego DataFrame według kolumny 'original_order'
    merged_df = merged_df.sort_values(by='original_order')

    # Konwersja wartości w kolumnach 'Tytuł artykułu' i 'Tekst artykułu' na stringi w połączonym DataFrame
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)

    # Znalezienie indeksu ostatniego 'True' w kolumnie 'do PBL'
    last_true_index = merged_df[merged_df['do PBL'] == True].index[-1]

    # Ograniczenie DataFrame do wierszy do ostatniego 'True' włącznie
    merged_df = merged_df.loc[:last_true_index]
    merged_df = merged_df.reset_index(drop=True)


    # Ograniczenie do wybranych kolumn
    selected_columns = merged_df[selected_columns_list]

    return selected_columns

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "do PBL", "hasła przedmiotowe"]):
    # Load data from JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)

    # Limit JSON DataFrame to 'Link' and 'Tekst artykułu' columns
    df_json = df_json[['Link', 'Tekst artykułu']]
    df_json['Tekst artykułu'] = df_json['Tekst artykułu'].astype(str)

    # Load data from Excel file
    df_excel = pd.read_excel(excel_file_path)
    df_excel['original_order'] = df_excel.index

    # Merge DataFrames
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")
    merged_df = merged_df.sort_values(by='original_order')
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)

    # Find index of last 'True' in 'do PBL' where 'hasła przedmiotowe' is filled
    filtered_df = merged_df[(merged_df['do PBL'] == True) & (merged_df['hasła przedmiotowe'].notna())]

    if not filtered_df.empty:
        last_true_filled_index = filtered_df.index[-1]
        # Limit DataFrame to rows up to the last 'True' inclusively where 'hasła przedmiotowe' is filled
        merged_df = merged_df.loc[:last_true_filled_index]
    else:
        # If the conditions are not met, return an empty DataFrame
        return pd.DataFrame(columns=selected_columns_list)

    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df[merged_df['do PBL'].isin([True, False])]
    # Limit to selected columns
    selected_columns = merged_df[selected_columns_list]


    return selected_columns

from transformers import BertTokenizer,EarlyStoppingCallback, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModel,HerbertTokenizerFast
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# Załaduj dane z wcześniej przygotowanego DataFrame (df)
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
    combined_df = pd.concat(merged_dfs, ignore_index=True)
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification, HerbertTokenizerFast, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModel
import joblib

# Ustawienie loggerów
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)
datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)

# Przygotowanie danych
# Zakładam, że masz DataFrame 'combined_df' z kolumną 'do PBL', która zawiera wartości True lub False

# Usuwanie wierszy z brakującymi wartościami w 'do PBL'
df = combined_df.dropna(subset=['do PBL']).copy()

# Upewnienie się, że wartości w 'do PBL' są typu string
df['do PBL'] = df['do PBL'].astype(str)

# Wyświetlenie unikalnych wartości w 'do PBL'
unique_values = df['do PBL'].unique()
print(f"Unikalne wartości w 'do PBL': {unique_values}")
count_0_0 = df[df['do PBL'] == '0.0'].shape[0]
print(f"Liczba wierszy z '0.0': {count_0_0}")

# Liczba wierszy z '1.0'
count_1_0 = df[df['do PBL'] == '1.0'].shape[0]
print(f"Liczba wierszy z '1.0': {count_1_0}")
df['do PBL'] = df['do PBL'].replace({'0.0': 'False', '1.0': 'True'})

# Sprawdzenie unikalnych wartości po mapowaniu
unique_values = df['do PBL'].unique()
print(f"Unikalne wartości w 'do PBL' po mapowaniu: {unique_values}")
count_0_0 = df[df['do PBL'] == 'False'].shape[0]
count_1_0 = df[df['do PBL'] == 'True'].shape[0]
# Zakodowanie etykiet
# Zakodowanie etykiet
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['do PBL'])
print(f"Mapowanie etykiet: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Połączenie tytułu i tekstu artykułu w jednym polu
df['combined_text'] = df['Tytuł artykułu'].astype(str) + " " + df['Tekst artykułu'].astype(str)

# Przygotowanie tokenizatora i modelu
tokenizer = HerbertTokenizerFast.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-base-cased",
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
)

# Funkcja do tokenizacji i kodowania danych
def tokenize_and_encode(examples):
    return tokenizer(
        examples['combined_text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

# Tworzenie datasetu
dataset = Dataset.from_pandas(df[['combined_text', 'label']])
dataset = dataset.map(tokenize_and_encode, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Podział na zbiór treningowy i walidacyjny
train_test_dataset = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    'train': train_test_dataset['train'],
    'eval': train_test_dataset['test']
})

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.02,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",  # Ustawiamy 'eval_accuracy' jako metrykę do monitorowania
    greater_is_better=True,                 # Ponieważ chcemy maksymalizować dokładność
    report_to=["none"]
)

# Definicja funkcji do obliczania metryk
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0)
    return {
        'accuracy': acc,    # Nazwa metryki będzie automatycznie prefiksowana jako 'eval_accuracy'
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Import EarlyStoppingCallback
from transformers import EarlyStoppingCallback

# Inicjalizacja trenera z funkcją wczesnego zatrzymania
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['eval'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Trening modelu
trainer.train()

# Ewaluacja modelu
results = trainer.evaluate()

# Wyświetlenie wyników
print(results)

# Zapisanie modelu i tokenizatora
model_path = "./model_NOWY_TRUEFALSE"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Zapisanie LabelEncoder
joblib.dump(label_encoder, os.path.join(model_path, 'label_encoder.joblib'))