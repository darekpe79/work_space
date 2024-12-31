# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:03:24 2024

@author: darek
"""

import pandas as pd
import json


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

import logging
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoModelForSequenceClassification, HerbertTokenizerFast
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import joblib
import os
import numpy as np

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

# Mapowanie '0.0' na 'False' i '1.0' na 'True'
df['do PBL'] = df['do PBL'].replace({'0.0': 'False', '1.0': 'True'})

# Wyświetlenie unikalnych wartości w 'do PBL'
unique_values = df['do PBL'].unique()
print(f"Unikalne wartości w 'do PBL' po mapowaniu: {unique_values}")

# Sprawdzenie liczby wystąpień każdej klasy
value_counts = df['do PBL'].value_counts()
print("Liczba wystąpień przed undersamplingiem:")
print(value_counts)

# Ustalanie docelowego stosunku klas (np. 1.5 razy więcej 'False' niż 'True')
desired_ratio = 1.3  # Możesz dostosować tę wartość według potrzeb

# Obliczenie liczby przykładów do zachowania z klasy 'False'
count_true = value_counts['True']
count_false = value_counts['False']
keep_false_count = int(count_true * desired_ratio)

# Upewnienie się, że nie przekraczamy dostępnej liczby przykładów klasy 'False'
keep_false_count = min(keep_false_count, count_false)

# Losowe próbkowanie przykładów z klasy 'False'
df_false = df[df['do PBL'] == 'False'].sample(n=keep_false_count, random_state=42)
df_true = df[df['do PBL'] == 'True']

# Połączenie danych
df_adjusted = pd.concat([df_false, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)

# Ponowne sprawdzenie liczby wystąpień po dostosowaniu
value_counts_adjusted = df_adjusted['do PBL'].value_counts()
print("Liczba wystąpień po dostosowaniu:")
print(value_counts_adjusted)


label_encoder = LabelEncoder()
df_adjusted['label'] = label_encoder.fit_transform(df_adjusted['do PBL'])
print(f"Mapowanie etykiet: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print(f"Unique labels: {df_adjusted['label'].unique()}")

# Połączenie tytułu i tekstu artykułu w jednym polu
df_adjusted['combined_text'] = df_adjusted['Tytuł artykułu'].astype(str) + " " + df_adjusted['Tekst artykułu'].astype(str)

# Diagnostyka danych
print(df_adjusted[df_adjusted['label'] == 0].sample(5)['combined_text'])
print(df_adjusted[df_adjusted['label'] == 1].sample(5)['combined_text'])
assert not df_adjusted['combined_text'].isna().any(), "Puste wartości w 'combined_text'"

# Przygotowanie tokenizatora i modelu
tokenizer = HerbertTokenizerFast.from_pretrained("allegro/herbert-large-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-large-cased",
    num_labels=len(label_encoder.classes_),
    problem_type="single_label_classification"
)

# Funkcja do tokenizacji i kodowania danych
def tokenize_and_encode(examples):
    return tokenizer(
        examples['combined_text'],
        padding='max_length',
        truncation=True,
        max_length=514
    )

# Tworzenie datasetu
dataset = Dataset.from_pandas(df_adjusted[['combined_text', 'label']])
dataset = dataset.map(tokenize_and_encode, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Podział na zbiór treningowy i walidacyjny
train_test_dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset_dict = DatasetDict({
    'train': train_test_dataset['train'],
    'eval': train_test_dataset['test']
})

# Diagnostyka podziału danych
print("Równowaga klas w zbiorze treningowym:")
train_labels = dataset_dict['train']['label']
print(np.unique(train_labels, return_counts=True))

print("Równowaga klas w zbiorze walidacyjnym:")
eval_labels = dataset_dict['eval']['label']
print(np.unique(eval_labels, return_counts=True))

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True,
    warmup_steps=500,
    weight_decay=0.02,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,  # Zmniejszona wartość learning rate
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    report_to=["none"]
)


# Definicja funkcji do obliczania metryk
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Diagnostyka
    print(f"Rzeczywiste etykiety: {np.unique(labels, return_counts=True)}")
    print(f"Przewidywane etykiety: {np.unique(preds, return_counts=True)}")

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Funkcja do wizualizacji macierzy pomyłek
def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['False', 'True'])
    disp.plot(cmap='Blues')

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

resume_training = False  # Zmień na False, aby zacząć od nowa
checkpoint_path = "D:/Nowa_praca/dane_model_jezykowy/TRUE FALSE CHECKPOint/results/checkpoint-4104"

if resume_training:
    print(f"Kontynuowanie treningu od checkpointu: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("Rozpoczęcie nowego treningu.")
    trainer.train()

# Ewaluacja modelu
results = trainer.evaluate()
print("Wyniki ewaluacji:", results)

# Macierz pomyłek
predictions = trainer.predict(dataset_dict['eval'])
plot_confusion_matrix(predictions.label_ids, predictions.predictions.argmax(-1))

# Zapisanie modelu i tokenizatora
model_path = "./model_NOWY_TRUEFALSE"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Zapisanie LabelEncoder
joblib.dump(label_encoder, os.path.join(model_path, 'label_encoder.joblib'))

