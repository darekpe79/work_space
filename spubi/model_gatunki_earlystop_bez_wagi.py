# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:47:29 2024

@author: dariu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:00:12 2024

@author: dariu
"""

import pandas as pd
import json

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', 'forma/gatunek', 'do PBL']):
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

    # Filtracja tylko wierszy, gdzie 'do PBL' jest True
    if 'do PBL' in merged_df.columns:
        filtered_df = merged_df[merged_df['do PBL'] == True]

        # Ograniczenie do wybranych kolumn i usunięcie wierszy z pustymi wartościami w 'forma/gatunek'
        if 'forma/gatunek' in filtered_df.columns:
            selected_columns = filtered_df[selected_columns_list]
            selected_columns = selected_columns.dropna(subset=['forma/gatunek'])
            return selected_columns
        else:
            return pd.DataFrame(columns=selected_columns_list)
    else:
        return pd.DataFrame(columns=selected_columns_list)


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
    df = pd.concat(merged_dfs, ignore_index=True)
#df=df.head(300)
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)
# Usunięcie wierszy gdzie 'forma/gatunek' jest pusty
df = df.dropna(subset=['forma/gatunek'])

df = df.assign(forma_gatunek_split=df['forma/gatunek'].str.split(r'[;,]')).explode('forma_gatunek_split')


# Usuń zbędne białe znaki z hasła przedmiotowego
df['forma_gatunek_split'] = df['forma_gatunek_split'].str.strip()


# Opcjonalnie: Usuń wiersze z pustymi hasłami po stripowaniu
df = df[df['forma_gatunek_split'] != '']
df = df.reset_index(drop=True)

df = df.dropna(subset=['forma/gatunek'])
min_samples = 100
class_counts = df['forma_gatunek_split'].value_counts()
classes_to_keep = class_counts[class_counts >= min_samples].index.tolist()

# Odfiltrowujemy wiersze z klasami poniżej progu
df = df[df['forma_gatunek_split'].isin(classes_to_keep)]
# Zresetuj indeksy DataFrame

# Połącz tytuł i tekst artykułu w jednym polu
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

# Kodowanie etykiet
label_encoder = LabelEncoder()

df['labels'] = label_encoder.fit_transform(df['forma_gatunek_split'])
num_classes = len(label_encoder.classes_)

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")
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

# Mapowanie funkcji tokenizującej do datasetu
# Mapowanie funkcji tokenizującej do datasetu
dataset = Dataset.from_pandas(df[['combined_text', 'labels']])
dataset = dataset.map(tokenize_and_encode, batched=True)

# Usuń niepotrzebne kolumny, aby uniknąć ostrzeżeń
dataset = dataset.remove_columns(["combined_text", "__index_level_0__"])

# Ustawienie kolumny 'labels' jako 'label' i sformatowanie danych
dataset = dataset.rename_column("labels", "label")
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
    num_train_epochs=10,               # Ustawienie większej liczby epok
    per_device_train_batch_size=8,     # Zwiększenie batch size, jeśli to możliwe
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,       # Załadowanie najlepszego modelu na końcu
    metric_for_best_model="accuracy",  # Metryka używana do wyboru najlepszego modelu
    greater_is_better=True,
    save_total_limit=2,                # Maksymalna liczba zapisanych modeli
    no_cuda=True                       # Ustaw na False, jeśli masz dostęp do GPU
)

# Definicja funkcji do obliczania metryk
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Inicjalizacja trenera bez wag klas
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['eval'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Wczesne zatrzymanie po 2 epokach bez poprawy
)

# Trening modelu
trainer.train()

# Ewaluacja modelu
results = trainer.evaluate()

# Wyniki
print(results)

# Zapisanie modelu i tokenizer'a
model_path = "model_best"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Zapisanie LabelEncoder
import joblib
joblib.dump(label_encoder, 'label_encoder.joblib')