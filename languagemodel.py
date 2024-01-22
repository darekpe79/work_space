# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:20:57 2023

@author: dariu
"""
import pandas as pd
import json

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "forma/gatunek"]):
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

    # Ograniczenie do wybranych kolumn i usunięcie wierszy z pustymi wartościami w 'forma/gatunek'
    if 'forma/gatunek' in merged_df.columns:
        selected_columns = merged_df[selected_columns_list]
        selected_columns = selected_columns.dropna(subset=['forma/gatunek'])
        return selected_columns
    else:
        return pd.DataFrame(columns=selected_columns_list)


# Ścieżki do plików
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'
json_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afront_2022-09-08.json'
excel_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afront_2022-09-08.xlsx'
json_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/artpapier_2022-10-05.json'
excel_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/artpapier_2022-10-05.xlsx'
json_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/audycjekulturalne_2022-10-11.json'
excel_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/audycjekulturalne_2022-10-11.xlsx'
json_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/bylam_widzialam_2023-02-21.json'
excel_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/bylam_widzialam_2023-02-21.xlsx'
json_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/czas_kultury_2023-03-24.json'
excel_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/czas_kultury_2023-03-24.xlsx'
json_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_dziennik_2023-10-23.json'
excel_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_dziennik_2023-10-23.xlsx'
json_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/intimathule_2022-09-09.json'
excel_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/intimathule_2022-09-09.xlsx'
json_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/jerzy_sosnowski_2022-09-09.json'
excel_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jerzy_sosnowski_2022-09-09.xlsx'
json_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/komnen_kastamonu_2022-09-12.json'
excel_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/komnen_kastamonu_2022-09-12.xlsx'
json_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/krzysztof_jaworski_2022-12-08.json'
excel_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/krzysztof_jaworski_2022-12-08.xlsx'
json_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pgajda_2022-09-13.json'
excel_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pgajda_2022-09-13.xlsx'
json_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/poeci_po_godzinach_2022-09-14.json'
excel_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/poeci_po_godzinach_2022-09-14.xlsx'
# ... więcej plików w razie potrzeby

# Użycie funkcji
df1 = load_and_merge_data(json_file_path, excel_file_path)
df2 = load_and_merge_data(json_file_path2, excel_file_path2)
df3 = load_and_merge_data(json_file_path3, excel_file_path3)
df4 = load_and_merge_data(json_file_path4, excel_file_path4)
df5 = load_and_merge_data(json_file_path5, excel_file_path5)
df6 = load_and_merge_data(json_file_path6, excel_file_path6)
df7 = load_and_merge_data(json_file_path7, excel_file_path7)
df8 = load_and_merge_data(json_file_path8, excel_file_path8)
df9 = load_and_merge_data(json_file_path9, excel_file_path9)
df10 = load_and_merge_data(json_file_path10, excel_file_path10)
df11 = load_and_merge_data(json_file_path11, excel_file_path11)
df12 = load_and_merge_data(json_file_path12, excel_file_path12)
df13 = load_and_merge_data(json_file_path13, excel_file_path13)
df14 = load_and_merge_data(json_file_path14, excel_file_path14)
df15 = load_and_merge_data(json_file_path15, excel_file_path15)

# ... wczytanie kolejnych par plików

# Połączenie wszystkich DataFrame'ów
combined_df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], ignore_index=True)
#%%
# import pandas as pd
# import json

# # Ścieżki do plików
# json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
# excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
# json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
# excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
# json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
# excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'

# # Wczytanie danych z pliku JSON
# with open(json_file_path, 'r', encoding='utf-8') as file:
#     json_data = json.load(file)
# df_json = pd.DataFrame(json_data)
# df_json=df_json[['Link', 'Tekst artykułu']]
# # Wczytanie danych z pliku Excel
# df_excel = pd.read_excel(excel_file_path)


# # Połączenie danych po tytule artykułu
# merged_df = pd.merge(df_json, df_excel, on="Link", how="inner")

# # Wybór kolumn, w tym pełnych tekstów
# selected_columns = merged_df[['Tytuł artykułu', 'Tekst artykułu', "forma/gatunek"]]
# selected_columns=selected_columns.dropna(subset=['forma/gatunek'])
# with open(json_file_path2, 'r', encoding='utf-8') as file2:
#     json_data2 = json.load(file2)
# df_json2 = pd.DataFrame(json_data2)
# df_json2=df_json2[['Link', 'Tekst artykułu']]
# # Wczytanie danych z pliku Excel
# df_excel2 = pd.read_excel(excel_file_path2)
# merged_df2 = pd.merge(df_json2, df_excel2, on="Link", how="inner")
# selected_columns2 = merged_df2[['Tytuł artykułu', 'Tekst artykułu', "forma/gatunek"]]
# selected_columns2=selected_columns2.dropna(subset=['forma/gatunek'])
# with open(json_file_path3, 'r', encoding='utf-8') as file3:
#     json_data3 = json.load(file3)
# df_json3 = pd.DataFrame(json_data3)
# df_json3=df_json3[['Link', 'Tekst artykułu']]
# # Wczytanie danych z pliku Excel
# df_excel3 = pd.read_excel(excel_file_path3)
# merged_df3 = pd.merge(df_json3, df_excel3, on="Link", how="inner")
# selected_columns3 = merged_df3[['Tytuł artykułu', 'Tekst artykułu', "forma/gatunek"]]
# selected_columns3=selected_columns3.dropna(subset=['forma/gatunek'])
# combined_df = pd.concat([selected_columns, selected_columns2,selected_columns3], ignore_index=True)
# # Eksportowanie do pliku JSON
# output_json_path = 'merged_data.json'  # Ścieżka do zapisu pliku JSON
# selected_columns.to_json(output_json_path, orient='records', force_ascii=False)
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModel

# Załaduj dane z wcześniej przygotowanego DataFrame (df)
df = combined_df
#df=df.head(300)
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)
# Usunięcie wierszy gdzie 'forma/gatunek' jest pusty
df = df.dropna(subset=['forma/gatunek'])

# Połącz tytuł i tekst artykułu w jednym polu
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

# Kodowanie etykiet
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['forma/gatunek'])

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
train_test_dataset = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    'train': train_test_dataset['train'],
    'test': train_test_dataset['test']
})

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,              # liczba epok
    per_device_train_batch_size=4,   # rozmiar batcha
    per_device_eval_batch_size=4,
    warmup_steps=500,                # kroki rozgrzewki
    weight_decay=0.01,               # waga decay
    logging_dir='./logs',
    evaluation_strategy="epoch",
    no_cuda=True  # Używanie CPU
)

# Inicjalizacja trenera
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_dict['train'],
#     eval_dataset=dataset_dict['test']
# )
from sklearn.metrics import accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    compute_metrics=compute_metrics
)

# Trening modelu
trainer.train()

# Ewaluacja modelu
results = trainer.evaluate()

# Wyniki
print(results)
model_path = "model_4epoch"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
import joblib

# Zapisanie LabelEncoder
joblib.dump(label_encoder, 'label_encoder.joblib')


#ładowanie modelu
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
import joblib



# W późniejszym czasie, aby wczytać LabelEncoder:
label_encoder = joblib.load('label_encoder.joblib')


# from transformers import BertTokenizer, BertForSequenceClassification

# model_path = "C:/Users/dariu/model/"
# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(model_path)
tform = df.iloc[30]['forma/gatunek']
tyt = df.iloc[30]['Tytuł artykułu']
text = df.iloc[30]['Tekst artykułu']
text=tyt+' '+text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Przewidywanie za pomocą modelu
model.eval()  # Ustawienie modelu w tryb ewaluacji
with torch.no_grad():  # Wyłączenie obliczeń gradientów
    outputs = model(**inputs)

# Pobranie wyników
predictions = torch.softmax(outputs.logits, dim=1)
predicted_index = predictions.argmax().item()
predicted_label = label_encoder.inverse_transform([predicted_index])[0]

print(f"Przewidziana kategoria: {predicted_label}")
#Przewidywanie dla większej ilosci kategorii
results = []

for i in range(100,200):
    title = df.iloc[i]['Tytuł artykułu']
    text = df.iloc[i]['Tekst artykułu']
    full_text = title + " " + text
    inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    predicted_index = predictions.argmax().item()
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    results.append({'Title': title, 'Text': text, 'Original': df.iloc[i]['forma/gatunek'], 'Predicted': predicted_label})

results_df = pd.DataFrame(results)

#%%przykład użycia allegro

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModel.from_pretrained("allegro/herbert-base-cased")

output = model(
    **tokenizer.batch_encode_plus(
        [
            (
                "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
                "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy."
            )
        ],
    padding='longest',
    add_special_tokens=True,
    return_tensors='pt'
    )
)
#%%Hasla przedmiotowe
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import logging
import pandas as pd
import json

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

    # Ograniczenie do wybranych kolumn i usunięcie wierszy z pustymi wartościami w 'hasła przedmiotowe'
    if 'hasła przedmiotowe' in merged_df.columns:
        selected_columns = merged_df[selected_columns_list]
        selected_columns = selected_columns.dropna(subset=['hasła przedmiotowe'])
        return selected_columns
    else:
        return pd.DataFrame(columns=selected_columns_list)


# Ścieżki do plików
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'
json_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afront_2022-09-08.json'
excel_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afront_2022-09-08.xlsx'
json_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/artpapier_2022-10-05.json'
excel_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/artpapier_2022-10-05.xlsx'
json_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/audycjekulturalne_2022-10-11.json'
excel_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/audycjekulturalne_2022-10-11.xlsx'
json_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/bylam_widzialam_2023-02-21.json'
excel_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/bylam_widzialam_2023-02-21.xlsx'
json_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/czas_kultury_2023-03-24.json'
excel_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/czas_kultury_2023-03-24.xlsx'
json_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_dziennik_2023-10-23.json'
excel_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_dziennik_2023-10-23.xlsx'
json_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/intimathule_2022-09-09.json'
excel_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/intimathule_2022-09-09.xlsx'
json_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/jerzy_sosnowski_2022-09-09.json'
excel_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jerzy_sosnowski_2022-09-09.xlsx'
json_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/komnen_kastamonu_2022-09-12.json'
excel_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/komnen_kastamonu_2022-09-12.xlsx'
json_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/krzysztof_jaworski_2022-12-08.json'
excel_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/krzysztof_jaworski_2022-12-08.xlsx'
json_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pgajda_2022-09-13.json'
excel_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pgajda_2022-09-13.xlsx'
json_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/poeci_po_godzinach_2022-09-14.json'
excel_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/poeci_po_godzinach_2022-09-14.xlsx'
# ... więcej plików w razie potrzeby

# Użycie funkcji
df1 = load_and_merge_data(json_file_path, excel_file_path)
df2 = load_and_merge_data(json_file_path2, excel_file_path2)
df3 = load_and_merge_data(json_file_path3, excel_file_path3)
df4 = load_and_merge_data(json_file_path4, excel_file_path4)
df5 = load_and_merge_data(json_file_path5, excel_file_path5)
df6 = load_and_merge_data(json_file_path6, excel_file_path6)
df7 = load_and_merge_data(json_file_path7, excel_file_path7)
df8 = load_and_merge_data(json_file_path8, excel_file_path8)
df9 = load_and_merge_data(json_file_path9, excel_file_path9)
df10 = load_and_merge_data(json_file_path10, excel_file_path10)
df11 = load_and_merge_data(json_file_path11, excel_file_path11)
df12 = load_and_merge_data(json_file_path12, excel_file_path12)
df13 = load_and_merge_data(json_file_path13, excel_file_path13)
df14 = load_and_merge_data(json_file_path14, excel_file_path14)
df15 = load_and_merge_data(json_file_path15, excel_file_path15)

# ... wczytanie kolejnych par plików

# Połączenie wszystkich DataFrame'ów
df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], ignore_index=True)

logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)
df_excel = pd.read_excel('C:/Users/dariu/Downloads/Mapowanie działów.xlsx')
df_excel['połączony dział'] = df_excel['nr działu'].astype(str) + " " + df_excel['nazwa działu']

mapowanie = pd.Series(df_excel['string uproszczony'].values, index=df_excel['połączony dział']).to_dict()
def mapuj_hasla(lista_hasel):
    return [mapowanie.get(haslo.strip()) if haslo is not None else None for haslo in lista_hasel]

# Rozdzielenie haseł, mapowanie i ponowne połączenie
df['hasła przedmiotowe'] = df['hasła przedmiotowe'].str.split('; ')
df['rozwiniete_haslo'] = df['hasła przedmiotowe'].apply(mapuj_hasla)

# Usuwanie None z list w kolumnie 'rozwiniete_haslo'
df['rozwiniete_haslo'] = df['rozwiniete_haslo'].apply(lambda x: [i for i in x if i is not None])

# Usunięcie wierszy, gdzie 'rozwiniete_haslo' jest puste
df = df[df['rozwiniete_haslo'].map(len) > 0]

# Połącz tytuł i tekst artykułu w jednym polu
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

# Rozdzielenie kategorii
#df['hasła przedmiotowe'] = df['hasła przedmiotowe'].apply(lambda x: x.split('; '))
#usunięcie pustych kategorii
#df['hasła przedmiotowe'] = df['hasła przedmiotowe'].apply(lambda x: [i.strip() for i in x if i.strip()])

# Użycie MultiLabelBinarizer do przekształcenia kategorii w format binarny
mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('rozwiniete_haslo')),
                          columns=mlb.classes_,
                          index=df.index))
print (df)
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-base-cased",
    num_labels=len(mlb.classes_),  # liczba etykiet wynika z MultiLabelBinarizer
    problem_type="multi_label_classification"  # zmiana na klasyfikację wieloetykietową
)


# Funkcja do tokenizacji i kodowania danych
def tokenize_and_encode(examples):
    return tokenizer(examples['combined_text'], padding='max_length', truncation=True, max_length=512)

# Mapowanie funkcji tokenizującej do datasetu
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_encode, batched=True)

# Podział na zbiór treningowy i walidacyjny
train_test_dataset = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    'train': train_test_dataset['train'],
    'test': train_test_dataset['test']
})

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    no_cuda=True
)

# Funkcja do obliczania metryk dla klasyfikacji wieloetykietowej

def compute_metrics(pred):
    # prognozy są w formacie logitów; konwersja do formatu prawdopodobieństwa
    preds_probs = torch.sigmoid(torch.from_numpy(pred.predictions)).numpy()
    preds = np.where(preds_probs > 0.5, 1, 0)  # prognozy binarne na podstawie progu 0.5

    labels = pred.label_ids

    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')

    return {'precision': precision, 'recall': recall, 'f1': f1}

# Inicjalizacja trenera z uwzględnieniem klasyfikacji wieloetykietowej
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    compute_metrics=compute_metrics
)

# Trening modelu
trainer.train()

# Ewaluacja modelu
results = trainer.evaluate()

# Wyniki
print(results)




#%%wieloetykietowosc
import json
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, ClassLabel
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset as TorchDataset

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', 
                        base_columns_list=['Tytuł artykułu', 'Tekst artykułu'],
                        additional_columns_list=['byt 1', 'byt 2', 'byt 3']):
    
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
    merged_df['combined_text'] = merged_df['Tytuł artykułu'] + " " + merged_df['Tekst artykułu']
    # Wybór kolumn
    selected_columns_list = base_columns_list + additional_columns_list + ['combined_text']
    selected_columns = merged_df[selected_columns_list]

    # Usunięcie wierszy, gdzie wszystkie kolumny 'byt1', 'byt2', 'byt3' są puste
    selected_columns = selected_columns.dropna(subset=additional_columns_list, how='all')

    return selected_columns

json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'
json_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afront_2022-09-08.json'
excel_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afront_2022-09-08.xlsx'
json_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/artpapier_2022-10-05.json'
excel_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/artpapier_2022-10-05.xlsx'
json_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/audycjekulturalne_2022-10-11.json'
excel_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/audycjekulturalne_2022-10-11.xlsx'
json_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/bylam_widzialam_2023-02-21.json'
excel_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/bylam_widzialam_2023-02-21.xlsx'
json_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/czas_kultury_2023-03-24.json'
excel_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/czas_kultury_2023-03-24.xlsx'
json_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_dziennik_2023-10-23.json'
excel_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_dziennik_2023-10-23.xlsx'
json_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/intimathule_2022-09-09.json'
excel_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/intimathule_2022-09-09.xlsx'
json_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/jerzy_sosnowski_2022-09-09.json'
excel_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jerzy_sosnowski_2022-09-09.xlsx'
json_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/komnen_kastamonu_2022-09-12.json'
excel_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/komnen_kastamonu_2022-09-12.xlsx'
json_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/krzysztof_jaworski_2022-12-08.json'
excel_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/krzysztof_jaworski_2022-12-08.xlsx'
json_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pgajda_2022-09-13.json'
excel_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pgajda_2022-09-13.xlsx'
json_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/poeci_po_godzinach_2022-09-14.json'
excel_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/poeci_po_godzinach_2022-09-14.xlsx'
# ... więcej plików w razie potrzeby

# Użycie funkcji
df1 = load_and_merge_data(json_file_path, excel_file_path)
df2 = load_and_merge_data(json_file_path2, excel_file_path2)
df3 = load_and_merge_data(json_file_path3, excel_file_path3)
df4 = load_and_merge_data(json_file_path4, excel_file_path4)
df5 = load_and_merge_data(json_file_path5, excel_file_path5)
df6 = load_and_merge_data(json_file_path6, excel_file_path6)
df7 = load_and_merge_data(json_file_path7, excel_file_path7)
df8 = load_and_merge_data(json_file_path8, excel_file_path8)
df9 = load_and_merge_data(json_file_path9, excel_file_path9)
df10 = load_and_merge_data(json_file_path10, excel_file_path10)
df11 = load_and_merge_data(json_file_path11, excel_file_path11)
df12 = load_and_merge_data(json_file_path12, excel_file_path12)
df13 = load_and_merge_data(json_file_path13, excel_file_path13)
df14 = load_and_merge_data(json_file_path14, excel_file_path14)
df15 = load_and_merge_data(json_file_path15, excel_file_path15)

# ... wczytanie kolejnych par plików

# Połączenie wszystkich DataFrame'ów
df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], ignore_index=True)






class NERDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)


    def __getitem__(self, item):
        text = self.texts[item]
        labels = self.labels[item]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Dopasowanie długości etykiet do max_len, zaczynając od początku listy labels
        padded_labels = [label2id.get(label, 0) for label in labels]  # Używaj label2id
        padded_labels += [label2id["O"]] * (self.max_len - len(padded_labels))
        padded_labels = torch.tensor(padded_labels[:self.max_len], dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': padded_labels
        }
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


texts = df['combined_text'].tolist()


# Zaktualizuj tagi na podstawie swoich danych


# Mapowanie etykiet na identyfikatory
label2id = {
    "O": 0,  # Dla słów, które nie są nazwanymi jednostkami
    "czasopismo": 1,
    "film": 2,
    "instytucja": 3,
    "książka": 4,
    "miejscowość": 5,
    "osoba": 6,
    "spektakl": 7,
    "wydarzenie": 8
}

# Przykład konwersji kolumn `byt1`, `byt2`, `byt3` na listę etykiet
tags = [[label2id.get(val, 0) for val in row] for row in df[['byt 1', 'byt 2', 'byt 3']].values]

id2label = {v: k for k, v in label2id.items()}
#model = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")
# Przygotuj tokenizator i model
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("allegro/herbert-base-cased", num_labels=len(label2id))
model.config.id2label = id2label
model.config.label2id = label2id

# Przygotuj zestaw danych
dataset = NERDataset(texts=texts, labels=tags, tokenizer=tokenizer, max_len=128)

# Konwersja do Hugging Face Dataset i podział na zbiory
hf_dataset = Dataset.from_dict({"text": texts, "labels": tags})
train_test_dataset = hf_dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    'train': train_test_dataset['train'],
    'test': train_test_dataset['test']
})
sample = dataset[0]
print("Testowe uruchomienie modelu na pojedynczej próbce danych:")
# with torch.no_grad():
#     outputs = model(**{k: v.unsqueeze(0) for k, v in sample.items()})
#     print(outputs)
# Parametry treningu
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)
print("Przykładowe dane treningowe:", dataset_dict['train'][0])
# Inicjalizacja trenera
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    compute_metrics=compute_metrics
)

# Rozpoczęcie treningu
trainer.train()
#%% Drugi kodzik wiele etykiet
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch

class NERDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        labels = self.labels[item]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        padded_labels = [label2id.get(label, 0) for label in labels]
        padded_labels += [label2id["O"]] * (self.max_len - len(padded_labels))
        padded_labels = torch.tensor(padded_labels[:self.max_len], dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': padded_labels
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# Twoje dane
texts = df['combined_text'].tolist()
tags = [[label2id.get(val, 0) for val in row] for row in df[['byt 1', 'byt 2', 'byt 3']].values]

# Używane identyfikatory etykiet
label2id = {
    "O": 0,  # Dla słów, które nie są nazwanymi jednostkami
    "czasopismo": 1,
    "film": 2,
    "instytucja": 3,
    "książka": 4,
    "miejscowość": 5,
    "osoba": 6,
    "spektakl": 7,
    "wydarzenie": 8
}

# Inicjalizacja tokenizatora i modelu
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("allegro/herbert-base-cased", num_labels=len(label2id))
model.config.id2label = {v: k for k, v in label2id.items()}
model.config.label2id = label2id

# Podział na zbiory treningowe i testowe
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, tags, test_size=0.2)

# Utworzenie zbiorów treningowego i testowego
train_dataset = NERDataset(train_texts, train_labels, tokenizer, max_len=128)
test_dataset = NERDataset(test_texts, test_labels, tokenizer, max_len=128)

# Tworzenie DataLoaderów
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Parametry treningu
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Inicjalizacja trenera
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # bezpośrednio używaj train_dataset
    eval_dataset=test_dataset,    # bezpośrednio używaj test_dataset
    compute_metrics=compute_metrics
)

# Rozpoczęcie treningu
trainer.train()

# Ewaluacja po treningu
results = trainer.evaluate()
print(results)
# Zapisz model i tokenizer
model.save_pretrained("byty_1epoch")

# Zapisz tokenizator
tokenizer.save_pretrained("byty_1epoch")

# Tekst, dla którego chcesz przeprowadzić rozpoznawanie nazwanych jednostek
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Załaduj model
model = AutoModelForTokenClassification.from_pretrained("byty_1epoch")

# Załaduj tokenizator
tokenizer = AutoTokenizer.from_pretrained("byty_1epoch")
article_text = "Sherlock Holmes i doktor Watson w nowej kampanii reklamowej OLX Klasyczni bohaterowie stworzeni przez Arthura Conana Doyle’a pojawili się w nowej kampanii reklamowej serwisu ogłoszeniowego OLX. Sherlock Holmes i doktor Watson mają w zrealizowanych filmikach humorystycznych ostrzegać, przypominać i tłumaczyć, jak nie dać się oszukać w sieci. Seria krótkich filmów w zabawny sposób gra z wizerunkiem postaci stworzonych przez Arthura Conana Doyle’a. Jacek Borusiński z Mumio wciela się w cokolwiek średnio błyskotliwego Sherlocka, natomiast Sebastian Stankiewicz partneruje mu w roli Watsona i w tej wersji słynnego duetu to raczej on nosi głowę na karku. Ich detektywistyczne biuro odwiedzają współcześni bohaterowie padający ofiarą internetowych łotrów. „Bezpieczeństwo naszych użytkowników jest dla nas niezwykle ważne, dlatego regularnie wracamy do tego tematu w kolejnych kampaniach wideo. To już nasza piąta aktywacja poświęcona tej tematyce, a jednocześnie swego rodzaju powrót do korzeni, ponieważ OLX.pl, na bardzo wczesnym etapie swojego istnienia, dostępny był pod adresem Szerlok.pl” – mówi Michał Wiśniewski z OLX. Za kreację kampanii odpowiada agencja PZL. „Kradzieże tożsamości, wyłudzanie danych czy phishing to obrzydliwe przestępstwa. Jesteśmy w PZL przekonani, że poruszanie tych tematów śmiertelnie poważnie mogłoby podziałać na odbiorców wręcz odpychająco. Jak więc podać treść ostrzeżeń bez oceniania i dydaktyzmu? Zdecydowaliśmy się na komediową formę, mając nadzieję, że tak podane treści dotrą do odbiorców i zapobiegną w jakiejś mierze szerzeniu się e-zła” – mówi Sławomir Szczęśniak z PZL, współautor koncepcji kreatywnej. Produkcję klipów powierzono firmie Lucky Luciano Pictures. W ramach kampanii powstały cztery spoty, każdy poświęcony innemu oszustwu w sieci. Dwa z nich zostały już udostępnione w sieci. Kampania jest dostępna w mediach społecznościowych marki OLX. "

# Tokenizacja i przekształcenie tekstu
encoding = tokenizer(article_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Używanie modelu do predykcji na nowym tekście
model.eval()
with torch.no_grad():
    outputs = model(**encoding)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Przetwarzanie predykcji
predicted_labels = [model.config.id2label[prediction.item()] for prediction in predictions[0]]
print(predicted_labels)


# Funkcja do przypisywania słowom etykiet i ekstrakcji fragmentów tekstu
def extract_entities(text, model, tokenizer):
    # Tokenizacja tekstu
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    # Wyciągnięcie przewidywanych etykiet
    predictions = torch.argmax(outputs.logits, dim=2)
    token_ids = inputs["input_ids"].squeeze().tolist()  # Lista ID tokenów

    # Konwersja ID tokenów na słowa (tokeny)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Mapowanie etykiet do tokenów i ich grupowanie
    entities = []
    current_entity = {"word": "", "label": ""}
    for token, label_idx in zip(tokens, predictions.squeeze().tolist()):
        # Pomiń specjalne tokeny
        if token.startswith("##") or token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        label = model.config.id2label[label_idx]
        if label != "O":  # Oznacza to, że token jest częścią nazwanej jednostki
            if label == current_entity["label"]:
                # Kontynuacja obecnej jednostki
                current_entity["word"] += " " + token
            else:
                # Nowa jednostka, zapisz poprzednią jeśli istnieje
                if current_entity["word"]:
                    entities.append(current_entity)
                current_entity = {"word": token, "label": label}
        else:
            # Jeśli poprzednio była jednostka, zapisz ją
            if current_entity["word"]:
                entities.append(current_entity)
                current_entity = {"word": "", "label": ""}
    
    # Dodaj ostatnią jednostkę, jeśli jest
    if current_entity["word"]:
        entities.append(current_entity)

    return entities

# Przykładowe wywołanie funkcji
text = "Jan Kowalski mieszka w Warszawie. Czytał Władcę Pierścieni."
entities = extract_entities(text, model, tokenizer)

# Wyświetl wyniki
for entity in entities:
    print(f"{entity['word']} -> {entity['label']}")

#%% TensorFlow

import os
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import spacy

# Załaduj polski model językowy
nlp = spacy.load('pl_core_news_sm')

# Pobierz polskie stop words
stop_words = nlp.Defaults.stop_words

def clean_text(text):
    # Usunięcie znaków specjalnych
    text = re.sub(r'\W', ' ', text)
    # Usunięcie wszystkich pojedynczych znaków
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Zamiana wielokrotnych spacji na pojedynczą spację
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Konwersja na małe litery
    text = text.lower()
    # Usunięcie stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
df = combined_df
# Przygotowanie danych
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']
df['combined_text']=df['combined_text'].apply(clean_text)

texts = df['combined_text'].values
labels = pd.get_dummies(df['forma/gatunek']).values

# Tokenizacja i sekwencje
tokenizer = Tokenizer(num_words=7000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

# Podział na zestawy treningowe i testowe
split = int(len(X) * 0.9)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = labels[:split], labels[split:]

# Budowa modelu
model = Sequential()
model.add(Embedding(input_dim=7000, output_dim=64, input_length=100))
model.add(LSTM(64))
model.add(Dense(labels.shape[1], activation='softmax'))

# Kompilacja i trenowanie
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=7, batch_size=32, validation_data=(X_test, Y_test))
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Dokładność testowa:', test_acc)
import numpy as np
# Przewidywania i rzeczywiste etykiety
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(Y_test, axis=1)

# Dodaj przewidywania i etykiety do DataFrame
df_test = df[split:].reset_index(drop=True)
df_test['Rzeczywista Etykieta'] = true_labels
df_test['Przewidywana Etykieta'] = predicted_labels
genre_mapping = {index: genre for index, genre in enumerate(df['forma/gatunek'].unique())}

# Zamiana cyfrowych etykiet na nazwy gatunków
df_test['Rzeczywista Etykieta'] = df_test['Rzeczywista Etykieta'].map(genre_mapping)
df_test['Przewidywana Etykieta'] = df_test['Przewidywana Etykieta'].map(genre_mapping)
# Teraz df_test zawiera teksty, rzeczywiste etykiety i przewidywane etykiety
combined_df.to_excel("combined_data.xlsx", index=False)
