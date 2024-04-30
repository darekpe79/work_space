# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:26:08 2024

@author: dariu
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

# -*- coding: utf-8 -*-
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
json_file_path16 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/biuroliterackie_biblioteka_2022-11-08.json'
excel_file_path16 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/biuroliterackie_2022-11-08.xlsx'
json_file_path17 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/chalwazwyciezonym_2023-02-01.json'
excel_file_path17 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/chalwazwyciezonym_2023-02-01.xlsx'
json_file_path18 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/cultureave_2023-02-20.json'
excel_file_path18 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/cultureave_2023-10-12.xlsx'
json_file_path19 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/eteatr_2023-10-12.json'
excel_file_path19 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/eteatr_2023-10-12.xlsx'
json_file_path20 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_org_pl_2023-02-06.json'
excel_file_path20 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_org_pl_2023-02-06.xlsx'
json_file_path21 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/gazetakulturalnazelow_2023-10-26.json'
excel_file_path21 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/gazetakulturalnazelow_2023-10-26.xlsx'
json_file_path22 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/hiperrealizm_2023-11-07.json'
excel_file_path22 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/hiperrealizm_2023-11-07.xlsx'
json_file_path23 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/kempinsky_2023-11-06.json'
excel_file_path23 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/kempinsky_2023-11-06.xlsx'
json_file_path24 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/kochampolskiekino_2023-02-02.json'
excel_file_path24 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/kochampolskiekino_2023-02-02.xlsx'
json_file_path25 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/martafox_2023-10-06.json'
excel_file_path25 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/martafox_2023-10-06.xlsx'
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
df16 = load_and_merge_data(json_file_path16, excel_file_path16)
df17 = load_and_merge_data(json_file_path17, excel_file_path17)
df18 = load_and_merge_data(json_file_path18, excel_file_path18)
df19= load_and_merge_data(json_file_path19, excel_file_path19)
df20= load_and_merge_data(json_file_path20, excel_file_path20)
df21= load_and_merge_data(json_file_path21, excel_file_path21)
df22= load_and_merge_data(json_file_path22, excel_file_path22)
df23= load_and_merge_data(json_file_path23, excel_file_path23)
df24= load_and_merge_data(json_file_path24, excel_file_path24)
df25= load_and_merge_data(json_file_path25, excel_file_path25)
# ... wczytanie kolejnych par plików

# Połączenie wszystkich DataFrame'ów
df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25], ignore_index=True)

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModel

# Załaduj dane z wcześniej przygotowanego DataFrame (df)

#df=df.head(300)
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)
# Usunięcie wierszy gdzie 'forma/gatunek' jest pusty
df = df.dropna(subset=['do PBL'])

# Połącz tytuł i tekst artykułu w jednym polu
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']
df['do PBL'] = df['do PBL'].astype(str)
# Kodowanie etykiet
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['do PBL'])



tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-base-cased",
    num_labels=len(label_encoder.classes_), #unikatowe etykiety
    problem_type="single_label_classification"
)


# Funkcja do tokenizacji i kodowania danych
def tokenize_and_encode(examples):
    return tokenizer(examples['combined_text'], padding='max_length', truncation=True, max_length=512)

# Mapowanie funkcji tokenizującej do datasetu
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_encode, batched=True)

# Podział na zbiór treningowy i walidacyjny
train_test_dataset = dataset.train_test_split(test_size=0.15)
dataset_dict = DatasetDict({
    'train': train_test_dataset['train'],
    'test': train_test_dataset['test']
})

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,  # Zmieniono wartość
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Zapewnia, że ewaluacja jest wykonywana co epokę
    save_strategy="no",
    #learning_rate=5e-5,  # Dodano szybkość uczenia
    #load_best_model_at_end=True,  # Wczytuje najlepszy model po zakończeniu treningu
    #metric_for_best_model="accuracy",  # Używa dokładności jako metryki do early stopping
    no_cuda=True
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

