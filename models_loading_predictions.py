# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:13:19 2024

@author: dariu
"""

import pandas as pd
import json
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import textwrap
import networkx as nx
from spacy.lang.pl import Polish
import spacy
from collections import defaultdict
from fuzzywuzzy import fuzz
from collections import defaultdict
from definicje import *


def load_and_merge_data(json_file_path, excel_file_path, common_column='Link'):
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

    # Filtracja DataFrame do wierszy, w których 'hasła przedmiotowe' oraz 'forma/gatunek' są puste
    filtered_df = merged_df[merged_df['hasła przedmiotowe'].isna() & merged_df['forma/gatunek'].isna()]

    # Ograniczenie do kolumn 'Tytuł artykułu' i 'Tekst artykułu'
    final_df = filtered_df[['Tytuł artykułu', 'Tekst artykułu']]

    return final_df


def load_and_merge_data(json_file_path, excel_file_path, common_column='Link'):
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

    # Znalezienie indeksu ostatniego True w kolumnie 'do PBL'
    last_true_index = merged_df[merged_df['do PBL'] == True].index[-1]

    # Zwrócenie wszystkich wierszy począwszy od wiersza następującego po ostatnim True
    final_df = merged_df.loc[last_true_index+1:, ['Tytuł artykułu', 'Tekst artykułu']]

    return final_df

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link'):
    # Wczytanie danych z pliku JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)

    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']].astype(str)

    # Wczytanie danych z pliku Excel i dodanie kolumny 'index_copy' zachowującej oryginalną kolejność
    df_excel = pd.read_excel(excel_file_path)
    df_excel['index_copy'] = df_excel.index

    # Połączenie DataFrame'ów
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")

    # Znalezienie indeksu ostatniego True w kolumnie 'do PBL' używając 'index_copy'
    last_true_index_copy = merged_df[merged_df['do PBL'] == True]['index_copy'].max()

    # Filtracja DataFrame, aby zwrócić wiersze, które następują po ostatnim True, używając 'index_copy'
    final_df = merged_df[merged_df['index_copy'] > last_true_index_copy][['Tytuł artykułu', 'Tekst artykułu', 'Link']]
    #final_df = merged_df[merged_df['index_copy'] > last_true_index_copy]
    return final_df
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
sampled_df = df.sample(n=100, random_state=42)
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, 
       df11, df12, df13, df14, df15, df16, df17, df18, df19, 
       df20, df21, df22, df23, df24, df25]

# Wybieranie 5 pierwszych wierszy z każdego DataFrame'u
first_five_each = [df.head(5) for df in dfs]
sampled_df =pd.concat(first_five_each, ignore_index=True)

#%%DF bez dodatkowych warunków:


def load_and_merge_data(json_file_path, excel_file_path, common_column='Link'):
    # Wczytanie danych z pliku JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)

    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']].astype(str)

    # Wczytanie danych z pliku Excel
    df_excel = pd.read_excel(excel_file_path)

    # Połączenie DataFrame'ów, zachowując wszystkie wiersze i kolumny z pliku Excel oraz pełny tekst z JSONa
    merged_df = pd.merge(df_excel, df_json, on=common_column, how="left")

    return merged_df

# Przykładowe użycie
# final_df = load_and_merge_data('path/to/json_file.json', 'path/to/excel_file.xlsx')
# print(final_df)

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModel, HerbertTokenizerFast
model_path = "C:/Users/dariu/model_5epoch_gatunek_large"
model_genre = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = HerbertTokenizerFast.from_pretrained(model_path)
import joblib



# W późniejszym czasie, aby wczytać LabelEncoder:
label_encoder = joblib.load('C:/Users/dariu/model_5epoch_gatunek_large/label_encoder_gatunek5.joblib')
# TRUE FALSE
model_path = "C:/Users/dariu/model_TRUE_FALSE_4epoch_base_514_tokens/"
model_t_f = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer_t_f =  HerbertTokenizerFast.from_pretrained(model_path)

label_encoder_t_f = joblib.load('C:/Users/dariu/model_TRUE_FALSE_4epoch_base_514_tokens/label_encoder_true_false4epoch_514_tokens.joblib')

model_path_hasla = "model_hasla_8epoch_base"
model_hasla = AutoModelForSequenceClassification.from_pretrained(model_path_hasla)
tokenizer_hasla = HerbertTokenizerFast.from_pretrained(model_path_hasla)




# W późniejszym czasie, aby wczytać LabelEncoder:
label_encoder_hasla = joblib.load('C:/Users/dariu/model_hasla_8epoch_base/label_encoder_hasla_base.joblib')
sampled_df['combined_text'] =sampled_df['Tytuł artykułu'].astype(str) + " </tytuł>" + sampled_df['Tekst artykułu'].astype(str)

def predict_categories(df, text_column):
    # Lista na przewidywane dane
    predictions_list = []
    
    for index, row in df.iterrows():
        text = row[text_column]
        
        # Tokenizacja i przewidywanie dla modelu True/False
        inputs_t_f = tokenizer_t_f(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs_t_f = model_t_f(**inputs_t_f)
        predictions_t_f = torch.softmax(outputs_t_f.logits, dim=1)
        predicted_index_t_f = predictions_t_f.argmax().item()
        predicted_label_t_f = label_encoder_t_f.inverse_transform([predicted_index_t_f])[0]
        
        # Inicjalizacja pustych wartości dla gatunku i haseł
        genre = ''
        haslo = ''
        
        # Jeśli wynik True, przeprowadź dalsze przewidywania
        if predicted_label_t_f == 'True':
            # Przewidywanie gatunku
            inputs_genre = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs_genre = model_genre(**inputs_genre)
            predictions_genre = torch.softmax(outputs_genre.logits, dim=1)
            predicted_index_genre = predictions_genre.argmax().item()
            genre = label_encoder.inverse_transform([predicted_index_genre])[0]
            
            # Przewidywanie hasła
            inputs_hasla = tokenizer_hasla(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs_hasla = model_hasla(**inputs_hasla)
            predictions_hasla = torch.softmax(outputs_hasla.logits, dim=1)
            predicted_index_hasla = predictions_hasla.argmax().item()
            haslo = label_encoder_hasla.inverse_transform([predicted_index_hasla])[0]
        
        # Dodanie przewidzianych wartości i innych kolumn z oryginalnego wiersza
        row_data = [text, predicted_label_t_f, genre, haslo] + [row[col] for col in df.columns if col != text_column]
        predictions_list.append(row_data)
    
    # Tworzenie nowych nazw kolumn dla wynikowego DataFrame
    new_columns = [text_column, 'True/False', 'Gatunek', 'Hasło'] + [col for col in df.columns if col != text_column]
    
    # Tworzenie DataFrame z przewidzianymi kategoriami i oryginalnymi danymi
    predicted_df = pd.DataFrame(predictions_list, columns=new_columns)
    
    return predicted_df

def predict_categories(df, text_column):
    predictions_list = []
    
    for index, row in df.iterrows():
        text = row[text_column]
        
        # Tokenizacja i przewidywanie dla modelu True/False
        inputs_t_f = tokenizer_t_f(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs_t_f = model_t_f(**inputs_t_f)
        predictions_t_f = torch.softmax(outputs_t_f.logits, dim=1)
        predicted_index_t_f = predictions_t_f.argmax().item()
        predicted_label_t_f = label_encoder_t_f.inverse_transform([predicted_index_t_f])[0]
        confidence_t_f = predictions_t_f.max().item() * 100  # Procent pewności
        
        genre = ''
        haslo = ''
        confidence_genre = ''  # Początkowa wartość pewności dla gatunku
        confidence_haslo = ''  # Początkowa wartość pewności dla hasła
        
        if predicted_label_t_f == 'True':
            # Przewidywanie gatunku
            inputs_genre = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs_genre = model_genre(**inputs_genre)
            predictions_genre = torch.softmax(outputs_genre.logits, dim=1)
            predicted_index_genre = predictions_genre.argmax().item()
            genre = label_encoder.inverse_transform([predicted_index_genre])[0]
            confidence_genre = predictions_genre.max().item() * 100  # Procent pewności
            
            # Przewidywanie hasła
            inputs_hasla = tokenizer_hasla(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs_hasla = model_hasla(**inputs_hasla)
            predictions_hasla = torch.softmax(outputs_hasla.logits, dim=1)
            predicted_index_hasla = predictions_hasla.argmax().item()
            haslo = label_encoder_hasla.inverse_transform([predicted_index_hasla])[0]
            confidence_haslo = predictions_hasla.max().item() * 100  # Procent pewności
        
        row_data = [text, predicted_label_t_f, confidence_t_f, genre, confidence_genre, haslo, confidence_haslo] + [row[col] for col in df.columns if col != text_column]
        predictions_list.append(row_data)
    
    new_columns = [text_column, 'True/False', 'Pewnosc T/F', 'Gatunek', 'Pewnosc Gatunku', 'Hasło', 'Pewnosc Hasła'] + [col for col in df.columns if col != text_column]
    
    predicted_df = pd.DataFrame(predictions_list, columns=new_columns)
    
    return predicted_df


# Przykład użycia funkcji:
result_df = predict_categories(sampled_df, 'combined_text')

result_df.to_csv('nowe_514_08_05_first_five.csv', sep='|', index=False)

df5.to_excel('results.xlsx', index=False, engine='openpyxl')
writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
result_df.to_excel(writer, sheet_name='Sheet1', index=False)
model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Utworzenie pipeline NER
nlp1 = pipeline("ner", model=model, tokenizer=tokenizer)
def combine_tokens(ner_results):
    combined_entities = []
    current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0, "start": None, "end": None}
    previous_index = None  # Zmienna do przechowywania indeksu poprzedniego tokenu

    for token in ner_results:
        # Sprawdzamy, czy bieżący token jest końcem słowa
        end_of_word = "</w>" in token['word']
        cleaned_word = token['word'].replace("</w>", "")

        # Sprawdzamy różnicę indeksów, jeśli poprzedni indeks jest ustawiony
        index_difference = token['index'] - previous_index if previous_index is not None else 0

        # Rozpoczęcie nowej jednostki
        if token['entity'].startswith('B-') or index_difference > 5:  # Dodatkowy warunek na różnicę indeksów
            if current_entity['word']:
                # Obliczamy średnią ocenę dla skompletowanej jednostki
                current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
                combined_entities.append(current_entity)
            current_entity = {"word": cleaned_word, "type": token['entity'][2:], "score_sum": token['score'],
                              "token_count": 1, "start": token['start'], "end": token['end']}
        # Kontynuacja obecnej jednostki
        elif token['entity'].startswith('I-') and current_entity['type'] == token['entity'][2:]:
            if previous_end_of_word:
                current_entity['word'] += " " + cleaned_word
            else:
                current_entity['word'] += cleaned_word
            current_entity['end'] = token['end']
            current_entity['score_sum'] += token['score']
            current_entity['token_count'] += 1

        previous_end_of_word = end_of_word
        previous_index = token['index']  # Aktualizacja indeksu poprzedniego tokenu

    # Dodajemy ostatnią jednostkę, jeśli istnieje
    if current_entity['word']:
        current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
        combined_entities.append(current_entity)

    return combined_entities    

threshold = 80

def group_similar_entities(entities, threshold):
    groups = []
    for entity in entities:
        added = False
        for group in groups:
            if any(fuzz.token_sort_ratio(entity, member) > threshold for member in group):
                group.append(entity)
                added = True
                break
        if not added:
            groups.append([entity])
    return groups

# def replace_entities_with_representatives(text, map):
#     # Sortowanie kluczy według długości tekstu rosnąco, aby najpierw zastąpić krótsze frazy
#     sorted_entities = sorted(map.keys(), key=len)
    
#     for entity in sorted_entities:
#         representative = map[entity]
#         # Zastąpienie klucza (bytu) jego wartością (reprezentantem) w tekście
#         text = text.replace(entity, representative)
    
#     return text
def replace_entities_with_representatives(text, map):
    # Tworzenie odwrotnego mapowania dla szybkiego sprawdzenia, czy dana fraza została już użyta jako zastąpienie
    reverse_map = {v: k for k, v in map.items()}
    used_replacements = set()  # Zbiór użytych zastąpień
    
    # Sortowanie kluczy według długości tekstu malejąco
    sorted_entities = sorted(map.keys(), key=len, reverse=True)
    
    for entity in sorted_entities:
        representative = map[entity]

        # Jeśli reprezentant był już użyty jako zastąpienie, pomijamy dalsze zastępowania tego reprezentanta
        if representative in used_replacements:
            continue
        
        # Zastępowanie tylko jeśli reprezentant nie jest częścią wcześniej zastąpionych fraz
        if not any(rep in text for rep in used_replacements if rep != entity):
            pattern = r'\b{}\b'.format(re.escape(entity))
            # Aktualizacja tekstu tylko, gdy fraza nie została jeszcze zastąpiona
            if re.search(pattern, text):
                text = re.sub(pattern, representative, text)
                used_replacements.add(representative)

    return text
# def replace_entities_with_representatives(text, map):
#     # Przygotowanie posortowanej listy fraz do zastąpienia na podstawie ich długości (od najdłuższej do najkrótszej)
#     sorted_entities = sorted(map.keys(), key=len, reverse=True)
#     output_text = ""  # Tutaj będziemy "odkładać" już przetworzone części tekstu
#     current_index = 0  # Indeks wskazujący na aktualną pozycję w tekście

#     while current_index < len(text):
#         replaced = False
#         for entity in sorted_entities:
#             # Szukanie frazy w tekście począwszy od bieżącego indeksu
#             match = re.search(r'\b{}\b'.format(re.escape(entity)), text[current_index:])
#             if match:
#                 start, end = match.span()
#                 # Dodanie do wynikowego tekstu części przed znalezioną frazą oraz zastąpionej frazy
#                 output_text += text[current_index:current_index+start] + map[entity]
#                 # Aktualizacja indeksu do pozycji za zastąpioną frazą
#                 current_index += end
#                 replaced = True
#                 break  # Przechodzimy do kolejnego fragmentu tekstu po zastąpieniu

#         # Jeśli w danym przebiegu pętli nie znaleziono żadnej frazy do zastąpienia, przesuwamy indeks o jeden
#         # i dodajemy bieżący znak do wynikowego tekstu
#         if not replaced:
#             output_text += text[current_index]
#             current_index += 1

#     return output_text


import requests
import re

def preprocess_text(text):
    # Usuwanie dat z tekstu, np. "Emma Goldman, 1869-1940" staje się "Emma Goldman"
    return re.sub(r',?\s*\d{4}(-\d{4})?', '', text)

def check_viaf_with_fuzzy_match(entity_name, threshold=87):
    base_url = "http://viaf.org/viaf/AutoSuggest"
    query_params = {'query': entity_name}
    best_match = None
    best_score = 0
    
    try:
        response = requests.get(base_url, params=query_params)
        response.raise_for_status()
        data = response.json()

        # Dodatkowe sprawdzenie, czy 'result' jest w danych i czy nie jest None
        if data and data.get('result') is not None:
            for result in data['result'][:10]:
                original_term = result.get('term')
                score_with_date = fuzz.token_sort_ratio(entity_name, original_term)
                if score_with_date > best_score and score_with_date >= threshold:
                    best_score = score_with_date
                    best_match = result
                
                term_without_date = preprocess_text(original_term)
                score_without_date = fuzz.token_sort_ratio(entity_name, term_without_date)
                if score_without_date > best_score and score_without_date >= threshold:
                    best_score = score_without_date
                    best_match = result

    except requests.RequestException as e:
        print(f"Error querying VIAF: {e}")
    
    if best_match:
        viaf_id = best_match.get('viafid')
        return f"http://viaf.org/viaf/{viaf_id}", best_score
    
    return None, None
from tqdm import tqdm

nlp = spacy.load("pl_core_news_lg")
result_df['Chosen_Entity'] = pd.NA
result_df['VIAF_URL'] = pd.NA
result_df['Entity_Type'] = pd.NA

for index, row in tqdm(result_df[result_df['True/False'] == "True"].iterrows()):
    text = row['combined_text']
    tokens = tokenizer.tokenize(text)
    max_tokens = 512  # Przykładowe ograniczenie modelu
    token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]
    # Analiza każdego fragmentu osobno
    ner_results = []
    for fragment in fragments:
        ner_results.extend(nlp1(fragment))
    combined_entities = combine_tokens(ner_results)
    
    combined_entities_selected = [entity for entity in combined_entities if entity['score'] >= 0.90]
    entities = [(entity['word'], entity['type']) for entity in combined_entities_selected]
    
    
    doc = nlp(text.lower())
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    #lemmatized_text=text.lower()

    # Lematyzacja bytów i grupowanie
    lemmatized_entities = []
    entity_lemmatization_dict = {}
    for entity in entities:
        doc_entity = nlp(entity[0].lower())
        lemmatized_entity = " ".join([token.lemma_ for token in doc_entity])
        lemmatized_entities.append(lemmatized_entity)
        if lemmatized_entity not in entity_lemmatization_dict:
            entity_lemmatization_dict[lemmatized_entity] = {entity}
        else:
            entity_lemmatization_dict[lemmatized_entity].add(entity)
    
    entity_groups = group_similar_entities(lemmatized_entities, threshold)
    representatives = [sorted(group, key=lambda x: len(x))[0] for group in entity_groups]

    entity_to_representative_map = {}
    for group in entity_groups:
        representative = sorted(group, key=lambda x: (len(x), x))[0]
        for entity in group:
            entity_to_representative_map[entity] = representative

    
    updated_text = replace_entities_with_representatives(lemmatized_text, entity_to_representative_map)
    list_of_new_entities = list(set(entity_to_representative_map.values()))
    
    entity_counts = {entity: 0 for entity in list_of_new_entities}
    title_end_pos = updated_text.find("< /tytuł >")
    if title_end_pos == -1:
        title_end_pos = updated_text.find("< /tytuł>")
    
    for entity in list_of_new_entities:
        total_occurrences = updated_text.count(entity)
        entity_counts[entity] += total_occurrences
        if updated_text.find(entity) < title_end_pos:
            entity_counts[entity] += 50
    
    sorted_entity_counts = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    choosen_ents = [ent for ent in sorted_entity_counts if ent[1] > 5]
    
    # Dodawanie informacji o wybranym bycie do list
    
    if choosen_ents:
        first_entity_info = choosen_ents[0]
        result_df.at[index, 'Chosen_Entity'] = first_entity_info[0]
        
        original_entities = entity_lemmatization_dict.get(first_entity_info[0], [])
        viaf_url, entity_type = None, "Not found"
        if original_entities:
            viaf_url, _ = check_viaf_with_fuzzy_match(next(iter(original_entities))[0])  # Pobieranie pierwszego elementu z setu
            entity_type = next(iter(original_entities))[1]
        
        result_df.at[index, 'VIAF_URL'] = viaf_url if viaf_url else "Not found"
        result_df.at[index, 'Entity_Type'] = entity_type
    else:
        result_df.at[index, 'Chosen_Entity'] = pd.NA
        result_df.at[index, 'VIAF_URL'] = "Not found"
        result_df.at[index, 'Entity_Type'] = pd.NA
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
    if choosen_ents:
        first_entity_info = choosen_ents[0]
        chosen_entities.append(first_entity_info[0])
        # Szukanie VIAF i typu bytu
        original_entities = entity_lemmatization_dict.get(first_entity_info[0], [])
        viaf_url, entity_type = None, list(original_entities)[0][1]
        for original_entity in original_entities:
            _entity = next(iter(original_entity))  # Pobieranie pierwszego elementu z setu
            viaf_result, _ = check_viaf_with_fuzzy_match(_entity)
            if viaf_result:
                viaf_url = viaf_result
                 
                break
        viaf_urls.append(viaf_url if viaf_url else "Not found")
        entity_types.append(entity_type)
    else:
        chosen_entities.append(None)
        viaf_urls.append("Not found")
        entity_types.append(None)

# Dodawanie wyników do DataFrame
result_df['Chosen_Entity'] = pd.Series(chosen_entities, index=result_df.index)
result_df['VIAF_URL'] = pd.Series(viaf_urls, index=result_df.index)
result_df['Entity_Type'] = pd.Series(entity_types, index=result_df.index)









for index, row in result_df[result_df['True/False'] == "True"].iterrows():
    text=row['combined_text']
    max_length = 512  # Maksymalna długość fragmentu tekstu, dostosuj w zależności od modelu i ograniczeń pamięci
    fragments = textwrap.wrap(text, max_length, break_long_words=False, replace_whitespace=False)

    # Analiza każdego fragmentu osobno
    ner_results = []
    for fragment in fragments:
        ner_results.extend(nlp1(fragment))
    combined_entities = combine_tokens(ner_results)
    combined_entities_selected=[]
    for entity in combined_entities:
        if entity['score']>=0.90:
            combined_entities_selected.append(entity)
    entities = [(entity['word'],entity['type']) for entity in combined_entities_selected]
    nlp = spacy.load("pl_core_news_lg")

    # Przetwarzanie tekstu
    doc = nlp(text.lower())
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    # Ponowne przetworzenie lematyzowanego tekstu, aby umożliwić analizę zdań


    # Lematyzacja bytów
    lemmatized_entities = []
    entity_lemmatization_dict = {}
    for entity in entities:
        doc_entity = nlp(entity[0].lower())
        lemmatized_entity = " ".join([token.lemma_ for token in doc_entity])
        lemmatized_entities.append(lemmatized_entity)
        if lemmatized_entity in entity_lemmatization_dict:
            # Dodajemy oryginalną formę do set, aby zapewnić unikalność
            entity_lemmatization_dict[lemmatized_entity].add(entity)
        else:
            # Tworzymy nowy set z oryginalną formą jako pierwszym elementem
            entity_lemmatization_dict[lemmatized_entity] = {entity}
            
    entity_groups = group_similar_entities(lemmatized_entities, threshold)        
    representatives = [sorted(group, key=lambda x: len(x))[0] for group in entity_groups]

    entity_to_representative_map = {}
    for group in entity_groups:
        representative = sorted(group, key=lambda x: len(x))[0]
        for entity in group:
            entity_to_representative_map[entity] = representative 
    updated_text = replace_entities_with_representatives(lemmatized_text, entity_to_representative_map)        
    list_of_new_entities=(list(entity_to_representative_map.values()))
    unique(list_of_new_entities)
    entity_counts = {entity: 0 for entity in list_of_new_entities}

    # Znalezienie końca tytułu
    title_end_pos = updated_text.find("< /tytuł >")

    # Zliczanie wystąpień
    for entity in list_of_new_entities:
        # Liczenie wszystkich wystąpień bytu
        total_occurrences = updated_text.count(entity)
        entity_counts[entity] += total_occurrences
        
        # Sprawdzenie, czy byt występuje w tytule i dodanie dodatkowego punktu
        if updated_text.find(entity) < title_end_pos:
            entity_counts[entity] += 50

    # Sortowanie i wyświetlanie wyników
    sorted_entity_counts = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    choosen_ents=[]
    for ent in sorted_entity_counts:
        if ent[1]>5:
            choosen_ents.append(ent)
            
    if choosen_ents:
        # Skupiamy się tylko na pierwszym wybranym bycie
        first_entity_info = choosen_ents[0]  # Uwzględniamy, że choosen_ents może zawierać więcej informacji niż tylko nazwę bytu
        first_entity = first_entity_info[0]  # Nazwa bytu
    
        # Inicjalizujemy zmienne dla wyników VIAF i typu bytu
        viaf_url = None
        entity_type = None
    
        # Szukamy oryginalnego bytu i jego typu w naszym słowniku
        if first_entity in entity_lemmatization_dict:
            for original_entity, _entity_type in entity_lemmatization_dict[first_entity]:
                entity_type = _entity_type  # Ustawiamy typ bytu
                viaf_result, _ = check_viaf_with_fuzzy_match(original_entity[0])  # Sprawdzamy pierwszy oryginalny byt w VIAF
                
                # Sprawdzamy, czy otrzymaliśmy wynik z VIAF
                if viaf_result:
                    viaf_url = viaf_result  # Przypisujemy URL VIAF, jeśli znaleziono
                    break  # Zatrzymujemy pętlę, jeśli znaleźliśmy wynik
    
        # Dodajemy informacje do result_df
        result_df['Chosen_Entity'] = first_entity
        result_df['VIAF_URL'] = viaf_url if viaf_url else "Not found"
        result_df['Entity_Type'] = entity_type

# Dalej możemy kontynuować z zapisem do Excela lub innymi operacjami
result_df.to_excel('updated_results.xlsx', index=False, engine='openpyxl')
    
import pandas as pd
import requests
from fuzzywuzzy import fuzz
import spacy
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import textwrap

# Przygotowanie modelu NER i spaCy
model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer)
nlp_spacy = spacy.load("pl_core_news_lg")

# Funkcje pomocnicze
def lemmatize_text(text):
    return " ".join([token.lemma_ for token in nlp_spacy(text.lower())])

def check_viaf_with_fuzzy_match(query, threshold=90):
    base_url = "http://viaf.org/viaf/AutoSuggest"
    params = {'query': query}
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            for result in data.get('result', []):
                if fuzz.partial_ratio(query.lower(), result['term'].lower()) >= threshold:
                    return f"http://viaf.org/viaf/{result['viafid']}"
    except Exception as e:
        print(f"Error querying VIAF: {e}")
    return None

# Główna funkcja przetwarzająca tekst
def process_text(text):
    fragments = textwrap.wrap(text, 512, break_long_words=False, replace_whitespace=False)
    ner_results = [nlp_ner(fragment) for fragment in fragments]
    entities = [(result['word'].replace('</w>', ''), result['entity'].split('-')[1]) for sublist in ner_results for result in sublist if result['score'] > 0.9]
    lemmatized_entities = [(lemmatize_text(entity), type) for entity, type in entities]
    lemmatized_text = lemmatize_text(text)
    entity_counts = {entity: lemmatized_text.count(entity) for entity, _ in lemmatized_entities}
    viaf_links = {entity: check_viaf_with_fuzzy_match(entity) for entity in entity_counts.keys() if entity_counts[entity] > 0}
    return entity_counts, viaf_links

# Aktualizacja DataFrame
def update_df_with_entities_and_viaf(df):
    for index, row in df.iterrows():
        if row['True/False']:
            entity_counts, viaf_links = process_text(row['combined_text'])
            for entity, count in entity_counts.items():
                df.at[index, f'entity_{entity}_count'] = count
                df.at[index, f'entity_{entity}_viaf'] = viaf_links.get(entity)
    return df

# Przykładowe użycie
df = pd.DataFrame({
    'combined_text': ["""Zmarł Feliks W. Kres, autor „Księgi Całości”, prekursor polskiej fantasy </tytuł> Jeden z najważniejszych polskich autorów fantasty, autor popularnego cyklu „Księga Całości”, Feliks W. Kres nie żyje. Pisarz zmarł w wieku 56 lat po walce z chorobą. Wiadomość o jego śmierci przekazało wydawnictwo Fabryka Słów.

    „Nie ma słów właściwych, by przekazać taką informację. Po zaciętej walce z chorobą odszedł od nas Feliks W. Kres. Kresie, pozostawiłeś po sobie pustkę, której jeszcze nie umiemy ogarnąć. Pytania, na które już nigdy nie poznamy odpowiedzi. Myśli, którymi już się z tobą nie podzielimy. Teraz jednak, przede wszystkim, jesteśmy całym sercem z twoją żoną i najbliższymi” – napisało wydawnictwo na swoim profilu na Facebooku.

    Urodzony 1 czerwca 1966 roku Feliks W. Kres (właściwie nazywał się Witold Chmielecki) był prekursorem polskiej fantasy. Zadebiutował już 1983 roku utworem „Mag” nadesłanym na konkurs w „Fantastyce”. Było to pierwsze w Polsce opowiadanie napisane w tym gatunku, które ukazało się na łamach profesjonalnego czasopisma. Trzy lata później miał już złożoną do druku swoją książkę. Na księgarskich półkach zobaczył ją dopiero na początku lat 90.

    „Dla krajowego twórcy to był najgorszy czas z możliwych. Nadrabialiśmy półwiecze zaległości – my, Polacy. Drukowano wtedy wszystko, co tylko było opatrzone anglosaskim nazwiskiem. Jako czytelnik byłem wniebowzięty, wreszcie miałem pełne półki i wybór. Natomiast jako autor – bo jeszcze nie pisarz – nosiłem maszynopisy od wydawcy do wydawcy. Radzono mi – to dzisiaj brzmi anegdotycznie – bym sygnował książki Felix V. Craes albo w ogóle – bo ja wiem… – John Brown. Byle nie rodzimo brzmiącym nazwiskiem” – wspominał w książce „Galeria dla dorosłych”, dodając, że wolał jednak pozostać przy swoim polsko brzmiącym pseudonimie.

    W latach 1991-1996 Kres co roku wydawał książkę. Jak sam przyznawał, chyba żaden inny polski autor-fantasta nie mógł tego o sobie powiedzieć. „Nie jestem dziś szczególnie dumny z nagród, które wówczas zebrałem, bo też jaka była konkurencja?… Raz i drugi, pamiętam, napotkano poważne trudności ze znalezieniem pięciu dzieł rodzimej produkcji, niezbędnych do tego, by w ogóle przeprowadzić konkurs – mówiąc inaczej: cokolwiek napisano, a otarło się o fantastykę, automatycznie dostawało nominację do nagrody, choćby nawet autor nie znał gramatyki i ortografii” – pisał z wrodzonym dystansem.

    Bez względu na to, jaki poziom prezentowała konkurencja, Feliks W. Kres nie przeszedł do historii polskiej fantasy tylko dlatego, że był jej prekursorem. W przypadku autora nie było mowy o literackiej nieporadności. Stworzony przez niego cykl „Księga Całości”, rozgrywający się w świecie Szereru, gdzie żyją tylko trzy rozumne gatunki – ludzie, koty i sępy – uchodzi za prawdziwą klasykę gatunku i jedno z najważniejszych dzieł polskiej fantasy. Pozycji tej nie zagroził nawet trochę młodszy i znacznie popularniejszy „Wiedźmin” Andrzeja Sapkowskiego.

    W 2011 roku Kres oświadczył, że rezygnuje z dalszego pisania. Jego powrót po blisko dekadzie, ogłoszony przez wydawnictwo Fabryka Słów, był w polskim świecie fantastyki dużym wydarzeniem. Najpierw wznowiono w poprawionej wersji tomy „Księgi Całości”, które ukazały się przed laty. Pierwszy nowy tom, „Najstarsza z Potęg”, zapowiedziany jest na listopad. Autor nie dożył jego publikacji."""],
    'True/False': [True]
})

updated_df = update_df_with_entities_and_viaf(df)
print(updated_df)










text='''Cień świata. W scenicznej interpretacji „Nowych Aten” Maciej Gorczyński opowiada o sile wyobraźni i podstępach rozumu. Prapremiera spektaklu na podstawie pierwszej polskiej encyklopedii pióra księdza Benedykta Chmielowskiego była najjaśniejszym punktem VI Festiwalu Teatrów Błądzących w Gardzienicach.

Opisać systematykę stworzenia – oto zadanie heroiczne, godne świętego męża. Takie ambicje miał Chmielowski, autor pierwszej polskiej encyklopedii. Powstałe w połowie XVIII wieku dzieło nosiło tytuł „Nowe Ateny”. W porównaniu z dokonaniami działających w tym samym czasie francuskich encyklopedystów, wydaje się ono kuriozalne. Podjęcie trudu tworzenia encyklopedii motywowane było wiarą w rozum, motywacje miał więc Chmielowski podobne do Francuzów. W jego dziele znaleźć można jednak także praktyczne informacje na temat bazyliszków, smoków i czartów. Czy jest to powód wystarczający, aby księgę tę zostawić w biblioteczkach zaściankowej szlachty, a samemu wrócić do Denisa Diderota i jego kolegów? Nie może być zaskoczeniem, że teatr idzie w sukurs wyobraźni, co w wyreżyserowanych przez Macieja Gorczyńskiego „Nowych Atenach” cieszy szczególnie. Jest to przedstawienie bogate frenetycznymi momentami i adorujące imaginację, a w ten sposób opowiadające się po jednej ze stron konfliktu pomiędzy racjonalizmem i wyobraźnią. Kapłanowi z Firlejowa przydany w nim został nie lada sojusznik – William Blake.

Nad sceną zawieszono duże gumowe piłki i rzucono nań kosmiczne wizualizacje, dzięki czemu spełnić się mogły słowa zapomnianego poety: „Te gwiazdy to są kule, i na hakach wiszą”. Z tyłu sceny ściana-labirynt z delikatnego, zwiewnego materiału. Znikający w niej aktorzy zamieniali się w cienie. W ten sposób przypominali, że cała encyklopedyczna systematyzacja jest zaledwie próbą chwytania odbicia rzeczywistości, której promienie padają na powierzchnię wyobraźni. Właśnie dzięki wyobraźni ptaki/owady/motyle mogły przybrać postać delikatnych chustek fruwających na oddechach tańczących aktorów. Taniec był najważniejszą techniką teatralną w przedstawieniu, które za jego pomocą dotykało różnych tematów: od sfery duchowej jednostki, przez relacje międzyludzkie, narodowe stereotypy, aż po miejsce człowieka we wszechświecie.

Prapremiera przedstawienia odbyła się w Gardzienicach w ramach VI Festiwalu Teatrów Błądzących. Gorczyński z Ośrodkiem Praktyk Teatralnych współpracuje od wielu lat, był tam między innymi aktorem, ale i archiwistą. To wyjaśnia zaangażowanie przez niego estetyki, która została wypracowana przez zespół Włodzimierza Staniewskiego. Imponuje jej twórcze przekształcenie, możliwe zapewne dzięki temu, że aktorzy w „Nowych Atenach” to nie „ludzie gardzieniccy”, a studenci Wydziału Teatru Tańca krakowskiej PWST w Bytomiu. Angażując ich, reżyser przetacza świeżą krew w ramy starzejącej się konwencji teatralnej i ożywia tradycję, która w głównym zespole „Gardzienic” zdaje się obumierać i kostnieć. Młodzi artyści imponowali warsztatem. Skoncentrowani na dopracowanej choreografii – niekiedy popisowej (jak w przypadku Daniela Leżonia, grającego księdza i czarnoksiężnika), innym razem dowodzącej zespołowego porozumienia – umiejętnie żonglując konwencjami inspirowanymi muzycznością (w szerokim tego słowa znaczeniu, odnoszonym do ruchu, pieśni, rytmu). Oszczędny biały śpiew przypominał gardzienickie „Metamorfozy”, ale nie uciekał w folklor. Fisharmonia oraz poręczny dzwon nie tylko generowały dźwięki, ale także pełniły zadania scenograficzne. Ową muzyczność, zakorzenioną gdzieś w powidokach ludowości, skontrowano wybrzmiewającym w finale utworem Marianne Faithfull „City Of Quartz”. Pozytywkowa melodia i panosząca się w tekście „kurwa babilońska” korespondowały z przechodzącym wcześniej przez scenę korowodem średniowiecznych idiotów/opętanych, w swej – znakomicie odegranej – pokraczności sięgających wyższych rejestrów estetyki. Wszystko to podkreślało nierozłączność przeciwieństw. W całym przedstawieniu Gorczyński dość uważnie bada możliwość zaślubin tego, co ciemne, z tym, co świetliste, piekła z niebem, jak mógłby stwierdzić Blake.

Dopóki taniec i choreografia koncentrowały się na „Nowych Atenach”, całość przedstawienia wchodziła w rejestry niemal baśniowe. Każde sięgniecie do Blake’a działało jednak na niekorzyść spektaklu, ukazując pęknięcia, w których gubiła się przewodnia koncepcja widowiska. Obecność myśli angielskiego wizjonera przejawiała się bardziej w ruchu niż w słowie, podczas gdy muzycznie zapętlone cytaty z Chmielowskiego odpowiadały za integralność świata przedstawionego, którego centrum był sam ksiądz/czarnoksiężnik. Leżoń – operując maską, będącą atrybutem ciemnych sił – raz po raz wizualizował dwoistość ludzkiej natury. „Jest czort! Jest zło! Jest system kopernikański! Ale jest też cebulka…”. W rozdwojonym bohaterze odbijało się szaleństwo. Widowiskowe, transgresyjne momenty przypominające opętanie przestrzegały przed fanatyczną wiarą w cokolwiek. Nie tylko w koguty o wężowych ogonach, ale także w zdrowy rozsądek, który podpowiadał kapłanowi, że teoria Mikołaja Kopernika nie może być prawdziwa. Duchowny dzielnie walczył z bestią ślepego zawierzenia – mając na uwadze, że „smoka pokonać trudno, ale starać się trzeba”.

Demoniczność kontrowano dużą dawką humoru. Gorczyński i współautor adaptacji Kajetan Mojsak
 nie ulegli jednak możliwej pokusie i nie uczynili z „Nowych Aten” pretekstu do rechotu nad zaściankowością i ciemnotą. Komizmem artyści grali subtelnie, choć wyraźnie, jak w chwilach prezentacji żołnierza polskiego o dumnie wyprężonej piersi otulonej westchnieniami omdlewających panien, czy też kozackiego najazdu, gdzie stepowi wojacy dosiadają drewnianych wierzchowców gniadej maści. Jakie jest krzesło, każdy widzi – mógłby dopowiedzieć Chmielowski. Jaki jest teatr – także. Gorczyński potrafi spojrzeć na rzeczywistość przez teatr jak przez lunetę. W „Nowych Atenach” znalazł punkt, w którym wyobraźnia poddaje w wątpliwość rozsądek i zmusza do podważenia prymatu rozumu, co szczególnie ważne w dzisiejszych czasach nieustannego postępu.'''
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Przewidywanie za pomocą modelu
model_genre.eval()  # Ustawienie modelu w tryb ewaluacji
with torch.no_grad():  # Wyłączenie obliczeń gradientów
    outputs = model_genre(**inputs)

# Pobranie wyników
predictions = torch.softmax(outputs.logits, dim=1)
predicted_index = predictions.argmax().item()
predicted_label = label_encoder.inverse_transform([predicted_index])[0]

print(f"Przewidziana kategoria: {predicted_label}")
confidence = predictions[0, predicted_index].item()

print(f"Przewidziana kategoria: {predicted_label} z pewnością: {confidence * 100:.2f}%")
#%% TRUE FALSE
model_path = "C:/Users/dariu/model_TRUE_FALSE_4epoch/"
model_t_f = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer_t_f = AutoTokenizer.from_pretrained(model_path)
import joblib



# W późniejszym czasie, aby wczytać LabelEncoder:
label_encoder_t_f = joblib.load('C:/Users/dariu/model_TRUE_FALSE_4epoch/label_encoder_true_false4epoch.joblib')


inputs = tokenizer_t_f(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Przewidywanie za pomocą modelu
model_t_f.eval()  # Ustawienie modelu w tryb ewaluacji
with torch.no_grad():  # Wyłączenie obliczeń gradientów
    outputs_t_f = model_t_f(**inputs)

# Pobranie wyników
predictions = torch.softmax(outputs_t_f.logits, dim=1)
predicted_index = predictions.argmax().item()
predicted_label = label_encoder_t_f.inverse_transform([predicted_index])[0]

print(f"Przewidziana kategoria: {predicted_label}")
confidence = predictions[0, predicted_index].item()

print(f"Przewidziana kategoria: {predicted_label} z pewnością: {confidence * 100:.2f}%")


#%% HASLA

model_path_hasla = "model_hasla_6epoch"
model_hasla = AutoModelForSequenceClassification.from_pretrained(model_path_hasla)
tokenizer_hasla = AutoTokenizer.from_pretrained(model_path_hasla)
import joblib



# W późniejszym czasie, aby wczytać LabelEncoder:
label_encoder_hasla = joblib.load('C:/Users/dariu/model_hasla_6epoch/label_encoder_hasla.joblib')


inputs = tokenizer_hasla(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Przewidywanie za pomocą modelu
model_hasla.eval()  # Ustawienie modelu w tryb ewaluacji
with torch.no_grad():  # Wyłączenie obliczeń gradientów
    outputs = model_hasla(**inputs)

# Pobranie wyników
predictions = torch.softmax(outputs.logits, dim=1)
predicted_index = predictions.argmax().item()
predicted_label = label_encoder_hasla.inverse_transform([predicted_index])[0]

print(f"Przewidziana kategoria: {predicted_label}")

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib

# Krok 1: Wczytanie modelu, tokenizatora i LabelEncoder
model_path = "C:/Users/User/Desktop/materiał_do_treningu/model_TRUE_FALSE_5epoch"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
label_encoder = joblib.load('C:/Users/User/Desktop/materiał_do_treningu/label_encoder_true_false5epoch.joblib')

# Przygotuj DataFrame z tekstem do klasyfikacji
df = pd.DataFrame({
    'combined_text': ["Przykładowy tekst do klasyfikacji True/False", "Inny tekst do sprawdzenia"]
})

# Krok 2: Przygotowanie danych do predykcji
def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

encoded_texts = encode_texts(df['combined_text'].tolist())

# Krok 3: Wykonanie predykcji
model.eval()  # Przestawienie modelu w tryb ewaluacji
with torch.no_grad():
    outputs = model(**encoded_texts)
predictions = torch.softmax(outputs.logits, dim=1).argmax(dim=1)

# Krok 4: Interpretacja wyników
df['Prediction'] = label_encoder.inverse_transform(predictions.numpy())

print(df[['combined_text', 'Prediction']])





#%%Zero shot przykład
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
sequence_to_classify = "Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU"
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)

from transformers import pipeline
import pandas as pd

# Załadowanie zero-shot classification pipeline z wielojęzycznym modelem
classifier = pipeline("zero-shot-classification", model="xlm-roberta-large")

# Załadowanie etykiet z pliku Excel
df = pd.read_excel('/path/to/your/labels.xlsx')  # Podmień na prawdziwą ścieżkę do pliku Excel
labels = df['ColumnWithLabels'].tolist()  # Podmień 'ColumnWithLabels' na nazwę kolumny z etykietami

# Przykładowy tekst do klasyfikacji
text = "Tutaj wpisz tekst, który chcesz sklasyfikować."

# Dokonanie klasyfikacji zero-shot
results = classifier(text, candidate_labels=labels)

# Wyświetlenie wyników
print(results)


#%%GLINER NER
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_base")

text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

labels = ["person", "award", "date", "competitions", "teams"]

entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
    
#%%Zero shot learning   
from transformers import pipeline
import pandas as pd

# Załadowanie zero-shot classification pipeline z wielojęzycznym modelem
classifier = pipeline("zero-shot-classification", model="xlm-roberta-large")

# Załadowanie etykiet z pliku Excel
df = pd.read_excel('C:/Users/dariu/Mapowanie działów.xlsx')  # Podmień na prawdziwą ścieżkę do pliku Excel
labels = df['string uproszczony'].unique().tolist() 


# Dokonanie klasyfikacji zero-shot
results = classifier(text, candidate_labels=labels)
results = []
for i, label in enumerate(labels):
    print(f"Sprawdzam etykietę {i+1}/{len(labels)}: {label}")
    result = classifier(text, candidate_labels=[label], multi_label=True)
    results.append(result)
# Wyświetlenie wyników
print(results)


def classify_text(label):
    print(f"Sprawdzam etykietę: {label}")
    result = classifier(text, candidate_labels=[label], multi_label=True)
    return label, result
from transformers import pipeline
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
# Używanie ThreadPoolExecutor do klasyfikacji etykiet w wielu wątkach
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_label = {executor.submit(classify_text, label): label for label in labels}
    for future in as_completed(future_to_label):
        label = future_to_label[future]
        try:
            data = future.result()
            results.append(data)
        except Exception as exc:
            print(f'{label} wygenerował wyjątek: {exc}')
            
            

