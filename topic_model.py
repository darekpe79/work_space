# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:07:19 2024

@author: dariu
"""
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import re
from nltk.corpus import stopwords
import nltk
import spacy
import joblib
from sklearn.preprocessing import LabelEncoder
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
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
df1 = load_and_merge_data(json_file_path, excel_file_path)

# -*- coding: utf-8 -*-
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
from transformers import AutoTokenizer, AutoModel
import torch
from bertopic import BERTopic

# Ustawienie, że wszystkie operacje mają być wykonane na CPU
device = torch.device("cpu")

# Załaduj tokenizer i model HerBERT
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModel.from_pretrained("allegro/herbert-base-cased").to(device)

# Funkcja do przekształcania tekstów w wektory
def embed_text(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.pooler_output

# Przykładowe dokumenty
docs = df1['Tekst artykułu'].to_list()

# Zamień teksty na wektory
import numpy as np

embeddings = np.array([embed_text(doc).squeeze().detach().cpu().numpy() for doc in docs])

# Sprawdź, czy osadzenia są poprawne
print("Kształt osadzeń:", embeddings.shape)
print("Przykładowe osadzenia:", embeddings[0][:10])  # Wyświetl pierwsze 10 elementów pierwszego osadzenia

# Kontynuuj tylko, jeśli osadzenia wydają się być poprawne
if embeddings.shape[0] > 0 and embeddings.shape[1] > 0:
    # Użyj BERTopic z własnymi wektorami
    topic_model = BERTopic(embedding_model=None)
    topics, _ = topic_model.fit_transform(docs, embeddings)

    # Wyświetl wyniki
    print(topic_model.get_topic_info())  # Podsumowanie tematów
else:
    print("Problem z osadzeniami, sprawdź generowanie wektorów.")
print(topics[2])
print("\nNazwa tematu dla pierwszego tekstu:", topic_model.get_topic(topics[2]))
# Wyświetlenie słów kluczowych dla tematu nr 4
topic_info = topic_model.get_topic(4)
print("Słowa kluczowe dla tematu 4:", topic_info)
