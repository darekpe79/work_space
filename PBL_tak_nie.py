# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:03:41 2024

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

    # Znalezienie indeksu ostatniego 'True' w kolumnie 'do PBL', gdzie 'hasła przedmiotowe' jest wypełnione
    last_true_filled_index = merged_df[(merged_df['do PBL'] == True) & (merged_df['hasła przedmiotowe'].notna())].index[-1]

    # Ograniczenie DataFrame do wierszy do ostatniego 'True' włącznie, gdzie 'hasła przedmiotowe' jest wypełnione
    merged_df = merged_df.loc[:last_true_filled_index]
    merged_df = merged_df.reset_index(drop=True)

    # Ograniczenie do wybranych kolumn
    selected_columns = merged_df[selected_columns_list]

    return selected_columns



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
train_test_dataset = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    'train': train_test_dataset['train'],
    'test': train_test_dataset['test']
})

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir="C:/Users/User/Desktop/materiał_do_treningu/",
    num_train_epochs=4,              # liczba epok
    per_device_train_batch_size=4,   # rozmiar batcha
    per_device_eval_batch_size=4,
    warmup_steps=500,                # kroki rozgrzewki
    weight_decay=0.01,               # waga decay
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_steps=100,
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

model_path = "C:/Users/User/Desktop/materiał_do_treningu/model_TRUE_FALSE_4epoch"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
import joblib

# Zapisanie LabelEncoder
joblib.dump(label_encoder, 'C:/Users/User/Desktop/materiał_do_treningu/label_encoder_true_false.joblib')


#%% Logistyczna regresja
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

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
    text = ' '.join(word for word in text.split () if word not in stop_words)
    return text

df = combined_df
# Połącz tytuł i tekst artykułu w jednym polu
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']
df['combined_text']=df['combined_text'].apply(clean_text)
from sklearn.preprocessing import LabelEncoder
df['do PBL'] = df['do PBL'].astype(str)
# Kodowanie etykiet
label_encoder = LabelEncoder()
df['do PBL'] = label_encoder.fit_transform(df['do PBL'])
# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['do PBL'], test_size=0.2, random_state=42)

# Przekształcenie tekstu na reprezentację TF-IDF
vectorizer = TfidfVectorizer(max_features=50000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Trenowanie modelu regresji logistycznej
model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

# Predykcja na zbiorze testowym
predictions = model.predict(X_test_tfidf)

# Ocena modelu
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

import joblib

# Zapisanie modelu regresji logistycznej
joblib.dump(model, 'logistic_regression_model.pkl')

# Opcjonalnie, zapisz również TfidfVectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

import joblib

# Zapisanie LabelEncoder
joblib.dump(label_encoder, 'label_encoder_logistic_reg.pkl')


#%%
loaded_model = joblib.load('logistic_regression_model.pkl')

# Wczytanie TfidfVectorizer, jeśli został zapisany
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder_logistic_reg.pkl')
# results = []

# for i in range(100, 200):
#     title = df.iloc[i]['Tytuł artykułu']
#     text = df.iloc[i]['Tekst artykułu']
#     full_text = title + " " + text
#     tfidf = loaded_vectorizer.transform([full_text])
#     predicted = loaded_model.predict(tfidf)
#     predicted_label = label_encoder.inverse_transform(predicted)[0]

#     results.append({
#         'Title': title,
#         'Text': text,
#         'Original': df.iloc[i]['do PBL'],
#         'Predicted': predicted_label
#     })

# results_df = pd.DataFrame(results)



# Wczytanie modelu
loaded_model = joblib.load('logistic_regression_model.pkl')

# Wczytanie TfidfVectorizer, jeśli został zapisany
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Przygotowanie danych do przewidywania
# (przykład dla jednego rekordu, ale można to zrobić dla wielu)
sample_title = "Olga Tokarczuk w kapitule konkursu na najlepszy polski scenariusz"
sample_text = '''Wkrótce odbędzie się 12 edycja Script Fiesty, czyli festiwalu poświęconego scenopisarstwu. Podczas trwania wydarzenia odbędą się dwa konkursy. W kapitule jednego z nich zasiądzie polska noblistka – Olga Tokarczuk. Olga Tokarczuk to jedna z najważniejszych współczesnych polskich pisarek, która w 2018 roku uhonorowana została Nagrodą Nobla w dziedzinie literatury, z kolei w zeszłym roku został jej przyznany akademicki tytuł honorowy w Uniwersytecie w Tel Awiwie. Autorka "Ksiąg Jakubowych" nie zamyka się tylko na tworzenie tekstów literackich i obecnie pracuje także nad grą na bazie jej twórczości, a teraz trafiła do kapituły konkursu Script Fiesta. Script Fiesta to wyjątkowy festiwal filmowy skupiony wokół scenariuszy i słowa. W tym roku od 18 do 21 kwietnia będzie trwała jego 12. edycja. Jest to jedno z największych tego typu wydarzeń w Europie. Podczas czterech dni odbędą się liczne projekcje filmowe, panele dyskusyjne i wykłady. Zorganizowane zostaną także dwa konkursy: na najlepszy scenariusz polskiego filmu, który miał swoją premierę w 2023 r. oraz na koncepcję serialu. Zgłoszenia do konkursu można nadsyłać do 21 stycznia. Do wygrania jest łącznie 70 tys. zł!

W Konkursie Głównym jury będzie miało za zadanie obejrzeć 10 filmów z 2023 r. i pośród nich wybrać zwycięski tytuł. Trzy produkcje zostaną wyświetlone podczas trwania festiwalu. Autorowi, autorce lub osobie autorskiej, która stworzy najlepszy scenariusz, będzie przysługiwała nagroda wysokości 50 tysięcy złotych. W kapitule konkursu znaleźli się między innymi: Jerzy Skolimowski, Maciej Ślesicki, Małgorzata Szumowska, Paweł Maślona, Paweł Maślona oraz Kaja Krawczyk-Wnuk. Wśród wybitnych scenarzystów zasiądzie również Olga Tokarczuk! Polska noblistka wraz z całą kapitułą będzie musiała wybrać najlepszy scenariusz.

O pracy scenarzysty członek Kapituły Konkursu Głównego – Paweł Maślona – mówi tak:

– Od dawna mam poczucie, że zawód scenarzysty to najbardziej niewdzięczny i najbardziej niedoceniany zawód w branży filmowej. Tym bardziej cieszę się, że powstał festiwal poświęcony w całości sztuce scenariopisarstwa i jest mi niezmiernie miło, że mogę być jego częścią.

Drugi z konkursów, czyli ten na koncept scenariusza serialowego, jest kierowany nie tylko do wziętych scenarzystów, ale też do amatorów. Każdy może więc nadesłać swój pomysł na scenariusz. Tym razem w jury zasiądą: Juliusz Machulski, Michał Kwieciński oraz Kalina Alabrudzińska. Wygrany otrzyma 20 tysięcy złotych.

Chociaż Tokarczuk głównie jest kojarzona ze swojej twórczości literackiej, to jednak jest ona także współtwórczynią scenariusza do produkcji "Pokot". Film został stworzony na bazie jej książki, zatytułowanej "Prowadź swój pług przez kości umarłych".'''
sample_combined_text = sample_title + " " + sample_text
sample_combined_text=clean_text(sample_combined_text)
# Przekształcenie tekstu do formatu TF-IDF
sample_tfidf = loaded_vectorizer.transform([sample_combined_text])

# Przewidywanie
predicted = loaded_model.predict(sample_tfidf)
predicted_proba = loaded_model.predict_proba(sample_tfidf)

# Jeśli używasz LabelEncoder, możesz odwrócić transformację, aby uzyskać oryginalną etykietę
predicted_label = label_encoder.inverse_transform(predicted)[0]

print("Przewidywana etykieta:", predicted_label)



#%% logistyczna regresja word2vec
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

df = combined_df
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

df['do PBL'] = df['do PBL'].astype(str)
label_encoder = LabelEncoder()
df['do PBL'] = label_encoder.fit_transform(df['do PBL'])

# Tokenizacja tekstu
tokenized_text = [text.split() for text in df['combined_text']]

# Trenowanie modelu Word2Vec
word2vec_model = Word2Vec(tokenized_text, vector_size=1000, window=5, min_count=1, workers=8)

# Funkcja do przekształcania tekstu w wektor
def document_vector(doc):
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    return np.mean(word2vec_model.wv[doc], axis=0) if doc else np.zeros(word2vec_model.vector_size)

# Przekształcenie tekstu w wektory
X = np.array([document_vector(text.split()) for text in df['combined_text']])
y = df['do PBL'].values

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu regresji logistycznej
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
predictions = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Zapisanie modelu regresji logistycznej
joblib.dump(model, 'logistic_regression_model.pkl')

# Opcjonalnie, zapisz również model Word2Vec
joblib.dump(word2vec_model, 'word2vec_model.pkl')

#%%Drzewo decyzyjne TFidf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = combined_df
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

df['do PBL'] = df['do PBL'].astype(str)
label_encoder = LabelEncoder()
df['do PBL'] = label_encoder.fit_transform(df['do PBL'])

X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['do PBL'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Trenowanie modelu drzewa decyzyjnego
model = DecisionTreeClassifier()

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
#%% Drzewo decyzyjne Word2vec
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Załóżmy, że 'combined_df' to Twoja ramka danych
df = combined_df

# Połącz tytuł i tekst artykułu w jednym polu
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

# Przygotowanie etykiet
df['do PBL'] = df['do PBL'].astype(str)
label_encoder = LabelEncoder()
df['do PBL'] = label_encoder.fit_transform(df['do PBL'])

# Tokenizacja tekstu
tokenized_text = [text.split() for text in df['combined_text']]

# Trenowanie modelu Word2Vec
word2vec_model = Word2Vec(tokenized_text, vector_size=1000, window=5, min_count=1, workers=4)

# Funkcja do przekształcania tekstu w wektor
def document_vector(doc):
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    return np.mean(word2vec_model.wv[doc], axis=0) if doc else np.zeros(word2vec_model.vector_size)

# Przekształcenie tekstu w wektory
X = np.array([document_vector(text.split()) for text in df['combined_text']])

# Etykiety
y = df['do PBL'].values

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trenowanie modelu drzewa decyzyjnego
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
predictions = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

#%% DO COLABU

# Importowanie niezbędnych bibliotek
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

# Definicja funkcji do wczytywania i łączenia danych z plików JSON i Excel
def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "do PBL", "hasła przedmiotowe"]):
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

    # Znalezienie indeksu ostatniego 'True' w kolumnie 'do PBL', gdzie 'hasła przedmiotowe' jest wypełnione
    last_true_filled_index = merged_df[(merged_df['do PBL'] == True) & (merged_df['hasła przedmiotowe'].notna())].index[-1]

    # Ograniczenie DataFrame do wierszy do ostatniego 'True' włącznie, gdzie 'hasła przedmiotowe' jest wypełnione
    merged_df = merged_df.loc[:last_true_filled_index]
    merged_df = merged_df.reset_index(drop=True)

    # Ograniczenie do wybranych kolumn
    selected_columns = merged_df[selected_columns_list]

    return selected_columns

# Wczytanie danych za pomocą zdefiniowanej funkcji
df1 = load_and_merge_data(json_file_path, excel_file_path)

# Przygotowanie środowiska do przetwarzania języka naturalnego
# Załaduj polski model językowy
nlp = spacy.load('pl_core_news_sm')

# Pobierz polskie stop words
stop_words = nlp.Defaults.stop_words

# Definicja funkcji do czyszczenia tekstu
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

# Przygotowanie danych do modelowania
df = df1
# Połącz tytuł i tekst artykułu w jednym polu
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']
df['combined_text'] = df['combined_text'].apply(clean_text)

# Kodowanie etykiet
df['do PBL'] = df['do PBL'].astype(str)
label_encoder = LabelEncoder()
df['do PBL'] = label_encoder.fit_transform(df['do PBL'])

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['do PBL'], test_size=0.2, random_state=42)

# Przekształcenie tekstu na reprezentację TF-IDF
vectorizer = TfidfVectorizer(max_features=50000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Trenowanie modelu regresji logistycznej
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predykcja na zbiorze testowym
predictions = model.predict(X_test_tfidf)

# Ocena modelu
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Zapisanie modelu i narzędzi
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder_logistic_reg.pkl')



