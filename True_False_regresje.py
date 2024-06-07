# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 08:41:30 2024

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

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "do PBL", "hasła przedmiotowe", 'Link']):
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
df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15])#,df16,df17,df18,df19,df20,df21,df22,df23,df24,df25], ignore_index=True)


nlp = spacy.load('pl_core_news_lg')

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
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']
df['combined_text'] = df['combined_text'].apply(clean_text)

# Kodowanie etykiet
df['do PBL'] = df['do PBL'].astype(str)
unique_values = df['do PBL'].unique()
print(f"Unique values in 'do PBL' after filtering: {unique_values}")
filtered_data= df[df['do PBL']=='1.0'] #['Link']
df['do PBL'] = df['do PBL'].map({'0.0': "False", '1.0': "True", 'True': "True", 'False': "False"})

# Usuwanie wierszy z nan
df = df.dropna(subset=['do PBL'])

unique_values = df['do PBL'].unique()
print(f"Unique values in 'do PBL' after explicit mapping and dropna: {unique_values}")

label_encoder = LabelEncoder()
df['do PBL'] = label_encoder.fit_transform(df['do PBL'])
X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['do PBL'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=70000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Trenowanie modelu regresji logistycznej
model = LogisticRegression(max_iter=10000)
model.fit(X_train_tfidf, y_train)

from sklearn.tree import DecisionTreeClassifier
# Inicjalizacja modelu drzewa decyzyjnego z określonymi parametrami
model = DecisionTreeClassifier(
    # max_depth określa maksymalną głębokość drzewa.
    # Im większa wartość, tym bardziej złożone może być drzewo, ale zwiększa się też ryzyko przeuczenia.
    max_depth=10,

    # min_samples_split definiuje minimalną liczbę próbek, jaką musi mieć węzeł, aby mógł być podzielony.
    # To ograniczenie pomaga zapobiegać zbyt drobnemu podziałowi, który mógłby prowadzić do przeuczenia.
    min_samples_split=5,

    # min_samples_leaf określa minimalną liczbę próbek, jakie muszą znaleźć się w liściu drzewa.
    # Większe wartości mogą zwiększać ogólną zdolność modelu do generalizacji, zmniejszając ryzyko przeuczenia.
    min_samples_leaf=4,

    # max_features określa maksymalną liczbę cech rozpatrywanych przy poszukiwaniu najlepszego podziału.
    # Ustawienie 'sqrt' oznacza, że w każdym podziale będzie brana pod uwagę pierwiastkowa liczba wszystkich cech.
    max_features='sqrt',

    # random_state zapewnia reprodukowalność wyników poprzez kontrolowanie losowości algorytmu.
    # Ustawienie konkretnej wartości (np. 42) pozwala na uzyskanie tych samych wyników przy każdym uruchomieniu.
    random_state=42
)
model.fit(X_train_tfidf, y_train)

#random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=10000,  # Liczba drzew w lesie
    max_depth=10000,      # Maksymalna głębokość każdego drzewa
    min_samples_split=5,  # Minimalna liczba próbek wymagana do podziału węzła
    min_samples_leaf=4,  # Minimalna liczba próbek wymagana w liściu węzła
    max_features='sqrt', # Liczba cech rozpatrywanych przy poszukiwaniu najlepszego podziału
    random_state=42     # Zapewnia reprodukowalność wyników
)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)

# Ocena modelu
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Zapisanie modelu i narzędzi
joblib.dump(model, 'RandomForest_model.pkl')
joblib.dump(vectorizer, 'tfidf_RandomForest.pkl')
joblib.dump(label_encoder, 'label_encoder_RandomForest.pkl')

#ładowanie modelu
loaded_model = joblib.load('RandomForest_model.pkl')

# Wczytanie TfidfVectorizer, jeśli został zapisany
loaded_vectorizer = joblib.load('tfidf_RandomForest.pkl')
#ładowanie label encoder
label_encoder = joblib.load('label_encoder_RandomForest.pkl')
print(label_encoder.classes_)
#  Działanie modelu :
sample_title =''
sample_text = '''J.R.R. Tolkien i czarodziej Gandalf otrzymali swoje ulice na warszawskim Mordorze. Na warszawskim Służewcu, w części nazywanej potocznie „Mordorem”, dwie ulice otrzymają nazwy kojarzące się z „Władcą Pierścieni”. Jak informuje Urząd m.st. Warszawy, patronem jednej z ulic został brytyjski pisarz J.R.R. Tolkien, drugiej – jeden z bohaterów jego książek, czarodziej Gandalf.

Mordor to żartobliwa nazwa, która przylgnęła do skupiska biurowców na Służewcu w Warszawie. Określenie to oddawało specyfikę tego miejsca zarówno pod względem zatłoczenia i wynikających z niego problemów komunikacyjnych, jak i z uwagi na charakter pracy w znajdujących się tam korporacjach, żartobliwie porównywanej z pracą orków w świecie stworzonym przez Tolkiena.

Kiedy na prężnie rozwijającym się Służewcu zaczęto budowę nowych odcinków dróg, początkowo planowano nazwać je Pirytowa i Tytanowa. W końcu jednak władze Mokotowa po konsultacjach z mieszkańcami stwierdziły, że niepisana tradycja zobowiązuje i skoro na rejon ten mówi się Mordor, to warto nawiązać do „Władcy Pierścieni”. Jak podawała w czerwcu „Gazeta Wyborcza”, wśród pomysłów, które rozważano, pojawił się m.in. Sauron. Ostatecznie wybrano Gandalfa, czyli dobrego czarodzieja, oraz jego twórcę J.R.R. Tolkiena. Co ciekawe, Zespół Nazewnictwa Miejskiego początkowo negatywnie zaopiniował obie nazwy, tłumacząc, że pierwotne nazwy Pirytowa i Tytanowa nawiązywały do sąsiednich dróg. Przychylono się jednak do propozycji, które spodobały się mieszkańcom.

W czwartek 17 listopada podczas sesji rady m.st. Warszawy ostatecznie przyjęto uchwały w sprawie nadania nowych nazw. Droga biegnąca od ulicy Suwak w kierunku wschodnim będzie od tej pory ulicą J.R.R. Tolkiena, natomiast droga biegnąca od ulicy Konstruktorskiej w kierunku północnym – ulicą Gandalfa. Obie ulice położone są w stosunku do siebie prostopadle.'''
sample_combined_text = sample_text
sample_combined_text=clean_text(sample_combined_text)
# Przekształcenie tekstu do formatu TF-IDF
sample_tfidf = loaded_vectorizer.transform([sample_combined_text])

# Przewidywanie
predicted = loaded_model.predict(sample_tfidf)
predicted_proba = loaded_model.predict_proba(sample_tfidf)

# Jeśli używasz LabelEncoder, możesz odwrócić transformację, aby uzyskać oryginalną etykietę
predicted_label = label_encoder.inverse_transform(predicted)[0]

print("Przewidywana etykieta:", predicted_label)
import pandas as pd
import json

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link'):
    # Spróbuj wczytać dane z pliku JSON z kodowaniem UTF-8, a jeśli się nie uda, użyj kodowania 'latin-1'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_content = file.read()
            if not json_content.strip():
                raise ValueError("Plik JSON jest pusty.")
            json_data = json.loads(json_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        try:
            with open(json_file_path, 'r', encoding='latin-1') as file:
                json_content = file.read()
                if not json_content.strip():
                    raise ValueError("Plik JSON jest pusty.")
                json_data = json.loads(json_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Nie udało się odczytać pliku JSON: {e}")
    
    df_json = pd.DataFrame(json_data)
    
    # Sprawdzenie, czy kolumny 'Link' i 'Tekst artykułu' istnieją w pliku JSON
    if 'Link' not in df_json.columns or 'Tekst artykułu' not in df_json.columns:
        raise ValueError("Brak wymaganych kolumn w pliku JSON.")
    
    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']].astype(str)
    
    # Wczytanie danych z pliku Excel
    df_excel = pd.read_excel(excel_file_path)
    
    # Połączenie DataFrame'ów, zachowując wszystkie wiersze i kolumny z pliku Excel oraz pełny tekst z JSONa
    merged_df = pd.merge(df_excel, df_json, on=common_column, how="left")
    
    # Usunięcie pustych wierszy
    merged_df.dropna(subset=['Link', 'Tekst artykułu'], inplace=True)
    
    return merged_df

# Przykładowe użycie
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/domagala2024-02-08.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/domagala_2024-02-08.xlsx'
df = load_and_merge_data(json_file_path, excel_file_path)
# Spróbuj wczytać dane z pliku JSON z różnymi kodowaniami


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
df = df[df['Tytuł artykułu'].apply(lambda x: isinstance(x, str))]
df = df[df['Tekst artykułu'].apply(lambda x: isinstance(x, str))]

# Połącz kolumny i zastosuj funkcję clean_text
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']
df['combined_text'] = df['combined_text'].apply(clean_text)
df['do PBL'] = df['do PBL'].astype(str)
unique_values = df['do PBL'].unique()
print(f"Unique values in 'do PBL' after filtering: {unique_values}")
label_encoder = LabelEncoder()
df['do PBL'] = label_encoder.fit_transform(df['do PBL'])

from tqdm import tqdm
import numpy as np

import joblib
df = df.head(100)
predictions = []
predictions_proba = []

for text in tqdm(df['combined_text']):
    cleaned_text = clean_text(text)
    tfidf_vector = loaded_vectorizer.transform([cleaned_text])
    pred = loaded_model.predict(tfidf_vector)
    pred_proba = loaded_model.predict_proba(tfidf_vector)
    
    # Przechowywanie wartości przewidywanych i prawdopodobieństw
    predictions.append(pred[0])
    predictions_proba.append(np.max(pred_proba))

# Dodaj kolumnę 'predictions' do DataFrame
df['predictions'] = predictions
df['predictions_proba'] = predictions_proba

# Porównanie wartości
df['comparison'] = np.where(df['do PBL'] == df['predictions'], 'Match', 'Mismatch')
#%%
from tqdm import tqdm
import pandas as pd

# Ładowanie danych
df = pd.read_excel('C:/Users/dariu/nowe_przewidywania.xlsx')

# Czyszczenie i transformacja tekstu
df['cleaned_text'] = df['combined_text'].apply(clean_text)
tfidf_matrix = loaded_vectorizer.transform(df['cleaned_text'])

# Predykcje dla całego zbioru
predictions = loaded_model.predict(tfidf_matrix)
predictions_proba = loaded_model.predict_proba(tfidf_matrix)

# Dekodowanie etykiet, jeśli używasz LabelEncoder
predictions_labels = label_encoder.inverse_transform(predictions)


# Zapisywanie wyników
df['predicted_label_logistic_R'] = predictions_labels
df['prediction_LogisticR'] = predictions_proba.max(axis=1) 
# Opcjonalnie zapisz zmodyfikowany DataFrame do nowego pliku Excel
df.to_excel('nowe_przewidywania04_06.xlsx', index=False)