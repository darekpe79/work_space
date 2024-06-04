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
joblib.dump(model, 'LogisticRegression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizerLogisticRegression.pkl')
joblib.dump(label_encoder, 'label_encoder_LogisticRegression.pkl')

#ładowanie modelu
loaded_model = joblib.load('C:/Users/dariu/logistic_regression/LogisticRegression_model.pkl')

# Wczytanie TfidfVectorizer, jeśli został zapisany
loaded_vectorizer = joblib.load('C:/Users/dariu/logistic_regression/tfidf_vectorizerLogisticRegression.pkl')
#ładowanie label encoder
label_encoder = joblib.load('C:/Users/dariu/logistic_regression/label_encoder_LogisticRegression.pkl')
print(label_encoder.classes_)
#  Działanie modelu :
sample_title =''
sample_text = '''SZCZYT WSZYSTKIEGO (UGO BARDI: 'WYDOBYCIE. JAK POSZUKIWANIE BOGACTW MINERALNYCH PUSTOSZY NASZĄ PLANETĘ')

A A A
Książka Ugo Bardiego „Wydobycie. Jak poszukiwanie bogactw mineralnych pustoszy naszą planetę” powstała jako kolejny raport Klubu Rzymskiego – stowarzyszenia czy też think tanku założonego w 1968 roku i zajmującego się badaniem globalnych problemów współczesnego świata, obejmujących zwłaszcza kwestie środowiskowe. Najsłynniejszą jak dotąd publikacją Klubu są „Granice wzrostu” z 1972 roku, diagnozujące (pisząc w największym skrócie), że utrzymanie modelu ekonomicznego opartego na wykładniczym wzroście gospodarczym jest niemożliwe ze względu na skończoność zasobów naturalnych Ziemi.

„Wydobycie” właściwie powtarza tę diagnozę, aktualizuje obliczenia autorów „Granic wzrostu” i pozytywnie weryfikuje większość ich prognoz. Jednocześnie nazywa ich przepowiednie „kasandrycznymi” w pełnym znaczeniu tego słowa: były to (i nadal pozostają) złowróżbne wieści, których nikt nie chciał wysłuchać, które jednak okazały się boleśnie prawdziwe. „Wydobycie” nie tylko je potwierdza, ale także umieszcza je w nowych kontekstach, jakimi są rozwój technologii uzależnionych od metali ziem rzadkich, postępujące zmiany klimatu czy propozycja ustanowienia nowej epoki geologicznej, czyli antropocenu – „epoki człowieka”, który jako gatunek uzyskał tak wielką sprawczość, że przekształca całą planetę na własną modłę i na własną zgubę. W tym sensie książkę Bardiego nazwać by można „Granicami wzrostu 2.0”.

Nieustannie powraca tu twierdzenie, że zapasy wszystkich surowców kluczowych dla funkcjonowania cywilizacji znajdują się na wyczerpaniu. Twierdzenie to poparte jest oczywiście wieloma danymi, wyliczeniami i modelami. Jak podpowiada tytuł książki, główny nacisk położony został na geologię, ale „Wydobycie” najwłaściwiej byłoby chyba nazwać publikacją nie tyle inter- czy nawet trans-, ile postdyscyplinarną. Łączy ona bowiem nauki ścisłe (geologia, chemia, klimatologia) z naukami społeczno-humanistycznymi (ekonomia, politologia, historia, kulturoznawstwo), zasypując przepaść pomiędzy „dwiema kulturami”, o których po II wojnie światowej pisał Charles Percy Snow (1999).

Trzy główne problemy, jakimi zajmuje się Bardi, to wyczerpywanie się „bogactw mineralnych” planety, wpływ kurczenia się zasobów na gospodarkę oraz efekty uboczne „wielkiego eksperymentu górniczego” (s. 249) przedsiębranego przez ludzkość: zanieczyszczenie środowiska, niszczenie naturalnych habitatów istot żywych oraz, najbardziej obecnie palący, kryzys klimatyczny. Książka podzielona jest na trzy części: w pierwszej autor przedstawia historię (albo raczej geohistorię) wytwarzania się złóż mineralnych Ziemi oraz korzystania z owych „darów Gai” (s. 38) przez ludzkość od czasów rewolucji neolitycznej aż po współczesność; część druga omawia nasze dzisiejsze bolączki związane nie tylko z wyczerpywaniem się tzw. konwencjonalnych złóż kopalin, ale także z nieuchronnym (z punktu widzenia nauk fizycznych), lecz uporczywie ignorowanym przez światowy paradygmat ekonomiczno-polityczny wpływem górnictwa na planetę i jej zdolność do podtrzymywania warunków sprzyjających życiu w postaci, w jakiej je znamy (w tym życiu ludzkiemu); wreszcie część trzecia projektuje możliwe scenariusze bliższej i dalszej przyszłości naznaczonej efektami wykopania i przetworzenia (spalenia, zużycia, porzucenia) przez nas olbrzymiej ilości „bogactw mineralnych”.

Lekcja, jaka płynie z pierwszego rozdziału, brzmi mniej więcej tak: zasoby mineralne, kształtowane na przestrzeni ewolucji geologicznej Ziemi przez setki tysięcy, a nawet miliony lat, są przez ludzkość wyczerpywane w zastraszającym tempie, uniemożliwiającym nawet częściowe ich odtworzenie. Bardi nazywa owe zasoby „darami Gai”, nawiązując do spopularyzowanej przez Jamesa Lovelocka koncepcji Gai. Głosi ona, że Ziemia jest „żywą” planetą, na której odbywają się liczne procesy stabilizujące ziemskie systemy i podtrzymujące warunki sprzyjające życiu. Pośród nich najważniejszy pozostaje geologiczny obieg węgla działający na zasadzie „termostatu”: gdy na Ziemi robi się zbyt gorąco, uruchamia on mechanizmy ochładzające, gdy zaś klimat za bardzo się ochładza, zostaje on „podgrzany” na przykład poprzez emisje CO2 z wulkanów. Sposób, w jaki ludzkość korzysta z „darów Gai”, jest jednak przez Bardiego utożsamiany z wojną wypowiedzianą planecie – taką, w której nie bierze się jeńców, ale też taką, której nie sposób wygrać. Tytułowe wydobycie odbywa się bowiem zbyt szybko, na zbyt dużą skalę i ze zbyt wysokimi (i wciąż rosnącymi) kosztami środowiskowo-klimatycznymi. Krótko mówiąc, Gaja „nie nadąża” za ludzkością, co stanowi jeden z argumentów na rzecz ustanowienia epoki antropocenu (zob. np. Steffen, Crutzen, McNeill 2007).

Przechodzimy tu już do drugiej części książki. Jej lekcja brzmi mniej więcej tak: w miarę wyczerpywania się tzw. konwencjonalnych źródeł (węgla, ropy naftowej, rzadkich metali itp.) wydobycie staje się coraz trudniejsze, pochłania coraz więcej energii i coraz bardziej szkodzi środowisku. Nie chodzi tylko o to, że trzeba kopać głębiej, ale także o kuszące z perspektywy krótkoterminowych zysków, jednak katastrofalne na dłuższą metę pomysły czerpania zasobów ze źródeł niekonwencjonalnych i niekonwencjonalnymi metodami (na przykład uzyskiwanie ropy z piasków bitumicznych, szczelinowanie, wiercenie w dnie oceanicznym, eksploatacja Arktyki). Bardi wielokrotnie podkreśla, że problemem jest tu nie tyle ilość dostępnych jeszcze minerałów (w pewnym sensie nie mylą się ekonomiści mówiący, że wszystkiego mamy pod dostatkiem), ile coraz gorsza jakość dostępnych złóż i związane z nią coraz większe koszty energetyczne wydobycia. Stwierdzenie to powraca przy okazji opisywania każdego pojedynczego przypadku – nie tylko zasobów ropy czy węgla, ale także na przykład kluczowego dla sektora rolniczego fosforu, głównych metali ery przemysłowej: niklu i cynku, czy wykorzystywanych jako katalizatory w technologii silnikowej platynowców.

Omawiając ekonomiczno-polityczne pomysły na zaradzenie temu postępującemu kryzysowi, Bardi posługuje się bardzo instruktywną metaforą „biegu Czerwonej Królowej”, znanego z „Przygód Alicji w Krainie Czarów” Lewisa Carrolla. Jak wiadomo, trzeba było biec w nim jak najszybciej tylko po to, by pozostać w miejscu, co „oznaczało wiele pracy, która do niczego nie prowadziła” (s. 251). Tak właśnie autor postrzega pomysły w rodzaju eksploatowania złóż niekonwencjonalnych, drobnych korekt systemu na przykład w postaci rozwoju elektromobilności opartej na silnikach napędzanych litem (metalem samym w sobie raczej rzadkim, którego wydobycie jest kosztowne zarówno ekonomicznie, jak i ekologicznie) czy kuriozalnych strategii w typie „premii za złomowanie” (wynagradzanie użytkowników za oddawanie na złom wciąż sprawnych samochodów w celu odzyskania wykorzystanych do ich produkcji metali, a „przy okazji” zachęcenia do kupna nowych modeli). Krótko mówiąc: wyścigu Czerwonej Królowej nie da się wygrać.

Dlatego lekcja, jaka płynie z trzeciej części książki, brzmi: jedyne, co może nas ocalić, to radykalna zmiana paradygmatu ekonomiczno-politycznego. Nie jest to po prostu kwestia „wyboru”: załamanie tak czy owak nastąpi i tylko od nas zależy, czy zwycięży polityka business as usual, która pozostawi nas kompletnie nieprzygotowanymi na koniec świata, jaki znamy, czy też zaakceptujemy zmianę i spróbujemy się na nią przygotować, co z pewnością będzie bolesne, ale przynajmniej da nam nadzieję na przetrwanie. Tu Bardi wymienia trzy możliwe podejścia: substytucję (zastępowanie rzadkich minerałów powszechniejszymi; przejście z paliw kopalnych na odnawialne źródła energii), recykling (odzyskiwanie surowców; porzucenie ekonomii uprzywilejowującej produkty jednorazowego użytku; idee zero waste) oraz radykalne zmniejszenie konsumpcji (odejście od praktyk postarzania produktów; postawienie na większą wydajność systemu; ruchy typu degrowth). Jak jednak podkreśla autor, każde z tych podejść ma swoje ograniczenia i choć są one krokiem w dobrym kierunku, to sytuacja, w jakiej się znaleźliśmy, wymaga o wiele poważniejszych działań. Aby je wdrożyć, rozpocząć trzeba od przepracowania żałoby po obecnym modelu życia, którego nie da się już utrzymać. Bardi odwołuje się tu do pięciu etapów żałoby opisanych przez Elisabeth Kübler-Ross (Kessler, Kübler-Ross 2005): zaprzeczenia, gniewu, prób targowania się, depresji i akceptacji. Podczas gdy wielu z nas znajduje się już na którymś z kolejnych stadiów, „opinia publiczna pozostaje pogrążona w stadium pierwszym: zaprzeczeniu” (s. 287).

***

„Wydobycie”, jak wiele książek podejmujących temat wpływu działalności człowieka na planetę, nie jest lekturą przyjemną. Jest to jednak lektura bardzo przystępna „retorycznie”: choć operuje wieloma danymi chemicznymi, geologicznymi, fizycznymi itp., pozostaje nie tylko zrozumiała, ale i fascynująca dla kogoś o wykształceniu humanistycznym. Opowiada bowiem niezwykle ciekawą naturalno-kulturową (geo)historię minerałów, metali i paliw kopalnych oraz ich wpływu na rozwój i upadek ludzkich cywilizacji. Weryfikuje przy tym wiele obiegowych opinii i reinterpretuje wiele ustalonych prawd historycznych (w pamięci pozostają tezy o kontroli hiszpańskich kopalń złota i srebra jako źródle rozkwitu imperium rzymskiego czy o braku dostępu do kluczowych zasobów ropy naftowej jako głównej przyczynie porażki ZSRR w zimnej wojnie).

Książkę Bardiego warto (a w zasadzie trzeba) czytać w kontekście diagnoz antropocenu, a także alternatywnej propozycji kapitałocenu. Sam autor afirmatywnie przywołuje koncepcję antropocenu, dostarczając wielu przemawiających za nią argumentów. Podejście to wydaje się jak najbardziej uzasadnione z punktu widzenia geologii: człowiek jako gatunek rzeczywiście przekroczył sprawczość pewnych pozostałych sił naturalnych i doprowadził do radykalnego przekształcenia planetarnych warunków życia, co – jak podkreślają orędowniczki i orędownicy antropocenu – spowodowało przekroczenie co najmniej kilku z dziewięciu „granic planetarnych” (Rockström i in. 2009).

Jak jednak zauważają rzeczniczki i rzecznicy kapitałocenu, z punktu widzenia nauk społecznych taka narracja zaciemnia pewne istotne kwestie. Od słabości koncepcji antropocenu nie jest wolne także „Wydobycie”. Przede wszystkim zbyt często operuje ono pojęciem ludzkości jako takiej, tak jakby zniszczenie planety stanowiło „przeznaczenie” naszego gatunku, tak jakby uprzywilejowanie krótkoterminowych zysków było wrodzoną wadą „natury” ludzkiej, a nie cechą konkretnego systemu ekonomiczno-politycznego, i tak jakby antropocen istniał, odkąd tylko na Ziemi pojawili się ludzie. Tymczasem, mimo że ludzkość „kopała” od zawsze, to naruszenie systemów ziemskich nastąpiło całkiem niedawno – prawdopodobnie po rewolucji przemysłowej, a najpewniej w okresie „wielkiego przyspieszenia” po II wojnie światowej. Nie spowodowała go też ludzkość „jako taka”, ale wielcy aktorzy ekonomiczno-polityczni. Choć Bardi zwraca na to wszystko uwagę i jednoznacznie stwierdza, że kapitalizm oparty na fetyszu wzrostu gospodarczego jest nie do utrzymania, to jednocześnie nie do końca potrafi wyjść poza ten paradygmat. Zdradza to retoryka, jaką się posługuje, pisząc o „bogactwach naturalnych” czy „kapitale naturalnym”, tak jakby minerały i paliwa kopalne od początku stanowiły tylko „zasób” czekający na kapitalizację.

Wydaje się, że sama ta „rynkowa” retoryka jest elementem problemu, którego nie należy lekceważyć. Stanowi ona bowiem element dualizmu Cywilizacja-Natura, analizowanego (i krytykowanego) w obrębie dyskursu kapitałocenu (zob. np. Moore 2016). Dualizm ów z jednej strony sankcjonuje „niezależność” Cywilizacji od Natury, z drugiej zaś – przekształca tę ostatnią w dostarczycielkę „tanich zasobów” i „taniej pracy” (zob. Patel, Moore 2019). Wspominam o tym, gdyż „Wydobycie” nierzadko zbliża się i do tej perspektywy, choćby wtedy, gdy opisuje gigantyczny problem odpadów, zwłaszcza zawierających metale ciężkie czy radioaktywne lub metale ziem rzadkich, bardzo trudne do pozyskania, a mimo to beztrosko porzucane na światowych wysypiskach śmieci wraz z wymienianymi co sezon lub dwa sprzętami elektrycznymi, w których się je wykorzystuje. W tym sensie Bardi, podobnie jak badaczki i badacze forsujący koncepcję kapitałocenu, obwieszcza koniec „taniej natury”. Zbliża się do ich propozycji także wtedy, gdy zauważa, że człowiek nie tylko wytwarza, czy też przekształca, sieci życia, ale i sam jest przez nie wytwarzany i przekształcany (zob. np. Moore 2015). Tak można czytać fascynujące historie wykorzystywania w starożytności siedmiu znanych wówczas metali, które doprowadziło do rozkwitu poszczególnych kultur i imperiów, czy wspomnianą już rolę dostępu do strategicznych złóż ropy jako głównego czynnika umożliwiającego Stanom Zjednoczonym zwycięstwo w zimnej wojnie z ZSRR.

Podsumowując, „Wydobycie” warto czytać w obu kontekstach – antropocenu i kapitałocenu – jednocześnie. Każdy z nich daje bowiem lekturze coś innego i do każdego z nich książka ta wnosi nieco inny wkład. I właśnie takie „całościowe” podejście wydaje się niezbędne, aby adekwatnie odpowiedzieć na kryzys ekologiczno-klimatyczny, nie zaś formułować cząstkowe rozwiązania dyktowane przez interesy kapitału, rozumu instrumentalnego czy partykularyzmów narodowych.'''
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

df = pd.read_excel('C:/Users/dariu/results_.xlsx')
from tqdm import tqdm
predictions = []
predictions_proba = []
for text in tqdm(df['combined_text']):
    cleaned_text = clean_text(text)
    tfidf_vector = loaded_vectorizer.transform([cleaned_text])
    pred = loaded_model.predict(tfidf_vector)
    pred_proba = loaded_model.predict_proba(tfidf_vector)
    
    # Jeśli używasz LabelEncoder, odkoduj etykietę
    if 'label_encoder' in locals():  # Sprawdza, czy 'label_encoder' jest zdefiniowany
        pred_label = label_encoder.inverse_transform(pred)[0]
    else:
        pred_label = pred[0]
    
    predictions.append(pred_label)
    predictions_proba.append(pred_proba[0][pred])  # Zapisuje prawdopodobieństwo dla przewidzianej klasy

# Dodanie wyników predykcji do DataFrame
df['predicted_label'] = predictions
df['prediction_probability'] = predictions_proba
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
df.to_excel('nowe_przewidywania.xlsx', index=False)