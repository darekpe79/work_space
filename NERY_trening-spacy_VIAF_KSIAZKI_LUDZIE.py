# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 08:57:40 2024

@author: dariu
"""

import requests
import json
import pandas as pd
import json

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "byt 1", "zewnętrzny identyfikator bytu 1", "Tytuł spektaklu"]):
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

    # Znalezienie indeksu ostatniego wystąpienia 'zewnętrzny identyfikator bytu 1'
    if 'zewnętrzny identyfikator bytu 1' in merged_df.columns:
        last_id_index = merged_df[merged_df['zewnętrzny identyfikator bytu 1'].notna()].index[-1]
        merged_df = merged_df.loc[:last_id_index]
    else:
        print("Brak kolumny 'zewnętrzny identyfikator bytu 1' w DataFrame.")

    merged_df = merged_df.reset_index(drop=True)

    # Ograniczenie do wybranych kolumn
    if set(selected_columns_list).issubset(merged_df.columns):
        selected_columns = merged_df[selected_columns_list]
    else:
        print("Nie wszystkie wybrane kolumny są dostępne w DataFrame.")
        selected_columns = merged_df

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
combined_df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], ignore_index=True)
combined_df = pd.concat([df1, df2], ignore_index=True)


#%%proba oznaczania jeden df2
df2['combined_text'] = df2['Tytuł artykułu'] + " " + df2['Tekst artykułu']
import pandas as pd
import spacy
import re

# Załadowanie modelu języka polskiego
nlp = spacy.load("pl_core_news_lg")

# Funkcja do oznaczania słów z tytułów spektakli w tekście
def mark_titles(text, title):
    # Escapowanie specjalnych znaków w tytule
    title_pattern = re.escape(title) + r"(?![\w-])"  # Aby uniknąć dopasowania w środku słowa, dodajemy negative lookahead
    # Oznaczanie tytułu w tekście znacznikami
    marked_text = re.sub(title_pattern, r"[SPEKTAKL]\g<0>[/SPEKTAKL]", text, flags=re.IGNORECASE)
    return marked_text


df2['Tytuł spektaklu'] = df2['Tytuł spektaklu'].fillna('')
df2['marked_text'] = df2.apply(lambda row: mark_titles(row['combined_text'], row['Tytuł spektaklu']), axis=1)
def prepare_data_for_ner(text):
    pattern = r"\[SPEKTAKL\](.*?)\[/SPEKTAKL\]"
    entities = []
    current_pos = 0
    clean_text = ""
    last_end = 0

    for match in re.finditer(pattern, text):
        start, end = match.span()
        clean_text += text[last_end:start]  # Dodaj tekst przed znacznikiem
        start_entity = len(clean_text)
        entity_text = match.group(1)
        clean_text += entity_text  # Dodaj tekst encji bez znaczników
        end_entity = len(clean_text)
        entities.append((start_entity, end_entity, "SPEKTAKL"))
        last_end = end  # Zaktualizuj pozycję ostatniego znalezionego końca znacznika

    clean_text += text[last_end:]  # Dodaj pozostały tekst po ostatnim znaczniku

    return clean_text, {"entities": entities}

df2['spacy_marked'] = df2['marked_text'].apply(prepare_data_for_ner)

import spacy

# Załaduj istniejący model lub utwórz nowy
nlp = spacy.load("pl_core_news_lg")  # Załaduj istniejący model dla języka polskiego
# nlp = spacy.blank("pl")  # Lub utwórz nowy pusty model dla języka polskiego, jeśli wolisz

if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe("ner")

# Dodaj nową etykietę do pipeline'u NER
ner.add_label("SPEKTAKL")

# Przygotowanie danych treningowych z DataFrame
TRAIN_DATA = []

# Iteruj przez wiersze i dodaj każdy jako przykład trenujący
for row in df2['spacy_marked']:
    text, annotation = row
    entities = annotation['entities']
    TRAIN_DATA.append((text, {'entities': entities}))

# Rozpoczęcie procesu trenowania
from spacy.training import Example
n_iter = 1000
learn_rate = 0.001
drop = 0.5

# Wznowienie procesu trenowania z dostosowanymi parametrami
optimizer = nlp.resume_training()
for i in range(n_iter):
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=drop, sgd=optimizer)

text = '''Zaczęło się od Rozmów bez słów. Później przyszedł czas na Rozmowy agonistyczne. A na zakończenie Parlament rzeczy. Widzowie zaangażowali się w działania teatralne, przełamali własne bariery i stanęli naprzeciw działań performatywnych. To oni podczas tego krótkie czasu stali się badaczami teatru w dziedzinie, które trudno nazwać sztuką
/fot. M. Zakrzewski/

Parlament rzeczy (Parliament of Things) to jedna z rozmów, która składa się na cykl Building Conversation zainicjowany przez Lotte van den Berg na Malta Festival Poznań 2016. Jest to dość nietypowe działanie, którego głównym celem jest pobudzenie widza do dialogu i rozważań na temat własnych zachowań oraz postrzegania ludzi i świata. Działanie to składa się z wielu faz. Jego początek to krótki spacer przenoszący widzów z obecnie widzianej przestrzeni Placu Wolności w Poznaniu do niewielkiego i pustego pokoju, w którym rozpocząć ma się dyskusja rzeczy. Zwykłych przedmiotów, pozornie ze sobą niepołączonych, które w trakcie wchodzenia w zadanie otrzymują poszczególne znaczenie, stają się symbolami danych zjawisk. W oparciu o książkę Brunona Latoura Nigdy nie byliśmy nowocześni (We Have Never Been Modern). Podstawowym zagadnieniem wypływającym z badań socjologa było zwrócenie wzroku w kierunku tego, co nieludzkie. Uświadomienie sobie, że nie tylko człowiek posiada moc sprawczą, że władzę ma ten, kto wpływa na otoczenie.

Dzieląc świat na ludzi i nie-ludzi (przedmioty, zwierzęta itp.) Latour wykazał, że zarówno jedna grupa jak i druga determinuje działania, a reakcje, jakie między nimi zachodzą nie mogą zostać oddzielone i analizowane osobno. Są nierozerwalną jednością. I to właśnie w myśl tej tezy uczestnicy Parlamentu rzeczy wybierając jakiś przedmiot (w tym przypadku było to ziarno) tworzyli wokół niego siatkę innych przedmiotów, które miały na nie wpływały bądź, na które wpływał on. Tak stworzył się krąg szesnastu rzeczy, które podczas dwóch sesji czterdziestopięciominutowych miały przemówić i podyskutować. Najtrudniejsze zadanie stało jednak przed uczestnikami, którzy musieli najpierw wyzbyć się swojej ludzkiej natury i wczuć w rzecz, którą sami wybrali, przemawiając jej głosem. Na czas rozmowy każdy był przedmiotem, który wybrał, lecz jeśli chciał, mógł w trakcie dyskusji go zmieniać i wypowiadać się także w imieniu nowo wybranej rzeczy.

Na początku każdy przedstawił się, kim jest i jakie jest jego zadanie względem przedmiotu głównego. Zaskakujące było to w jak szybki sposób uczestnicy wczuwając się w przedmioty próbowali przemawiać jego głosem, nie umniejszając mu, a wynosząc niekiedy na piedestał czyniąc go niezależnym i najważniejszym spośród wielu (tak np. stało się z elementem odpowiadającym za zjawisko zanieczyszczeń). Nawet ważniejszym od człowieka. Eksperyment trwał, a osoby coraz bardziej zacietrzewiały się w swoich racjach aż w końcu z perspektywy siebie zrozumiały, że nie są w stanie mówić inaczej niż swoim głosem. Przedmioty w ich rękach były tylko kolejnym przedłużeniem własnego bytu. Nikt jednak nie ośmielił się zaprzeczyć, że teoria Latoura nie jest właściwa. Ten model został potwierdzony jeszcze przed rozpoczęciem konwersacji, kiedy to tworząc sieć łączącą elementy z głównym przedmiotem uczestnicy dostrzegli, że żadnej z tych rzeczy nie da się oddzielić ani traktować gdzieś poza. Każdy z wybranych przedmiotów był ważny i bez niego wszystko traciłoby swój sens. Lecz wszystko to opierało się na ludzkim postrzeganiu.

Spacer powrotny i toczące się na nim dyskusje na temat eksperymentu zakończyły się wspólnym posiłkiem. Łączącym nie tylko dyskutujących, ale również ich działania performatywne z teatrem rzeczy. Rozmowa zainicjowana przez Lotte van der Berg stała się pewnego rodzaju dziełem sztuki wytworzonym przez widza. To on stanął w samym centrum pola gry i działając na dwóch płaszczyznach: aktor – widz, stwarzał teatr, którego był sam inicjatorem. To było świadome zmaganie się z formą, z działaniem z pogranicza sztuki, które odkryć należało w trakcie przedsięwzięcia artystyczno-badawczego. I choć uczestnicy zdali sobie sprawę, że ich działania zawsze będą grą wobec nie-ludzi, których działań nie dostrzegają to z większą uwagą będą starać się przyglądać elementom układanki, której są częścią. Eksperyment nie miał na celu zmienić ich myślenia, ale poprzez wczuwanie się w poszczególnie przedmioty pokazać, że całość materii jest ważna, że wszystko na siebie działa i każde odstępstwa od normy, choć czasami w niewielki stopniu, wpływają na zmiany całego wszechświata. A człowieka, jako pojedyncza jednostka pośród wielu, w końcu powinien to dostrzec. Być świadomy w jakim systemie żyje.'''
doc = nlp(text)
output_dir = "model_spektakl"

# Zapisanie modelu do dysku
nlp.to_disk(output_dir)
for ent in doc.ents:
    if ent.label_ == "SPEKTAKL":
        print(ent.text)
        

### Krok 1: Załaduj model z dysku


import spacy

# Załaduj wytrenowany model
model_path = "model_spektakl"
nlp = spacy.load(model_path)


### Krok 2: Użyj modelu do analizy tekstu


text_to_analyze = "Osiemnastka, reż. Natalia Fijewska-Zdanowska Pięciu bohaterów, którzy na widok publiczny wyciągną swoje problemy. Pięciu bohaterów, którzy zmuszeni zostaną do znalezienia przyczyn nieudanego życia. I niech nikogo nie zmyli tytuł tego spektaklu, bo nie o imprezie osiemnastkowej będziemy tutaj rozmawiać, a o bohaterach którzy robić będą dobrą minę do złej gry. Zabawa urodzinowa jest tylko pretekstem do tego, aby zamienić się w rozmowy o egzystencji i niespełnionych pragnieniach/fot. Rafał Latoszek/Dla jednych będzie to przewidywalny początek, dla innych dobre wprowadzenie do rozgrywającej się akcji. Bo zaczynamy zwykłą rozmową pomiędzy trzema parami, które spotykają się przypadkiem na wspólnej imprezie osiemnastkowej swoich dzieci. Jest miło i przyjemnie, dialogi bawią, nietrudno też zaśmiać się z tych wszystkich żartów sytuacyjnych które się zadziewają. Lecz trwające coraz dłużej rozmowy zaczynają gęstnieć, a w powietrzu unosi się dziwny zapach żalu i goryczy, który przeżywają bohaterowie. Akcja przyśpieszy kiedy okaże się, że młodzież która tak świetnie bawić się miała na zorganizowanej imprezie zniknęła, a pilnujący ich rodzice obiecali do rana nie wychodzić z domu, który mieści się gdzieś na skraju lasu. Powiało grozą - nie będę ukrywać. Pojawiła się też cała masa pytań, gdzie zniknęły dzieci, dlaczego rodzice nie mogą opuścić budynku, co wydarzy się dalej, kto zginie. Napięcie zaczyna narastać, a pytania zamiast znikać, tylko się mnożyły. Byłam dobrej myśli do tego co zadzieje się dalej, bo oprócz momentów grozy mieliśmy również dobry humor, który trzymał wysoki poziom.Aż tu nagle… Szok i niedowierzanie, bo spektakl zwolnił do maksimum, sprawy zaczęły zamiast się rozwiązywać nawarstwiać, pojawiło się całe mnóstwo wstawek o Janie Chrzcicielu i zapewne cytatów z Biblii, które ani nie są mi bliskie ani do tego dzieła nie pasowały. A szkoda, bo zamysł Natalii Fijewskiej-Zdanowskiej (scenariusz i reżysera), która głos oddaje ludziom dojrzałym z dość dużym już bagażem doświadczeń był naprawdę dobry. Bo co tu dużo mówić stykamy się z ludźmi, których marzenia pozostają w sferze zapomnienia, a najskrytsze pragnienia nigdy nie zostają wypowiedziane na głos. Reżyserka oddaje swój spektakl w ręce pokolenia niespełnionego, ludzi lat 70. i 80., które wychowało się na lęku i toksycznych relacjach. I to ich przeżycia powinny nas-widzów uwierać czy też wzbudzać jakąś refleksję. I zgodzę się, że na początku tak się dzieje, ale później akcja staje w miejscu problemy się nie rozwiązują, a dziejące się wydarzenia nagle znajdują ujście, bez większego ładu i składu.Mówiąc o spektaklu Osiemnastka nie można jednak zapomnieć o aktorach, którzy byli bardzo przekonujący i prawdziwi w swoich rolach. To było tak dobre połączenie różnych charakterów i osobowości, że momentami na scenie działy się rzeczy piękne. Na tyle, że nie sposób było nie oddać się w ręce bohaterów i dać im po prostu prowadzić się po niuansach tego dzieła. I na próżno dzielić tutaj obsadę na obóz lepiej i gorzej grający, bo to co pokazali na scenie mogło być tylko w tej pierwszej z grup. Naprawdę to był kawał solidnej pracy, doświadczenia które dobrze skomponowało się z opowiadaną historią postaci. I żałuję (bardzo), że koniec tego dzieła, mimo fantastycznej gry aktorskiej, był mało przekonywująco (i słaby). Ale wybrzmieć to musi głośno, bo jakby odrzucić to co złe, to dla Zuzanny Fijewskiej-Malesza, Agaty Kołodziejczyk, Dominiki Łakomskiej, Jarosława Boberka i Pawła Pabisiaka przyszła bym jeszcze raz chętnie na taką imprezę."

# Przetworzenie tekstu przez model
doc = nlp(text_to_analyze)

# Wyświetlenie znalezionych encji
for ent in doc.ents:
    print(ent.text, ent.label_)



text = "Osiemnastka, reż. Natalia Fijewska-Zdanowska Pięciu bohaterów, którzy na widok publiczny wyciągną swoje problemy. Pięciu bohaterów, którzy zmuszeni zostaną do znalezienia przyczyn nieudanego życia. I niech nikogo nie zmyli tytuł tego spektaklu, bo nie o imprezie osiemnastkowej będziemy tutaj rozmawiać, a o bohaterach którzy robić będą dobrą minę do złej gry. Zabawa urodzinowa jest tylko pretekstem do tego, aby zamienić się w rozmowy o egzystencji i niespełnionych pragnieniach/fot. Rafał Latoszek/Dla jednych będzie to przewidywalny początek, dla innych dobre wprowadzenie do rozgrywającej się akcji. Bo zaczynamy zwykłą rozmową pomiędzy trzema parami, które spotykają się przypadkiem na wspólnej imprezie osiemnastkowej swoich dzieci. Jest miło i przyjemnie, dialogi bawią, nietrudno też zaśmiać się z tych wszystkich żartów sytuacyjnych które się zadziewają. Lecz trwające coraz dłużej rozmowy zaczynają gęstnieć, a w powietrzu unosi się dziwny zapach żalu i goryczy, który przeżywają bohaterowie. Akcja przyśpieszy kiedy okaże się, że młodzież która tak świetnie bawić się miała na zorganizowanej imprezie zniknęła, a pilnujący ich rodzice obiecali do rana nie wychodzić z domu, który mieści się gdzieś na skraju lasu. Powiało grozą - nie będę ukrywać. Pojawiła się też cała masa pytań, gdzie zniknęły dzieci, dlaczego rodzice nie mogą opuścić budynku, co wydarzy się dalej, kto zginie. Napięcie zaczyna narastać, a pytania zamiast znikać, tylko się mnożyły. Byłam dobrej myśli do tego co zadzieje się dalej, bo oprócz momentów grozy mieliśmy również dobry humor, który trzymał wysoki poziom.Aż tu nagle… Szok i niedowierzanie, bo spektakl zwolnił do maksimum, sprawy zaczęły zamiast się rozwiązywać nawarstwiać, pojawiło się całe mnóstwo wstawek o Janie Chrzcicielu i zapewne cytatów z Biblii, które ani nie są mi bliskie ani do tego dzieła nie pasowały. A szkoda, bo zamysł Natalii Fijewskiej-Zdanowskiej (scenariusz i reżysera), która głos oddaje ludziom dojrzałym z dość dużym już bagażem doświadczeń był naprawdę dobry. Bo co tu dużo mówić stykamy się z ludźmi, których marzenia pozostają w sferze zapomnienia, a najskrytsze pragnienia nigdy nie zostają wypowiedziane na głos. Reżyserka oddaje swój spektakl w ręce pokolenia niespełnionego, ludzi lat 70. i 80., które wychowało się na lęku i toksycznych relacjach. I to ich przeżycia powinny nas-widzów uwierać czy też wzbudzać jakąś refleksję. I zgodzę się, że na początku tak się dzieje, ale później akcja staje w miejscu problemy się nie rozwiązują, a dziejące się wydarzenia nagle znajdują ujście, bez większego ładu i składu.Mówiąc o spektaklu Osiemnastka nie można jednak zapomnieć o aktorach, którzy byli bardzo przekonujący i prawdziwi w swoich rolach. To było tak dobre połączenie różnych charakterów i osobowości, że momentami na scenie działy się rzeczy piękne. Na tyle, że nie sposób było nie oddać się w ręce bohaterów i dać im po prostu prowadzić się po niuansach tego dzieła. I na próżno dzielić tutaj obsadę na obóz lepiej i gorzej grający, bo to co pokazali na scenie mogło być tylko w tej pierwszej z grup. Naprawdę to był kawał solidnej pracy, doświadczenia które dobrze skomponowało się z opowiadaną historią postaci. I żałuję (bardzo), że koniec tego dzieła, mimo fantastycznej gry aktorskiej, był mało przekonywująco (i słaby). Ale wybrzmieć to musi głośno, bo jakby odrzucić to co złe, to dla Zuzanny Fijewskiej-Malesza, Agaty Kołodziejczyk, Dominiki Łakomskiej, Jarosława Boberka i Pawła Pabisiaka przyszła bym jeszcze raz chętnie na taką imprezę."
word = "Osiemnastka"
text.startswith(word, 1)
for i in range(len(text)):
    print(i)
    if text.startswith(word, i):
        print(word, i)
indexes = [i for i in range(len(text)) if text.startswith(word, i)]
print("Numery indeksów słowa 'Osiemnastka':", indexes)
#%%
combined_df.dropna(subset=['byt 1'], inplace=True)
combined_df['combined_text'] = combined_df['Tytuł artykułu'] + " " + combined_df['Tekst artykułu']
combined_df.dropna(subset=['byt 1'], inplace=True)
filtered_df = combined_df[combined_df['byt 1'].str.contains('spektakl')]
filtered_df['zewnętrzny identyfikator bytu 1']
import requests
from bs4 import BeautifulSoup

def get_title_from_url(url):
    # Pobranie zawartości strony internetowej
    response = requests.get(url)
    
    # Sprawdzanie czy pobranie się udało
    if response.status_code == 200:
        # Parsowanie zawartości strony jako dokument HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Wyszukiwanie tagu tytułu
        title_tag = soup.find('title')
        
        # Wydobywanie tekstu z tagu tytułu i przycinanie do pierwszych dwukropków
        title_text = title_tag.text.split(':::')[0].strip()
        
        return title_text
    else:
        return None

# Dodanie nowej kolumny do DataFrame'u, zawierającej tytuły ze stron na podstawie linków
filtered_df['tytuł ze strony'] = filtered_df['zewnętrzny identyfikator bytu 1'].apply(get_title_from_url)

def get_viaf_info(viaf_id):
    url = f"https://viaf.org/viaf/{viaf_id}/viaf.json"
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        return "Błąd: Nie można pobrać danych"

# Przykład użycia
viaf_id = "95218067"
info = get_viaf_info(viaf_id)
atuor=info["titles"]["author"]["text"]
tytul=info["titles"]["expression"]["title"]
#ludzie
people_date=info['mainHeadings']['data']
for data in people_date:
    print(data['text'])
import difflib
import Levenshtein
import jellyfish

def compare_strings(str1, str2):
    # Używając SequenceMatcher z difflib
    seq_match = difflib.SequenceMatcher(None, str1, str2).ratio()
    print(f"SequenceMatcher similarity: {seq_match:.2f}")

    # Używając Levenshtein Distance
    levenshtein_distance = Levenshtein.distance(str1, str2)
    levenshtein_ratio = 1 - levenshtein_distance / max(len(str1), len(str2))
    print(f"Levenshtein similarity: {levenshtein_ratio:.2f}")

    # Używając Jaro-Winkler Distance
    jaro_winkler_similarity = jellyfish.jaro_winkler_similarity(str1, str2)
    print(f"Jaro-Winkler similarity: {jaro_winkler_similarity:.2f}")

# Przykładowe nazwy do porównania
str1 = "olszewski pawel"
str2 = "pawel olszewski"

compare_strings(str1, str2)

from fuzzywuzzy import fuzz

def compare_strings_fuzzy(str1, str2):
    # Proste dopasowanie
    simple_ratio = fuzz.ratio(str1, str2)
    print(f"Simple ratio: {simple_ratio}")

    # Dopasowanie z uwzględnieniem kolejności tokenów (słów)
    token_sort_ratio = fuzz.token_sort_ratio(str1, str2)
    print(f"Token sort ratio: {token_sort_ratio}")

    # Dopasowanie z uwzględnieniem unikalnych tokenów (słów) bez powtórzeń
    token_set_ratio = fuzz.token_set_ratio(str1, str2)
    print(f"Token set ratio: {token_set_ratio}")

# Przykładowe nazwy do porównania
str1 = "olszewski pawel"
str2 = "pawel olszewski"

compare_strings_fuzzy(str1, str2)

from fuzzywuzzy import process
import re

text = """
Wśród naszych współpracowników jest wielu utalentowanych ludzi. Na przykład, Paweł Jan Olszewski wniósł ogromny wkład w rozwój naszych projektów. Niezależnie od tego, czy mówimy o Olszewskim Pawle czy kogokolwiek innego, warto docenić ich ciężką pracę.
"""

reference = "olszewski pawel"

# Podział tekstu na zdania
sentences = re.split(r'\. |\?|!', text)

# Wyszukiwanie podobnych fraz w każdym zdaniu
for sentence in sentences:
    # Użycie extractOne do znalezienia najbardziej podobnej frazy w zdaniu
    best_match = process.extractOne(reference, [sentence], scorer=fuzz.token_sort_ratio)
    
    # Sprawdzenie, czy najlepsze dopasowanie ma akceptowalny poziom podobieństwa
    # Możesz dostosować próg podobieństwa zgodnie z potrzebami
    if best_match and best_match[1] > 60:  # Przykładowy próg podobieństwa
        print(f"Znaleziono dopasowanie: {best_match[0]} z podobieństwem: {best_match[1]}%")

# Uwaga: fuzz.token_sort_ratio ignoruje kolejność słów, co jest przydatne w tym przypadku
from difflib import SequenceMatcher

def find_similar_phrase(text, target_phrase, threshold=0.5):
    # Inicjalizacja obiektu SequenceMatcher
    seq_matcher = SequenceMatcher(None, text, target_phrase)

    # Znajdź wszystkie dopasowania powyżej określonego progu
    matches = seq_matcher.get_matching_blocks()

    # Filtruj dopasowania powyżej progu
    similar_matches = [match for match in matches if match.size / len(target_phrase) >= threshold]

    # Wypisz dopasowania
    for match in similar_matches:
        start = match.a
        end = start + match.size
        print(f"Dopasowanie: {text[start:end]} (Index: {start}-{end-1}, Podobieństwo: {match.size / len(target_phrase)})")

# Przykład użycia
full_text = "To jest pełny tekstyyy, w którym szukamy podobnych fraz."
target_phrase = "pełny telkst"

find_similar_phrase(full_text, target_phrase)


from fuzzywuzzy import fuzz

def find_similar_phrase_with_fuzzy(text, target_phrase, threshold=80):
    # Użyj funkcji token_sort_ratio z fuzzywuzzy
    similarity = fuzz.token_sort_ratio(text, target_phrase)

    # Sprawdź, czy podobieństwo jest powyżej progu
    if similarity >= threshold:
        print(f"Dopasowanie: {text} (Podobieństwo: {similarity}%)")
    else:
        print("Brak dopasowania powyżej progu.")

# Przykład użycia
full_text = "To jest pełny tekst, w którym szukamy podobnych fraz."
target_phrase = "pełny tekst"

find_similar_phrase_with_fuzzy(full_text, target_phrase)
