# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:44:46 2024

@author: dariu
"""

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

# Załadowanie modelu i tokienizatora
model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Utworzenie pipeline NER
nlp1 = pipeline("ner", model=model, tokenizer=tokenizer)

text ="""CO JEST ZA ZAKRĘTEM? (WOJCIECH CHMIELARZ: 'WYRWA') </tytuł>„Rzecz nazwawszy po imieniu, wszystko swój posiada cennik, nie podołasz zamówieniu to załatwi je twój zmiennik”. Pamiętacie piosenkę Gintrowskiego do serialu „Zmiennicy”? Bohater nowej powieści Wojciecha Chmielarza boleśnie przekona się, że za wszystko trzeba będzie jednak zapłacić, nawet jeśli ma się skromne oczekiwania wobec życia.Istnieją pisarze, na których książki czekam z utęsknieniem, ich zapowiedź odnotowuję notatką w telefonie, a po zakupie odcinam się od całego świata, żeby pochłaniać litery i z każdą stroną odczuwać żal, że już niedługo koniec. Tak właśnie jest z powieściami Wojciecha Chmielarza, nawet jeśli lekko mnie rozczarowują (jak ubiegłoroczna „Rana”), to i tak autor ma u mnie duży kredyt zaufania. Nie byłam fanką wolty, którą uczynił Chmielarz, przechodząc od kryminałów do thrillerów, a właściwie i to określenie gatunkowe w stosunku do „Wyrwy”, zważywszy na królującą w księgarniach produkcję literacką, nie jest do końca odpowiednie. Czytałam ostatnio wystarczająco dużo powieści, które udawały, że są oryginalne, brutalne, a były w istocie kolejną wariacją domestic noir ze straumatyzowaną bohaterka, lękającą się wyjścia z domu, listonosza i sąsiadów. Wyhodowałam więc w sobie apetyt na odmianę.Bohater „Wyrwy” – Maciej Tomski, przynależy do warszawskiej klasy średniej, jak sam o sobie mówi, przy czym ta średniość jest rozumiana nie tylko socjologicznie, ale najzupełniej dosłownie – średnia droga zawodowa (bo trudno nazwać ją karierą), mieszkanie, wakacje all inclusive, samochód i ambicje. Takiego everymanowi przydarza się coś, czego się lękamy – tajemnica. Otrzymuje wiadomość o śmierci żony, ale dlaczego zamiast w drodze spod Krakowa znajdowała się na Mazurach? Co ukrywała i dlaczego bohater tak długo dawał się zwodzić? Czy miała powody, by obawiać się o swoje życie? Jeśli gdzieś już słyszeliście taki wstęp, nie mylicie się – to ulubiony schemat Harlana Cobena lub Michela Bussi, znany nie tylko z książek, ale i seriali czy filmów. Na szczęście dla czytelników łatwy do opisania schemat schodzi na dalszy plan i tak, jak to było w „Ranie”, przestaje być ważne, kto i jak zamordował. Typowe dla kryminału pytanie o motyw zastąpi opowieść, w jaki sposób niezwiązane wydarzenia nakładają się i eksplodują w postaci tragedii oraz płynące z niej rozważania, kto jest winien, czy możliwe było inne zakończenie. Na swój użytek podobny wątek nazywam determinizmem przestępstwa – to rozważania, co wpłynęło na bieg wydarzeń – jak w klasycznej powieści Patricii Highsmith „Znajomi z pociągu”. Maciej szuka odpowiedzi na pytanie, co ukrywała Beata – dziennikarka prasy kobiecej, niespełniona pisarka, i rekonstruuje moment, w którym ich drogi zaczęły się rozchodzić, a małżeństwo zamieniło w rutynę pytań, kto odbierze dzieci ze szkoły. Czytelnikowi w pewnym wieku, z kredytem w banku i problemami szkolno-przedszkolnymi potomstwa łatwo się utożsamić z Maciejem lub przynajmniej odnaleźć znajomego, więc kibicuje mu w szukaniu odpowiedzi. Ale Maciej zaskoczy czytelnika, w jaki sposób – tego już nie odważę się napisać.W serialu „Fosse/Verdon” jeden z producentów tłumaczy Bobowi Fosse, że jego pomysły się nie sprawdzą, bo główny bohater „Całego tego zgiełku” nie przechodzi wewnętrznej przemiany, pozostaje taki sam, a ludzie tego nie lubią, bo oczekują, że bohater coś zrozumie i się zmieni. Ten łopatologiczny wywód w istocie ujmuje sedno – oczekujemy od bohatera przemiany, a nie unoszenia się na powierzchni wydarzeń, liczymy, że wraz ze zmianą przestrzeni nastąpi zmiana postaci. I Chmielarz nam, czytelnikom, daje taką przemianę – Maciej jest zmuszony zrezygnować z bierności, a może zrzucić maskę. Autor czyni z Macieja narratora o ograniczonej wiedzy, musi więc dodać protagonistę, który będzie w stanie uzupełnić luki w fabule. Niestety, ta druga postać – nie chcę zdradzać płci ani roli – nie jest już skonstruowana w tak udany sposób, szeleści papierem. Można ją w sumie polubić, ale brak jej wiarygodności. O ile jako czytelnik poczuję się jak pocący się w garniturze Maciej, o tyle z tą drugą postacią nie mam na to szans. Podejrzewam, że jej rola miała być pierwotnie mniejsza, ale z przyczyn formalnych rozrosła się do funkcji drugiego narratora.Niezaprzeczalną zaletą książki jest jej nieodkładalność, raczej nie warto planować chwilowej lektury, po której wraca się do pracy. Otóż, nie da się odłożyć „Wyrwy” ot, tak. „Coś być musi do cholery za zakrętem” – ponownie zacytuję Gintrowskiego. To, co mnie drażniło podczas lektury, to nadużywanie frazy „zamordowałem żonę, on/ona zamordował mi żonę” itd. itp. Drogi autorze – to, co intryguje za pierwszym razem, za drugim, trzecim, za kolejnym już irytuje. Efekciarskie. Bo czy to morderstwo w Orient Expressie, że mamy wielu morderców? Może było to pisane z myślą o scenariuszu – nasz bohater, o twarzy podobnej jakby do nikogo (Maciej Stuhr?) zaciska zęby i cedzi – „Aż dotrze do niego, że zamordował mi żonę”. Cięcie. I potem kolejna postać wchodzi do gry i kolejna kwestia o zamordowaniu żony.Chmielarz będzie czytany i będzie zdobywał nowych czytelników, bo to po prostu solidna firma. Czy jednak przejście od klasycznych, ponurych, mocnych powieści z cyklu jest udane? „Żmijowisko” powaliło mnie na ziemię zakończeniem, „Rana” rozczarowała przejściem w stronę thrillera psychologicznego, po lekturze „Wyrwy” ośmielę się zauważyć – król wrócił. """
text="CO JEST ZA ZAKRĘTEM? (WOJCIECH CHMIELARZ: 'WYRWA') </tytuł>„Rzecz nazwawszy po imieniu, wszystko swój posiada cennik, nie podołasz zamówieniu to załatwi je twój zmiennik”. Pamiętacie piosenkę Gintrowskiego do serialu „Zmiennicy”? Bohater nowej powieści Wojciecha Chmielarza boleśnie przekona się, że za wszystko trzeba będzie jednak zapłacić, nawet jeśli ma się skromne oczekiwania wobec życia.Istnieją pisarze, na których książki czekam z utęsknieniem, ich zapowiedź odnotowuję notatką w telefonie, a po zakupie"

# max_length = 2000  # Maksymalna długość fragmentu tekstu, dostosuj w zależności od modelu i ograniczeń pamięci
# fragments = textwrap.wrap(text, max_length, break_long_words=False, replace_whitespace=False)
proba=nlp1(text)

tokens = tokenizer.tokenize(text)

# Dzielenie tokenów na fragmenty o maksymalnej długości 'max_tokens'
max_tokens = 512  # Przykładowe ograniczenie modelu
token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

# Przekształcenie fragmentów tokenów z powrotem na listę tekstów
text_fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]

# Analiza każdego fragmentu osobno
ner_results = []
for fragment in text_fragments:
    ner_results.extend(nlp1(fragment))
    break
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
    
    
# def combine_tokens(ner_results):
#     combined_entities = []
#     current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0, "start": None, "end": None}
#     previous_end_of_word = False

#     for token in ner_results:
#         # Sprawdzamy, czy bieżący token jest końcem słowa
#         end_of_word = "</w>" in token['word']
#         cleaned_word = token['word'].replace("</w>", "")

#         # Rozpoczęcie nowej jednostki
#         if token['entity'].startswith('B-'):
#             if current_entity['word']:
#                 # Obliczamy średnią ocenę dla skompletowanej jednostki
#                 current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
#                 combined_entities.append(current_entity)
#             current_entity = {"word": cleaned_word, "type": token['entity'][2:], "score_sum": token['score'],
#                               "token_count": 1, "start": token['start'], "end": token['end']}
#             previous_end_of_word = end_of_word
#         # Kontynuacja obecnej jednostki
#         elif token['entity'].startswith('I-') and current_entity['type'] == token['entity'][2:]:
#             # Dodajemy spację przed bieżącym tokenem, jeśli poprzedni token był końcem słowa
#             if previous_end_of_word:
#                 current_entity['word'] += " " + cleaned_word
#             else:
#                 current_entity['word'] += cleaned_word
#             current_entity['end'] = token['end']
#             current_entity['score_sum'] += token['score']
#             current_entity['token_count'] += 1
#             previous_end_of_word = end_of_word

#     # Dodajemy ostatnią jednostkę, jeśli istnieje
#     if current_entity['word']:
#         current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
#         combined_entities.append(current_entity)

#     return combined_entities




# Wywołanie funkcji
combined_entities = combine_tokens(ner_results)
combined_entities_selected=[]
for entity in combined_entities:
    if entity['score']>=0.90:
        combined_entities_selected.append(entity)
        
        
        
#entities = [entity['word'] for entity in combined_entities_selected]  
entities = [(entity['word'],entity['type']) for entity in combined_entities_selected]  
#entities_list=entities_type
# Ładowanie spaCy
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



# Ładowanie modelu spaCy z wektorami


# Przekształcanie bytów na obiekty doc spaCy
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

entity_groups = group_similar_entities(lemmatized_entities, threshold)

# Zmiana tutaj: wybieranie najkrótszego bytu w grupie jako reprezentanta
representatives = [sorted(group, key=lambda x: len(x))[0] for group in entity_groups]

entity_to_representative_map = {}
for group in entity_groups:
    representative = sorted(group, key=lambda x: (len(x), x))[0]
    for entity in group:
        entity_to_representative_map[entity] = representative

for entity, representative in entity_to_representative_map.items():
    print(f'"{entity}" zostanie zastąpione przez "{representative}"')

def replace_entities_with_representatives(text, map):
    # Sortowanie kluczy według długości tekstu malejąco, aby najpierw zastąpić dłuższe frazy
    sorted_entities = sorted(map.keys(), key=len, reverse=True)
    
    for entity in sorted_entities:
        representative = map[entity]
        # Zastąpienie klucza (bytu) jego wartością (reprezentantem) w tekście
        text = text.replace(entity, representative)
    
    return text


updated_text = replace_entities_with_representatives(lemmatized_text, entity_to_representative_map)
# MOJA METODA
list_of_new_entities=(list(entity_to_representative_map.values()))
unique(list_of_new_entities)
#lemmatized_doc = nlp(updated_text)

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

print("Wystąpienia bytów z uwzględnieniem ważności tytułu:")
for entity, count in sorted_entity_counts:
    print(f"{entity}: {count}")  
choosen_ents=[]
for ent in sorted_entity_counts:
    if ent[1]>3:
        choosen_ents.append(ent)
        
        
        
        
import requests
import re

def preprocess_text(text):
    # Usuwanie dat z tekstu, np. "Emma Goldman, 1869-1940" staje się "Emma Goldman"
    return re.sub(r',?\s*\d{4}(-\d{4})?', '', text)

def check_viaf_with_fuzzy_match(entity_name, threshold=90):
    base_url = "http://viaf.org/viaf/AutoSuggest"
    query_params = {'query': entity_name}
    best_match = None
    best_score = 0
    
    try:
        response = requests.get(base_url, params=query_params)
        response.raise_for_status()
        data = response.json()
        if data and 'result' in data:
            for result in data['result'][:10]:
                # Sprawdzamy najpierw z oryginalnym terminem
                original_term = result.get('term')
                score_with_date = fuzz.token_sort_ratio(entity_name, original_term)
                if score_with_date > best_score and score_with_date >= threshold:
                    best_score = score_with_date
                    best_match = result
                
                # Następnie sprawdzamy po usunięciu dat
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



for entity in choosen_ents:
    if entity[0] in entity_lemmatization_dict:
        for original_entity, entity_type in entity_lemmatization_dict[entity[0]]:
            if entity_type == 'PER':
                result = check_viaf_with_fuzzy_match(original_entity)
                if result:
                    print(f"{original_entity} found in VIAF: {result}")
                else:
                    
                    print(f"{original_entity} not found in VIAF, checking alternative...")
                    
                    result = check_viaf_with_fuzzy_match(entity[0])
                    
            elif entity_type == 'ORG':
                result = check_viaf_with_fuzzy_match(original_entity)
                if result:
                    print(f"{original_entity} found in VIAF: {result}")
                else:
                    
                    print(f"{original_entity} not found in VIAF, checking alternative...")
                    
                    result = check_viaf_with_fuzzy_match(entity[0])
                # Tutaj logika dla organizacji, może podobna do osób
                pass
            
            
            elif entity_type == 'LOC':
                result = check_geonames(original_entity)
                if result:
                    print(f"{original_entity} found in GeoNames: {result}")
                else:
                    print(f"{original_entity} not found in GeoNames.")

# Przykład użycia
entity_name = "jan kowalski"
viaf_url = check_viaf(entity_name)
if viaf_url:
    print(f"Found in VIAF: {viaf_url}")
else:
    print(f"{entity_name} not found in VIAF.")





# Przykład użycia
entity_name = "Emma Goldman"
viaf_url, match_score = check_viaf_with_fuzzy_match(entity_name)
if viaf_url:
    print(f"Found in VIAF: {viaf_url} with score: {match_score}")
else:
    print(f"{entity_name} not found in VIAF or no match above threshold.")
    
    
    
# def check_viaf(entity_name):
#     base_url = "http://viaf.org/viaf/AutoSuggest"
#     query_params = {'query': entity_name}
#     try:
#         response = requests.get(base_url, params=query_params)
#         response.raise_for_status()  # Sprawdza czy odpowiedź serwera jest błędem (np. 404, 500)
#         data = response.json()
#         if data and 'result' in data and data['result']:
#             # Załóżmy, że interesuje nas pierwszy wynik
#             first_result = data['result'][0]
#             viaf_id = first_result.get('viafid')
#             if viaf_id:
#                 return f"http://viaf.org/viaf/{viaf_id}"
#     except requests.RequestException as e:
#         print(f"Error querying VIAF: {e}")
#     return None