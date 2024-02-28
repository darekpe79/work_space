# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import spacy
import pytextrank

from spacy.lang.pl.stop_words import STOP_WORDS  # Importowanie polskich stopwords

nlp = spacy.load("pl_core_news_lg")

# Konwersja listy stopwords do formatu oczekiwanego przez PyTextRank
stopwords_config = {"word": list(STOP_WORDS)}

# Dodawanie PyTextRank do potoku spaCy z konfiguracją zawierającą polskie stopwords
nlp.add_pipe("textrank", config={"stopwords": stopwords_config})

# example text
text = """Zmarł Feliks W. Kres, autor „Księgi Całości”, prekursor polskiej fantasy </tytuł> Jeden z najważniejszych polskich autorów fantasty, autor popularnego cyklu „Księga Całości”, Feliks W. Kres nie żyje. Pisarz zmarł w wieku 56 lat po walce z chorobą. Wiadomość o jego śmierci przekazało wydawnictwo Fabryka Słów.

„Nie ma słów właściwych, by przekazać taką informację. Po zaciętej walce z chorobą odszedł od nas Feliks W. Kres. Kresie, pozostawiłeś po sobie pustkę, której jeszcze nie umiemy ogarnąć. Pytania, na które już nigdy nie poznamy odpowiedzi. Myśli, którymi już się z tobą nie podzielimy. Teraz jednak, przede wszystkim, jesteśmy całym sercem z twoją żoną i najbliższymi” – napisało wydawnictwo na swoim profilu na Facebooku.

Urodzony 1 czerwca 1966 roku Feliks W. Kres (właściwie nazywał się Witold Chmielecki) był prekursorem polskiej fantasy. Zadebiutował już 1983 roku utworem „Mag” nadesłanym na konkurs w „Fantastyce”. Było to pierwsze w Polsce opowiadanie napisane w tym gatunku, które ukazało się na łamach profesjonalnego czasopisma. Trzy lata później miał już złożoną do druku swoją książkę. Na księgarskich półkach zobaczył ją dopiero na początku lat 90.

„Dla krajowego twórcy to był najgorszy czas z możliwych. Nadrabialiśmy półwiecze zaległości – my, Polacy. Drukowano wtedy wszystko, co tylko było opatrzone anglosaskim nazwiskiem. Jako czytelnik byłem wniebowzięty, wreszcie miałem pełne półki i wybór. Natomiast jako autor – bo jeszcze nie pisarz – nosiłem maszynopisy od wydawcy do wydawcy. Radzono mi – to dzisiaj brzmi anegdotycznie – bym sygnował książki Felix V. Craes albo w ogóle – bo ja wiem… – John Brown. Byle nie rodzimo brzmiącym nazwiskiem” – wspominał w książce „Galeria dla dorosłych”, dodając, że wolał jednak pozostać przy swoim polsko brzmiącym pseudonimie.

W latach 1991-1996 Kres co roku wydawał książkę. Jak sam przyznawał, chyba żaden inny polski autor-fantasta nie mógł tego o sobie powiedzieć. „Nie jestem dziś szczególnie dumny z nagród, które wówczas zebrałem, bo też jaka była konkurencja?… Raz i drugi, pamiętam, napotkano poważne trudności ze znalezieniem pięciu dzieł rodzimej produkcji, niezbędnych do tego, by w ogóle przeprowadzić konkurs – mówiąc inaczej: cokolwiek napisano, a otarło się o fantastykę, automatycznie dostawało nominację do nagrody, choćby nawet autor nie znał gramatyki i ortografii” – pisał z wrodzonym dystansem.

Bez względu na to, jaki poziom prezentowała konkurencja, Feliks W. Kres nie przeszedł do historii polskiej fantasy tylko dlatego, że był jej prekursorem. W przypadku autora nie było mowy o literackiej nieporadności. Stworzony przez niego cykl „Księga Całości”, rozgrywający się w świecie Szereru, gdzie żyją tylko trzy rozumne gatunki – ludzie, koty i sępy – uchodzi za prawdziwą klasykę gatunku i jedno z najważniejszych dzieł polskiej fantasy. Pozycji tej nie zagroził nawet trochę młodszy i znacznie popularniejszy „Wiedźmin” Andrzeja Sapkowskiego.

W 2011 roku Kres oświadczył, że rezygnuje z dalszego pisania. Jego powrót po blisko dekadzie, ogłoszony przez wydawnictwo Fabryka Słów, był w polskim świecie fantastyki dużym wydarzeniem. Najpierw wznowiono w poprawionej wersji tomy „Księgi Całości”, które ukazały się przed laty. Pierwszy nowy tom, „Najstarsza z Potęg”, zapowiedziany jest na listopad. Autor nie dożył jego publikacji."""

# load a spaCy model, depending on language, scale, etc.

# add PyTextRank to the spaCy pipeline

doc = nlp(text)

# examine the top-ranked phrases in the document
for phrase in doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    print(phrase.chunks)
from icecream import ic    
for phrase in doc._.phrases:
    ic(phrase)
    
for phrase in doc._.phrases:
    ic(phrase.rank, phrase.count, phrase.text)
    ic(phrase.chunks)
import spacy
from spacy.lang.pl.stop_words import STOP_WORDS  # Importowanie polskich stopwords

# Ładowanie modelu spaCy
nlp = spacy.load("pl_core_news_lg")

# Dodawanie PyTextRank do potoku spaCy
nlp.add_pipe("textrank", config={"stopwords": {"word": list(STOP_WORDS)}})

# Tekst do analizy


# Przetwarzanie tekstu
doc = nlp(text)

# Wydobywanie i wyświetlanie kluczowych fraz
print("Kluczowe frazy:")
for phrase in doc._.phrases:
    dir(phrase)
    
    print(f"{phrase.text} (Rank: {phrase.rank}, Count: {phrase.count})")
    

# Wydobywanie i wyświetlanie nazwanych encji
print("\nNazwane encje:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
#%%
 
nlp = spacy.load("pl_core_news_lg")    
nlp.add_pipe("positionrank")

doc = nlp(text)

for phrase in doc._.phrases:
    ic(phrase) 
    
nlp = spacy.load("pl_core_news_lg")    
nlp.add_pipe("topicrank")

doc = nlp(text)

for phrase in doc._.phrases:
    ic(phrase)  
    
    
    
    
from topicrankpy import extractinformation as t





from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModel

# Załaduj tokenizer i model HERBERT
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModel.from_pretrained("allegro/herbert-base-cased")

kw_model = KeyBERT(model=model, tokenizer=tokenizer)

# Przykładowy tekst
text = "Tutaj wprowadź swój polski tekst..."

# Ekstrakcja słów kluczowych
keywords = kw_model.extract_keywords(text, keyphrase_length=1, stop_words='polish')

print(keywords)



from transformers import pipeline
generator = pipeline("translation", model="sdadas/flan-t5-base-translator-en-pl")
sentence = "A team of astronomers discovered an extraordinary planet in the constellation of Virgo."
print(generator(sentence, max_length=512))
# [{'translation_text': 'Zespół astronomów odkrył niezwykłą planetę w gwiazdozbiorze Panny.'}]



#%% New approach clean text

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


max_length = 512  # Maksymalna długość fragmentu tekstu, dostosuj w zależności od modelu i ograniczeń pamięci
fragments = textwrap.wrap(text, max_length, break_long_words=False, replace_whitespace=False)

# Analiza każdego fragmentu osobno
ner_results = []
for fragment in fragments:
    ner_results.extend(nlp1(fragment))
    
    
def combine_tokens(ner_results):
    combined_entities = []
    current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0, "start": None, "end": None}
    previous_end_of_word = False

    for token in ner_results:
        # Sprawdzamy, czy bieżący token jest końcem słowa
        end_of_word = "</w>" in token['word']
        cleaned_word = token['word'].replace("</w>", "")

        # Rozpoczęcie nowej jednostki
        if token['entity'].startswith('B-'):
            if current_entity['word']:
                # Obliczamy średnią ocenę dla skompletowanej jednostki
                current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
                combined_entities.append(current_entity)
            current_entity = {"word": cleaned_word, "type": token['entity'][2:], "score_sum": token['score'],
                              "token_count": 1, "start": token['start'], "end": token['end']}
            previous_end_of_word = end_of_word
        # Kontynuacja obecnej jednostki
        elif token['entity'].startswith('I-') and current_entity['type'] == token['entity'][2:]:
            # Dodajemy spację przed bieżącym tokenem, jeśli poprzedni token był końcem słowa
            if previous_end_of_word:
                current_entity['word'] += " " + cleaned_word
            else:
                current_entity['word'] += cleaned_word
            current_entity['end'] = token['end']
            current_entity['score_sum'] += token['score']
            current_entity['token_count'] += 1
            previous_end_of_word = end_of_word

    # Dodajemy ostatnią jednostkę, jeśli istnieje
    if current_entity['word']:
        current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
        combined_entities.append(current_entity)

    return combined_entities




# Wywołanie funkcji
combined_entities = combine_tokens(ner_results)
combined_entities_selected=[]
for entity in combined_entities:
    if entity['score']>=0.90:
        combined_entities_selected.append(entity)
        
        
entities = [entity['word'] for entity in combined_entities_selected]  
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
    representative = sorted(group, key=lambda x: len(x))[0]
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



# Zastępowanie bytów w tekście ich reprezentantami
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
        entity_counts[entity] += 1

# Sortowanie i wyświetlanie wyników
sorted_entity_counts = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)

print("Wystąpienia bytów z uwzględnieniem ważności tytułu:")
for entity, count in sorted_entity_counts:
    print(f"{entity}: {count}")  
choosen_ents=[]
for ent in sorted_entity_counts:
    if ent[1]>3:
        choosen_ents.append(ent)
        


entity_lemmatization_dict



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





nlp = spacy.load("pl_core_news_lg")    
nlp.add_pipe("positionrank")

doc = nlp(updated_text)

for phrase in doc._.phrases:
    ic(phrase) 
    
    
nlp.add_pipe("textrank", config={"stopwords": {"word": list(STOP_WORDS)}})

# Tekst do analizy


# Przetwarzanie tekstu
doc = nlp(updated_text)

# Wydobywanie i wyświetlanie kluczowych fraz
print("Kluczowe frazy:")
for phrase in doc._.phrases:
    print(f"{phrase.text} (Rank: {phrase.rank}, Count: {phrase.count})")
    
#%%  
    
import morfeusz2

# Utworzenie instancji Morfeusza
morf = morfeusz2.Morfeusz()

# Tekst do analizy
text = "Jaś miał kota"

# Analiza morfoskładniowa tekstu
analysis = morf.analyse(text)

# Wyświetlenie wyników analizy
for interpretation in analysis:
    print(interpretation)
    
    
    
text = "Jaś miał kota"

# Dzielenie tekstu na słowa
words = text.split()

# Analiza każdego słowa
for word in words:
    analysis = morf.analyse(word)
    for interpretation in analysis:
        start_node, end_node, interp = interpretation
        form, lemma, tag, _, _ = interp
        print(f"Słowo: {form}, Lemat: {lemma}, Tag: {tag}")
        
        
        
lematy = []

# Przykładowe wyniki analizy
wyniki = [
    ("Jaś", "Jasia", "subst:pl:gen:f"),
    ("Jaś", "jaś", "subst:sg:nom:m2"),
    ("Jaś", "Jaś:Sf", "subst:sg.pl:nom.gen.dat.acc.inst.loc.voc:f"),
    ("Jaś", "Jaś:Sm1", "subst:sg:nom:m1"),
    ("miał", "miał", "subst:sg:nom.acc:m3"),
    ("miał", "mieć", "praet:sg:m1.m2.m3:imperf"),
    ("kota", "kota", "subst:sg:nom:f"),
    ("kota", "kot:Sm1", "subst:sg:gen.acc:m1"),
    ("kota", "kot:Sm2", "subst:sg:gen.acc:m2")
]

for slowo, lemat, tag in wyniki:
    if "praet" in tag or "subst" in tag:  # Filtrujemy interesujące nas tagi
        if ":" in lemat:  # Usuwamy dodatkowe informacje z lematu
            lemat = lemat.split(":")[0]
        lematy.append((slowo, lemat))

print(set(lematy))  # Usuwamy duplikaty