# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:19:00 2024

@author: dariu
"""
import pandas as pd
import json
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
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import logging
from transformers import AutoTokenizer, AutoModel, HerbertTokenizerFast
import joblib
from tqdm import tqdm
import numpy as np

text='''Cień świata. W scenicznej interpretacji „Nowych Aten” Maciej Gorczyński opowiada o sile wyobraźni i podstępach rozumu. " </tytuł>" Prapremiera spektaklu na podstawie pierwszej polskiej encyklopedii pióra księdza Benedykta Chmielowskiego była najjaśniejszym punktem VI Festiwalu Teatrów Błądzących w Gardzienicach.

Opisać systematykę stworzenia – oto zadanie heroiczne, godne świętego męża. Takie ambicje miał Chmielowski, autor pierwszej polskiej encyklopedii. Powstałe w połowie XVIII wieku dzieło nosiło tytuł „Nowe Ateny”. W porównaniu z dokonaniami działających w tym samym czasie francuskich encyklopedystów, wydaje się ono kuriozalne. Podjęcie trudu tworzenia encyklopedii motywowane było wiarą w rozum, motywacje miał więc Chmielowski podobne do Francuzów. W jego dziele znaleźć można jednak także praktyczne informacje na temat bazyliszków, smoków i czartów. Czy jest to powód wystarczający, aby księgę tę zostawić w biblioteczkach zaściankowej szlachty, a samemu wrócić do Denisa Diderota i jego kolegów? Nie może być zaskoczeniem, że teatr idzie w sukurs wyobraźni, co w wyreżyserowanych przez Macieja Gorczyńskiego „Nowych Atenach” cieszy szczególnie. Jest to przedstawienie bogate frenetycznymi momentami i adorujące imaginację, a w ten sposób opowiadające się po jednej ze stron konfliktu pomiędzy racjonalizmem i wyobraźnią. Kapłanowi z Firlejowa przydany w nim został nie lada sojusznik – William Blake.

Nad sceną zawieszono duże gumowe piłki i rzucono nań kosmiczne wizualizacje, dzięki czemu spełnić się mogły słowa zapomnianego poety: „Te gwiazdy to są kule, i na hakach wiszą”. Z tyłu sceny ściana-labirynt z delikatnego, zwiewnego materiału. Znikający w niej aktorzy zamieniali się w cienie. W ten sposób przypominali, że cała encyklopedyczna systematyzacja jest zaledwie próbą chwytania odbicia rzeczywistości, której promienie padają na powierzchnię wyobraźni. Właśnie dzięki wyobraźni ptaki/owady/motyle mogły przybrać postać delikatnych chustek fruwających na oddechach tańczących aktorów. Taniec był najważniejszą techniką teatralną w przedstawieniu, które za jego pomocą dotykało różnych tematów: od sfery duchowej jednostki, przez relacje międzyludzkie, narodowe stereotypy, aż po miejsce człowieka we wszechświecie.

Prapremiera przedstawienia odbyła się w Gardzienicach w ramach VI Festiwalu Teatrów Błądzących. Gorczyński z Ośrodkiem Praktyk Teatralnych współpracuje od wielu lat, był tam między innymi aktorem, ale i archiwistą. To wyjaśnia zaangażowanie przez niego estetyki, która została wypracowana przez zespół Włodzimierza Staniewskiego. Imponuje jej twórcze przekształcenie, możliwe zapewne dzięki temu, że aktorzy w „Nowych Atenach” to nie „ludzie gardzieniccy”, a studenci Wydziału Teatru Tańca krakowskiej PWST w Bytomiu. Angażując ich, reżyser przetacza świeżą krew w ramy starzejącej się konwencji teatralnej i ożywia tradycję, która w głównym zespole „Gardzienic” zdaje się obumierać i kostnieć. Młodzi artyści imponowali warsztatem. Skoncentrowani na dopracowanej choreografii – niekiedy popisowej (jak w przypadku Daniela Leżonia, grającego księdza i czarnoksiężnika), innym razem dowodzącej zespołowego porozumienia – umiejętnie żonglując konwencjami inspirowanymi muzycznością (w szerokim tego słowa znaczeniu, odnoszonym do ruchu, pieśni, rytmu). Oszczędny biały śpiew przypominał gardzienickie „Metamorfozy”, ale nie uciekał w folklor. Fisharmonia oraz poręczny dzwon nie tylko generowały dźwięki, ale także pełniły zadania scenograficzne. Ową muzyczność, zakorzenioną gdzieś w powidokach ludowości, skontrowano wybrzmiewającym w finale utworem Marianne Faithfull „City Of Quartz”. Pozytywkowa melodia i panosząca się w tekście „kurwa babilońska” korespondowały z przechodzącym wcześniej przez scenę korowodem średniowiecznych idiotów/opętanych, w swej – znakomicie odegranej – pokraczności sięgających wyższych rejestrów estetyki. Wszystko to podkreślało nierozłączność przeciwieństw. W całym przedstawieniu Gorczyński dość uważnie bada możliwość zaślubin tego, co ciemne, z tym, co świetliste, piekła z niebem, jak mógłby stwierdzić Blake.

Dopóki taniec i choreografia koncentrowały się na „Nowych Atenach”, całość przedstawienia wchodziła w rejestry niemal baśniowe. Każde sięgniecie do Blake’a działało jednak na niekorzyść spektaklu, ukazując pęknięcia, w których gubiła się przewodnia koncepcja widowiska. Obecność myśli angielskiego wizjonera przejawiała się bardziej w ruchu niż w słowie, podczas gdy muzycznie zapętlone cytaty z Chmielowskiego odpowiadały za integralność świata przedstawionego, którego centrum był sam ksiądz/czarnoksiężnik. Leżoń – operując maską, będącą atrybutem ciemnych sił – raz po raz wizualizował dwoistość ludzkiej natury. „Jest czort! Jest zło! Jest system kopernikański! Ale jest też cebulka…”. W rozdwojonym bohaterze odbijało się szaleństwo. Widowiskowe, transgresyjne momenty przypominające opętanie przestrzegały przed fanatyczną wiarą w cokolwiek. Nie tylko w koguty o wężowych ogonach, ale także w zdrowy rozsądek, który podpowiadał kapłanowi, że teoria Mikołaja Kopernika nie może być prawdziwa. Duchowny dzielnie walczył z bestią ślepego zawierzenia – mając na uwadze, że „smoka pokonać trudno, ale starać się trzeba”.

Demoniczność kontrowano dużą dawką humoru. Gorczyński i współautor adaptacji Kajetan Mojsak
 nie ulegli jednak możliwej pokusie i nie uczynili z „Nowych Aten” pretekstu do rechotu nad zaściankowością i ciemnotą. Komizmem artyści grali subtelnie, choć wyraźnie, jak w chwilach prezentacji żołnierza polskiego o dumnie wyprężonej piersi otulonej westchnieniami omdlewających panien, czy też kozackiego najazdu, gdzie stepowi wojacy dosiadają drewnianych wierzchowców gniadej maści. Jakie jest krzesło, każdy widzi – mógłby dopowiedzieć Chmielowski. Jaki jest teatr – także. Gorczyński potrafi spojrzeć na rzeczywistość przez teatr jak przez lunetę. W „Nowych Atenach” znalazł punkt, w którym wyobraźnia poddaje w wątpliwość rozsądek i zmusza do podważenia prymatu rozumu, co szczególnie ważne w dzisiejszych czasach nieustannego postępu.'''
model_path = "C:/Users/dariu/model_5epoch_gatunek_large"
model_genre = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = HerbertTokenizerFast.from_pretrained(model_path)

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
#sampled_df['combined_text'] =sampled_df['Tytuł artykułu'].astype(str) + " </tytuł>" + sampled_df['Tekst artykułu'].astype(str)



        
        # Tokenizacja i przewidywanie dla modelu True/False
inputs_t_f = tokenizer_t_f(text, return_tensors="pt", padding=True, truncation=True, max_length=514)
outputs_t_f = model_t_f(**inputs_t_f)
predictions_t_f = torch.softmax(outputs_t_f.logits, dim=1)
predicted_index_t_f = predictions_t_f.argmax().item()
predicted_label_t_f = label_encoder_t_f.inverse_transform([predicted_index_t_f])[0]
confidence_t_f = predictions_t_f.max().item() * 100  # Procent pewności

genre = ''
haslo = ''
confidence_genre = ''  # Początkowa wartość pewności dla gatunku
confidence_haslo = ''  # Początkowa wartość pewności dla hasła

inputs_genre = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=514)
outputs_genre = model_genre(**inputs_genre)
predictions_genre = torch.softmax(outputs_genre.logits, dim=1)
predicted_index_genre = predictions_genre.argmax().item()
genre = label_encoder.inverse_transform([predicted_index_genre])[0]
confidence_genre = predictions_genre.max().item() * 100  # Procent pewności

# Przewidywanie hasła
inputs_hasla = tokenizer_hasla(text, return_tensors="pt", padding=True, truncation=True, max_length=514)
outputs_hasla = model_hasla(**inputs_hasla)
predictions_hasla = torch.softmax(outputs_hasla.logits, dim=1)
predicted_index_hasla = predictions_hasla.argmax().item()
haslo = label_encoder_hasla.inverse_transform([predicted_index_hasla])[0]
confidence_haslo = predictions_hasla.max().item() * 100  # Procent pewności


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


nlp = spacy.load("pl_core_news_lg")

tokens = tokenizer.tokenize(text)
max_tokens = 514  # Przykładowe ograniczenie modelu
token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]
# Analiza każdego fragmentu osobno
ner_results = []
for fragment in fragments:
    ner_results.extend(nlp1(fragment))
combined_entities = combine_tokens(ner_results)

combined_entities_selected = [entity for entity in combined_entities if entity['score'] >= 0.92]
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
    
    
    original_entities = entity_lemmatization_dict.get(first_entity_info[0], [])
    
    entity = next(iter(original_entities))[0]
    viaf_url, entity_type = None, "Not found"
    if original_entities:
        viaf_url, _ = check_viaf_with_fuzzy_match(next(iter(original_entities))[0])  # Pobieranie pierwszego elementu z setu
        entity_type = next(iter(original_entities))[1]
    

      


