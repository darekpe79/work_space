# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:56:22 2024

@author: dariu
"""
import json
from transformers import AutoModelForTokenClassification, HerbertTokenizerFast, pipeline

# Ścieżki do modelu i pliku tag2id
model_path = "D:/Nowa_praca/adnotacje_spubi/model_29_11_2024/"
tag2id_path = "D:/Nowa_praca/adnotacje_spubi/model_29_11_2024/tag2id.json"

# Ładowanie modelu
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Ładowanie tokenizatora
tokenizer = HerbertTokenizerFast.from_pretrained(model_path)

# Załadowanie mapowania tagów z pliku JSON
with open(tag2id_path, "r", encoding="utf-8") as f:
    tag2id = json.load(f)

# Utworzenie odwróconego mapowania id2tag
id2tag = {v: k for k, v in tag2id.items()}

# Przypisanie mapowania do konfiguracji modelu
model.config.label2id = tag2id
model.config.id2label = id2tag

# Tworzenie pipeline NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Przykładowy tekst
text = ''' 
[rec. ks.:] Gabriele Muschter, Ruediger Thomas: Jenseits der Staatskultur. Traditionen autonomer Kunst in der DDR. Muenchen-Wien 1992.  
'''

# Analiza tekstu za pomocą pipeline
results = nlp(text)

# Wyświetlenie wyników
for entity in results:
    print(f"Tekst: {entity['word']}, Etykieta: {entity['entity_group']}, Skala pewności: {entity['score']:.2f}")

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import json
from transformers import AutoConfig
# Ścieżka do modelu
model_directory = "C:/Users/dariu/model_ner_3/"

# Ścieżka do pliku tag2id.json
tag2id_path = "C:/Users/dariu/model_ner_3/tag2id.json"

# Ładowanie mapowania tag2id
with open(tag2id_path, 'r') as f:
    tag2id = json.load(f)

# Odwrócenie mapowania tag2id na id2tag
id2tag = {v: k for k, v in tag2id.items()}

# Załaduj konfigurację modelu i zaktualizuj mapowania etykiet
config = AutoConfig.from_pretrained(model_directory)
config.label2id = tag2id
config.id2label = id2tag
config.save_pretrained(model_directory)

# Załaduj tokenizer i model z aktualną konfiguracją
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForTokenClassification.from_pretrained(model_directory, config=config)
print(type(tokenizer))

# Utwórz pipeline NER z agregacją
nlp1 = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Przykładowy tekst
text = 'Spektakl "Makbet" na podstawie dramatu Szekspira "MAKBET"'
text ="W dniach 12-14 maja 2024 roku w Centrum Kultury odbywa się Festiwal Literacki, na którym zostaną nagrodzone najlepsze powieści roku. Jednym z wyróżnionych autorów będzie Jan Kowalski za swoją książkę 'Tajemnice Starego Miasteczka'. Dodatkowo, wieczór zakończy się premierą spektaklu 'Romeo i Julia' w Teatrze Narodowym."

text = "Nowa adaptacja sztuki 'Hamlet' autorstwa Williama Szekspira zostanie wystawiona w Teatrze Wielkim w Warszawie podczas obchodów Dnia Teatru. Wydarzenie to przyciągnie miłośników klasycznych dramatów oraz krytyków sztuki. Dodatkowo, na targach książki w Krakowie pojawi się najnowsza powieść Agnieszki Nowak 'Cienie Przeszłości'."

text = "Podczas Międzynarodowego Festiwalu Filmowego w Gdyni odbędzie się pokaz nowej adaptacji powieści 'Lśnienie' autorstwa Stephena Kinga. Wydarzenie to zorganizuje się w partnerstwie z lokalnym wydawnictwem 'Literat'. Na scenie Teatru im. Marii Konopnickiej zaprezentowany zostanie dramat 'Dziady' Adama Mickiewicza."
max_tokens = 514  # Przykładowe ograniczenie modelu
tokens = tokenizer.tokenize(text)
token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]
# Przetwarzanie tekstu
ner_results = []
for fragment in fragments:
    ner_results.extend(nlp1(fragment))
ner_results = nlp1(text)

# Zwrócenie tylko encji z etykietami różnymi od 'O'
filtered_entities = [
    {
        "word": entity['word'],
        "type": entity['entity_group'],
        "score": entity['score']
    }
    for entity in ner_results
    if entity['entity_group'] != 'O'
]

# Wyświetlenie wyników
for entity in filtered_entities:
    print(f"Word: {entity['word']}, Type: {entity['type']}, Score: {entity['score']:.4f}")



#%%

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Ścieżka do modelu pietruszkowiec
model_checkpoint = "pietruszkowiec/herbert-base-ner"

# Załaduj tokenizer i model z aktualną konfiguracją
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Utwórz pipeline NER z agregacją
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
text = "Nazywam się Grzegorz Brzęszczyszczykiewicz, pochodzę "\
    "z Chrząszczyżewoszczyc, pracuję w Łękołodzkim Urzędzie Powiatowym"

# Przetwarzanie tekstu
ner_results = nlp(text)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Nazywam się Grzegorz Brzęszczyszczykiewicz, pochodzę "\
    "z Chrząszczyżewoszczyc, pracuję w Łękołodzkim Urzędzie Powiatowym"

ner_results = nlp(example)
print(ner_results)

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Załadowanie modelu i tokenizatora
model_checkpoint = "C:/Users/dariu/model_ner_3/"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Utworzenie pipeline NER
nlp1 = pipeline("ner", model=model, tokenizer=tokenizer)

# Przykładowy tekst
text = ' Spektakl "Makbet" na podstawie dramatu Szekspira "MAKBET"'  # Twój tekst tutaj

# Przetwarzanie tekstu
proba = nlp1(text)

# Wyświetlenie wyników
print(proba)

#%%
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json

# Ścieżka do katalogu, w którym zapisany jest model i tokenizator
model_directory = "C:/Users/dariu/model_ner_3/"

# Ścieżka do pliku JSON z mapowaniem tag2id
tag2id_path = "C:/Users/dariu/model_ner_3/tag2id.json"

# Ładowanie modelu
model = AutoModelForTokenClassification.from_pretrained(model_directory)

# Ładowanie tokenizatora
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Ładowanie mapowania tag2id
with open(tag2id_path, 'r') as f:
    tag2id = json.load(f)

# Odwrócenie mapowania tag2id na id2tag dla dekodowania predykcji
id2tag = {v: k for k, v in tag2id.items()}

import torch
import numpy as np
import torch.nn.functional as F
def predict_ner(text, model, tokenizer, id2tag):
    # Tokenizacja tekstu
    tokenized_input = tokenizer.encode_plus(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    
    input_ids = tokenized_input["input_ids"]
    
    # Predykcja modelu
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Dekodowanie etykiet
    predictions = torch.argmax(outputs.logits, dim=2)
    
    # Konwersja predykcji na listę etykiet
    predicted_labels = [id2tag[label_id.item()] for label_id in predictions[0]]
    
    # Pobranie tokenów
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Zwrócenie sparowanych tokenów i etykiet
    return [(token, label) for token, label in zip(tokens, predicted_labels) if token not in tokenizer.all_special_tokens]
def predict_ner(text, model, tokenizer, id2tag):
    """
    Funkcja do przewidywania Named Entities (NER) w podanym tekście.
    
    Args:
        text (str): Tekst do analizy.
        model (transformers.PreTrainedModel): Wytrenowany model NER.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer odpowiedni dla modelu.
        id2tag (dict): Mapa id etykiet na ich nazwy.
    
    Returns:
        list of dict: Lista wykrytych encji z tokenami, typem i średnim score.
    """
    # Tokenizacja tekstu
    tokenized_input = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
    
    input_ids = tokenized_input["input_ids"]
    attention_mask = tokenized_input["attention_mask"]
    
    # Predykcja modelu
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Logits do prawdopodobieństw
    probabilities = F.softmax(outputs.logits, dim=2)
    
    # Dekodowanie etykiet
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = [id2tag[label_id.item()] for label_id in predictions[0]]
    
    # Pobranie tokenów
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Pobranie score dla przewidzianych etykiet
    scores = probabilities[0, range(len(predictions[0])), predictions[0]].tolist()
    
    # Łączenie encji
    entities = []
    current_entity = None
    
    for token, label, score in zip(tokens, predicted_labels, scores):
        # Usunięcie tokenów specjalnych
        if token in tokenizer.all_special_tokens:
            continue
        
        # Sprawdzenie końca słowa
        end_of_word = "</w>" in token
        cleaned_word = token.replace("</w>", "")
        
        if label.startswith('B-'):
            # Jeśli jest to początek nowej encji, dodaj poprzednią encję do listy
            if current_entity:
                # Oblicz średnie prawdopodobieństwo
                current_entity['score'] /= current_entity['token_count']
                entities.append(current_entity)
            
            # Rozpocznij nową encję
            current_entity = {
                "word": cleaned_word,
                "type": label[2:],  # Usunięcie prefiksu 'B-'
                "score_sum": score,
                "token_count": 1
            }
        elif label.startswith('I-') and current_entity and label[2:] == current_entity["type"]:
            # Kontynuacja obecnej encji
            if end_of_word:
                current_entity["word"] += " " + cleaned_word
            else:
                current_entity["word"] += cleaned_word
            current_entity["score_sum"] += score
            current_entity["token_count"] += 1
        else:
            # Label 'O' lub inny typ encji
            if current_entity:
                # Oblicz średnie prawdopodobieństwo
                current_entity['score'] /= current_entity['token_count']
                entities.append(current_entity)
                current_entity = None
    
    # Dodanie ostatniej encji, jeśli istnieje
    if current_entity:
        current_entity['score'] /= current_entity['token_count']
        entities.append(current_entity)
    
    return entities
# Przykładowy tekst do analizy
text = ' Spektakl "Makbet" na podstawie dramatu Szekspira "MAKBET"'
tokens = tokenizer.tokenize(text)
max_tokens = 514  # Przykładowe ograniczenie modelu
token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]
# Użycie funkcji
predicted_ner = predict_ner(text, model, tokenizer, id2tag)
ner_results = []
for fragment in fragments:
    ner_results.extend(predict_ner(fragment, model, tokenizer, id2tag))
# Wyświetlenie wyników
for token, label in predicted_ner:
    print(f"{token}: {label}")
    
def combine_tokens(ner_results):
    combined_entities = []
    current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0}
    previous_end_of_word = False  # Czy poprzedni token kończył słowo?

    for token, label in ner_results:
        end_of_word = "</w>" in token
        cleaned_word = token.replace("</w>", "")
        
        if label.startswith('B-'):
            # Jeśli jest to początek nowej encji, dodaj poprzednią encję do listy
            if current_entity["word"]:
                combined_entities.append({
                    "word": current_entity["word"],
                    "type": current_entity["type"],
                    "score": current_entity["score_sum"] / current_entity["token_count"]
                })
            # Rozpocznij nową encję
            current_entity = {
                "word": cleaned_word,
                "type": label[2:],  # Usunięcie prefiksu B-
                "score_sum": 1.0,    # Brak informacji o score, ustawiamy domyślnie
                "token_count": 1
            }
        elif label.startswith('I-') and current_entity["type"] == label[2:]:
            # Kontynuacja obecnej encji
            if previous_end_of_word:
                # Jeśli poprzedni token kończył słowo, dodaj spację
                current_entity["word"] += " " + cleaned_word
            else:
                current_entity["word"] += cleaned_word
            current_entity["score_sum"] += 1.0  # Brak informacji o score
            current_entity["token_count"] += 1
        else:
            # Label 'O' lub inny typ encji
            if current_entity["word"]:
                combined_entities.append({
                    "word": current_entity["word"],
                    "type": current_entity["type"],
                    "score": current_entity["score_sum"] / current_entity["token_count"]
                })
                current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0}
        
        previous_end_of_word = end_of_word

    # Dodaj ostatnią encję, jeśli istnieje
    if current_entity["word"]:
        combined_entities.append({
            "word": current_entity["word"],
            "type": current_entity["type"],
            "score": current_entity["score_sum"] / current_entity["token_count"]
        })

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
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json
from tqdm import tqdm

# Ścieżka do modelu
model_path = "C:/Users/dariu/model_NER/"

# Załaduj model i tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Załaduj tag2id i utwórz id2tag
tag2id_path = f"{model_path}/tag2id.json"
with open(tag2id_path, 'r') as f:
    tag2id = json.load(f)
id2tag = {int(v): k for k, v in tag2id.items()}

def nlp1(text):
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True,
        is_split_into_words=False
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0]
    word_ids = encoding.word_ids()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    ner_results = []
    
    for idx, (pred_id, word_id) in enumerate(zip(predictions, word_ids)):
        if word_id is None:
            continue  # Pomijamy tokeny specjalne
        
        token = tokens[idx]
        tag_id = pred_id
        tag = id2tag.get(tag_id, 'O')
        start = int(offset_mapping[idx][0])
        end = int(offset_mapping[idx][1])
        
        ner_results.append({
            'word': token,
            'entity': tag,
            'start': start,
            'end': end
        })
    
    return ner_results

def combine_tokens(ner_results):
    combined_entities = []
    current_entity = None

    for token in ner_results:
        tag = token['entity']
        word = token['word']
        score = token['score']
        start = token['start']
        end = token['end']

        if tag.startswith('B-'):
            if current_entity:
                combined_entities.append(current_entity)
            current_entity = {
                'entity': tag[2:],
                'word': tokenizer.convert_tokens_to_string([word]),
                'start': start,
                'end': end,
                'score_sum': score,
                'token_count': 1
            }
        elif tag.startswith('I-') and current_entity and current_entity['entity'] == tag[2:]:
            current_entity['word'] += tokenizer.convert_tokens_to_string([word])
            current_entity['end'] = end
            current_entity['score_sum'] += score
            current_entity['token_count'] += 1
        else:
            if current_entity:
                combined_entities.append(current_entity)
                current_entity = None

    if current_entity:
        combined_entities.append(current_entity)

    for entity in combined_entities:
        entity['score'] = entity['score_sum'] / entity['token_count']
        del entity['score_sum']
        del entity['token_count']

    return combined_entities

def ner_on_long_text(text):
    max_length = 510
    tokens = tokenizer.tokenize(text)
    token_fragments = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]

    ner_results = []
    offset = 0
    for fragment in fragments:
        fragment_ner_results = nlp1(fragment)
        # Aktualizacja offsetów
        for entity in fragment_ner_results:
            entity['start'] += offset
            entity['end'] += offset
        ner_results.extend(fragment_ner_results)
        offset += len(fragment)

    combined_entities = combine_tokens(ner_results)
    return combined_entities

# Przykładowy tekst
sample_text = "Wczoraj czytałem książkę 'Pan Tadeusz' Adama Mickiewicza i oglądałem sztukę 'Dziady'."

entities = ner_on_long_text(sample_text)

print("Wyodrębnione jednostki:")
for entity in entities:
    print(f"Typ: {entity['entity']}, Tekst: {entity['word']}, Start: {entity['start']}, End: {entity['end']}, Score: {entity['score']:.4f}")




text = ' Spektakl "Makbet" na podstawie dramatu Szekspira "MAKBET"'
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
tokens = tokenizer.tokenize(text)
max_tokens = 514  # Przykładowe ograniczenie modelu
token_fragments = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
fragments = [tokenizer.convert_tokens_to_string(fragment) for fragment in token_fragments]
ner_results = []
for fragment in fragments:
    ner_results.extend(fragment)
combined_entities = combine_tokens(fragments)
