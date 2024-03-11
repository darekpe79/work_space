# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:04:10 2024

@author: dariu
"""

from transformers import BertTokenizer
import numpy as np
from transformers import HerbertTokenizerFast

# Initialize the tokenizer with the Polish model
tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-base-cased')

transformed_data_train = [('Who is Nishanth ada ? asdasd awdas qwe Zresztą, co by nie mówić jest to przecież operetka sanatoryjna/fot. Monika Stolarska/Na sam dźwięk słowa sanatorium włosy stają dęba, a przed oczami pojawiają się mroczki. Coś wewnętrznie zaczyna się dziać, a my sami wolelibyśmy uciec niż podjąć dyskusję na temat wszelkiej maści uzdrowisk. Ale na naszej drodze pojawia się Cezary Tomaszewski, który tę drogę ucieczki odcina, a następnie sadza na teatralnej widowni i „zarządza” grupowe oglądanie sanatoryjnej operetki.', {
        'entities': [(0, 20, 'PLAY'), (150, 175, 'PLAY')]})]

transformed_data_train=[('"Turnus mija, a ja niczyja", reż. Cezary Tomaszewski Sam tytuł tego spektaklu wydaje się nieco podejrzany, bo cóż możemy sobie pomyśleć kiedy słyszymy: Turnus mija, a ja niczyja. Okazuje się jednak, że to o czym pomyśleliśmy chociaż przez krótką chwilę jest bardzo złudne, a Cezary Tomaszewski na scenie MOS Teatru Słowackiego w Krakowie tworzy dzieło zaskakujące i nietypowe. Zresztą, co by nie mówić jest to przecież operetka sanatoryjna/fot. Monika Stolarska/Na sam dźwięk słowa sanatorium włosy stają dęba, a przed oczami pojawiają się mroczki. Coś wewnętrznie zaczyna się dziać, a my sami wolelibyśmy uciec niż podjąć dyskusję na temat wszelkiej maści uzdrowisk. Ale na naszej drodze pojawia się Cezary Tomaszewski, który tę drogę ucieczki odcina, a następnie sadza na teatralnej widowni i „zarządza” grupowe oglądanie sanatoryjnej operetki.',
  {'entities': [(1, 26, 'PLAY'), (150, 175, 'PLAY')]})]
len("Turnus mija, a ja niczyja, reż. Cezary Tomaszewski Sam tytuł tego spektaklu wydaje się nieco podejrzany, bo cóż możemy sobie pomyśleć kiedy słyszymy: Turnus mija, a ja niczyja. Okazuje się jednak, że to o czym pomyśleliśmy chociaż przez krótką chwilę jest bardzo złudne, a Cezary Tomaszewski na scenie MOS Teatru Słowackiego w Krakowie tworzy dzieło zaskakujące i nietypowe. Zresztą, co by nie mówić jest to przecież operetka sanatoryjna/fot. Monika Stolarska/Na sam dźwięk słowa sanatorium włosy stają dęba, a przed oczami pojawiają się mroczki. Coś wewnętrznie zaczyna się dziać, a my sami wolelibyśmy uciec niż podjąć dyskusję na temat wszelkiej maści uzdrowisk. Ale na naszej drodze pojawia się Cezary Tomaszewski, który tę drogę ucieczki odcina, a następnie sadza na teatralnej widowni i „zarządza” grupowe oglądanie sanatoryjnej operetki.")
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


json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
                
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'

# ... więcej plików w razie potrzeby

# Użycie funkcji

df2 = load_and_merge_data(json_file_path2, excel_file_path2)
import random
import os
import json
from spacy.util import minibatch, compounding
from spacy.training.example import Example

df2['combined_text'] = df2['Tytuł artykułu'] + " " + df2['Tekst artykułu']
import pandas as pd
import spacy
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from spacy.training import Example
from spacy.scorer import Scorer

from spacy.tokens import Span




# Funkcja do oznaczania słów z tytułów spektakli w tekście
def mark_titles(text, title):
    # Escapowanie specjalnych znaków w tytule
    title_pattern = re.escape(title) + r"(?![\w-])"  # Aby uniknąć dopasowania w środku słowa, dodajemy negative lookahead
    # Oznaczanie tytułu w tekście znacznikami
    marked_text = re.sub(title_pattern, r"[PLAY]\g<0>[/PLAY]", text, flags=re.IGNORECASE)
    return marked_text

df2.dropna(subset=['Tytuł spektaklu'], inplace=True)

df2['Tytuł spektaklu'] = df2['Tytuł spektaklu'].fillna('')
df2['marked_text'] = df2.apply(lambda row: mark_titles(row['combined_text'], row['Tytuł spektaklu']), axis=1)
def prepare_data_for_ner(text):
    pattern = r"\[PLAY\](.*?)\[/PLAY\]"
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
        entities.append((start_entity, end_entity, "PLAY"))
        last_end = end  # Zaktualizuj pozycję ostatniego znalezionego końca znacznika

    clean_text += text[last_end:]  # Dodaj pozostały tekst po ostatnim znaczniku

    return clean_text, {"entities": entities}

df2['spacy_marked'] = df2['marked_text'].apply(prepare_data_for_ner)
import json
import os
transformed_data=df2['spacy_marked'].to_list()
json_files_dir = 'D:/Nowa_praca/dane_model_jezykowy/jsony_spektakl/'
json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]

# Iterate over each JSON file
for json_file in json_files:
    file_path = os.path.join(json_files_dir, json_file)
    
    # Load the JSON data from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
        # Extract annotations from the JSON data
        for item in json_data['annotations']:
            text = item[0]  # Extract text
            text = text.replace("[/tytuł]", "")
            entities = item[1]['entities']  # Assuming this directly gives a list of tuples [(start, end, label), ...]
            tuples_list = [tuple(item) for item in item[1]['entities']]
            # Append to the existing dataset
            transformed_data.append((text, {'entities':tuples_list}))   
transformed_data
def find_nearest_acceptable_split_point(pos, text, total_length):
    """
    Finds the nearest split point that does not split words or sentences,
    preferring to split at punctuation followed by space or at natural sentence boundaries.
    """
    if pos <= 0:
        return 0
    if pos >= total_length:
        return total_length
    
    for offset in range(1, min(50, pos, total_length - pos) + 1):
        left = pos - offset
        right = pos + offset

        if left > 0 and text[left - 1] in '.?!' and text[left] == ' ':
            return left + 1

        if right < total_length and text[right - 1] in '.?!' and text[right] == ' ':
            return right + 1
    
    for offset in range(1, min(50, pos, total_length - pos) + 1):
        left = pos - offset
        right = pos + offset

        if left > 0 and text[left] == ' ':
            return left + 1

        if right < total_length and text[right] == ' ':
            return right + 1

    return pos

def split_text_around_entities_adjusted_for_four_parts(data_list):
    split_data = []
    
    for text, annotation in data_list:
        entities = sorted(annotation['entities'], key=lambda e: e[0])
        total_length = len(text)
        ideal_part_length = total_length // 5  # Adjusted for four parts
        
        split_points = [0]
        current_split = 0
        
        for _ in range(4):  # Adjusted to perform three splits for four parts
            proposed_split = current_split + ideal_part_length
            if proposed_split >= total_length:
                break
            
            adjusted_split = find_nearest_acceptable_split_point(proposed_split, text, total_length)
            
            for start, end, _ in entities:
                if adjusted_split > start and adjusted_split < end:
                    adjusted_split = end
                    break
            
            if adjusted_split != current_split:
                split_points.append(adjusted_split)
                current_split = adjusted_split
        
        split_points.append(total_length)
        
        last_split = 0
        for split in split_points[1:]:
            part_text = text[last_split:split].strip()
            part_entities = [(start - last_split, end - last_split, label) for start, end, label in entities if start >= last_split and end <= split]
            split_data.append((part_text, {'entities': part_entities}))
            last_split = split

    return split_data
transformed_data = split_text_around_entities_adjusted_for_four_parts(transformed_data)

transformed_data = [data for data in transformed_data if data[1]['entities']]

from sklearn.model_selection import train_test_split




transformed_data_train, transformed_data_eval = train_test_split(transformed_data, test_size=0.1, random_state=42)
#transformed_data_train=[transformed_data_train[0]]

# def adjust_data_for_ner(data, tokenizer, max_length=512):
#     adjusted_data = []

#     for text, annotation in data:
#         tokens = tokenizer.tokenize(text)
#         entities = annotation['entities']
#         current_chunk = []
#         current_entities = []
#         chunk_size = 0

#         for entity in entities:
#             start, end, label = entity
#             entity_tokens = tokenizer.tokenize(text[start:end])
#             entity_start = chunk_size + len(tokenizer.tokenize(text[:start]))
#             entity_end = entity_start + len(entity_tokens)

#             if entity_end <= max_length - 1:  # -1 to account for special tokens
#                 current_entities.append((entity_start, entity_end, label))
#             else:
#                 # If the entity doesn't fit, start a new chunk
#                 if current_chunk:
#                     adjusted_data.append((' '.join(current_chunk), {'entities': current_entities}))
#                 current_chunk = tokens[chunk_size:entity_start] + entity_tokens
#                 current_entities = [(0, len(entity_tokens), label)]
#                 chunk_size = entity_start

#         if current_chunk or (chunk_size < len(tokens)):
#             remaining_tokens = tokens[chunk_size:max_length - 1]
#             adjusted_data.append((' '.join(current_chunk + remaining_tokens), {'entities': current_entities}))

#     return adjusted_data
# adjusted_data_train = adjust_data_for_ner(transformed_data_train, tokenizer)
# adjusted_data_eval = adjust_data_for_ner(transformed_data_eval, tokenizer)

#TU Dam Nową funkcję?
from transformers import BertTokenizerFast
import torch

# Przygotowanie tokenizera
#tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-base-cased')



import torch

import torch

def prepare_data(data, tokenizer, tag2id, max_length=512):
    input_ids = []
    attention_masks = []
    labels = []
    
    for text, annotation in data:
        tokenized_input = tokenizer.encode_plus(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        offset_mapping = tokenized_input["offset_mapping"].squeeze().tolist()[1:-1]  # Usunięcie mapowania dla [CLS] i [SEP]
        sequence_labels = ['O'] * len(offset_mapping)  # Inicjalizacja etykiet jako 'O' dla tokenów (bez [CLS] i [SEP])
        
        for start, end, label in annotation['entities']:
            entity_start_index = None
            entity_end_index = None
            
            for idx, (offset_start, offset_end) in enumerate(offset_mapping):
                if start == offset_start or (start > offset_start and start < offset_end):
                    entity_start_index = idx
                if (end > offset_start and end <= offset_end):
                    entity_end_index = idx
                    break

            if entity_start_index is not None and entity_end_index is not None:
                sequence_labels[entity_start_index] = f'B-{label}'  # Begin label
                for i in range(entity_start_index + 1, entity_end_index + 1):
                    sequence_labels[i] = f'I-{label}'  # Inside label
        
        # Dodajemy 'O' dla [CLS] i [SEP] oraz dopasowujemy długość etykiet do max_length
        full_sequence_labels = ['O'] + sequence_labels + ['O'] * (max_length - len(sequence_labels) - 1)
        label_ids = [tag2id.get(label, tag2id['O']) for label in full_sequence_labels]
        
        input_ids.append(tokenized_input['input_ids'].squeeze().tolist())
        attention_masks.append(tokenized_input['attention_mask'].squeeze().tolist())
        labels.append(label_ids)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_masks, labels









# Mapowanie etykiet
tag2id = {'O': 0, 'B-PLAY': 1, 'I-PLAY': 2}

# Przygotowanie danych
input_ids, attention_masks, labels = prepare_data(transformed_data_train, tokenizer, tag2id)
# Przygotowanie danych ewaluacyjnych
input_ids_eval, attention_masks_eval, labels_eval = prepare_data(transformed_data_eval, tokenizer, tag2id)


# Weryfikacja wyników
print(input_ids.shape, attention_masks.shape, labels.shape)

example_idx = 1  # indeks przykładu, który chcemy wydrukować

# Konwersja input_ids do tokenów
tokens = tokenizer.convert_ids_to_tokens(input_ids[example_idx])

print(f"Tokens:\n{tokens}\n")
print(f"Input IDs:\n{input_ids[example_idx]}\n")
print(f"Attention Masks:\n{attention_masks[example_idx]}\n")
print(f"Tag IDs:\n{labels[example_idx]}\n")

# Wydrukuj skojarzone z tokenami etykiety (dla lepszej czytelności)
tags = [list(tag2id.keys())[list(tag2id.values()).index(tag_id)] if tag_id in tag2id.values() else 'PAD' for tag_id in labels[example_idx]]
print(f"Tags:\n{tags}\n")

# Przykład weryfikacji dla pojedynczego przykładu
example_text, example_annotations = transformed_data[1]
example_input_ids, example_attention_masks, example_labels = prepare_data([transformed_data[1]], tokenizer, tag2id)

print("Tekst:", example_text)
print("\nTokeny i ich etykiety:")
tokens = tokenizer.convert_ids_to_tokens(example_input_ids[0])
for token, label_id in zip(tokens, example_labels[0]):
    label = list(tag2id.keys())[list(tag2id.values()).index(label_id)]
    print(f"{token}: {label}")



from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    'allegro/herbert-base-cased',
    num_labels=len(tag2id)
)
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
eval_dataset = TensorDataset(input_ids_eval, attention_masks_eval, labels_eval)

# DataLoader dla danych ewaluacyjnych
eval_loader = DataLoader(
    eval_dataset,
    batch_size=16,  # Dostosuj zgodnie z potrzebami
    shuffle=False  # Nie ma potrzeby mieszać danych ewaluacyjnych
)
# Przygotowanie TensorDataset
train_dataset = TensorDataset(input_ids, attention_masks, labels)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # Możesz dostosować w zależności od zasobów
    sampler=RandomSampler(train_dataset)  # Mieszanie danych
)
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# Przenieś model na odpowiednie urządzenie (GPU, jeśli dostępne)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Pętla treningowa
model.train()
for epoch in range(3):  # Liczba epok
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attention_mask, b_labels = batch
        
        # Reset gradientów
        model.zero_grad()
        
        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        
        # Backward pass i optymalizacja
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")



from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Zakładając, że model został już wytrenowany i eval_loader jest gotowy

# Przenieś model na odpowiednie urządzenie (GPU lub CPU)
model.to(device)

# Przygotuj listy do zbierania prawdziwych etykiet i predykcji
true_labels = []
pred_labels = []

# Ustaw model w tryb ewaluacji
model.eval()

# Wyłącz obliczanie gradientów
with torch.no_grad():
    for batch in eval_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attention_mask, b_labels = batch
        
        # Przeprowadź predykcję
        outputs = model(b_input_ids, attention_mask=b_attention_mask)
        
        # Logits - wyniki przed funkcją aktywacji
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Dodaj predykcje i prawdziwe etykiety do list
        pred_labels.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

# Wyciąganie płaskich list prawdziwych etykiet i predykcji, pomijając padding
flat_true_labels = [item for sublist in true_labels for item in sublist if item != tag2id['O']]
flat_pred_labels = [item for sublist in pred_labels for item in sublist[:len(sublist)] if item != tag2id['O']]

# Oblicz metryki
precision = precision_score(flat_true_labels, flat_pred_labels, average='macro', zero_division=0)
recall = recall_score(flat_true_labels, flat_pred_labels, average='macro', zero_division=0)
f1 = f1_score(flat_true_labels, flat_pred_labels, average='macro', zero_division=0)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Zapisz model
model.save_pretrained('./saved_model')


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")
model = AutoModel.from_pretrained("allegro/herbert-large-cased")

output = model(
    **tokenizer.batch_encode_plus(
        [
            (
                "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
                "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy."
            )
        ],
    padding='longest',
    add_special_tokens=True,
    return_tensors='pt'
    )
)

















import torch

def encode_tags(tags, length, tag2id, default_tag='O', pad_tag=-100):
    """Koduje tagi do formatu ID, dodając padding."""
    tag_ids = [tag2id.get(tag, tag2id[default_tag]) for tag in tags]
    tag_ids += [pad_tag] * (length - len(tag_ids))
    return tag_ids

def prepare_data_for_bert(adjusted_data, tokenizer, tag2id, max_length=512):
    input_ids = []
    attention_masks = []
    tag_ids = []

    for text, annotations in adjusted_data:
        # Podział tekstu na tokeny
        tokens = text.split(' ')
        entity_tags = ['O'] * len(tokens)  # Domyślnie wszystkie tagi to 'O'

        for start, end, label in annotations['entities']:
            entity_tags[start] = f'B-{label}'  # Początkowy tag encji
            for i in range(start + 1, end):
                entity_tags[i] = f'I-{label}'  # Tagi wewnętrzne encji

        # Dodanie specjalnych tokenów
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        special_tokens_count = 2  # [CLS] i [SEP]
        entity_tags = ['O'] + entity_tags + ['O']  # Tagi dla [CLS] i [SEP]

        # Kodowanie tokenów i tagów
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids += [tokenizer.pad_token_id] * (max_length - len(token_ids))  # Padding do max_length
        attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))  # Maskowanie
        tag_ids_sequence = encode_tags(entity_tags, max_length, tag2id, pad_tag=-100)

        input_ids.append(token_ids)
        attention_masks.append(attention_mask)
        tag_ids.append(tag_ids_sequence)

    # Konwersja na tensory
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    tag_ids = torch.tensor(tag_ids, dtype=torch.long)

    return input_ids, attention_masks, tag_ids

# Przykład użycia:
tag2id = {'O': 0, 'B-PERSON': 1, 'I-PERSON': 2, 'B-LOC': 3, 'I-LOC': 4}
adjusted_data_train = [
    ('W ho</w> is</w> Ka mal</w> K hu mar</w> ?', {'entities': [(3, 8, 'PERSON')]}),
    ('I</w> li ke</w> London</w> and</w> Berlin</w> .', {'entities': [(3, 4, 'LOC'), (5, 6, 'LOC')]})
]
# Użycie fikcyjnego tokenizer do przykładu; w praktyce należy użyć tokenizera z modelu BERT
tokenizer = tokenizer # Załóżmy, że tokenizer jest już zdefiniowany

input_ids, attention_masks, tag_ids = prepare_data_for_bert(adjusted_data_train, tokenizer, tag2id, max_length=512)

example_idx = 0  # indeks przykładu, który chcemy wydrukować

# Konwersja input_ids do tokenów
tokens = tokenizer.convert_ids_to_tokens(input_ids[example_idx])

print(f"Tokens:\n{tokens}\n")
print(f"Input IDs:\n{input_ids[example_idx]}\n")
print(f"Attention Masks:\n{attention_masks[example_idx]}\n")
print(f"Tag IDs:\n{tag_ids[example_idx]}\n")

# Wydrukuj skojarzone z tokenami etykiety (dla lepszej czytelności)
tags = [list(tag2id.keys())[list(tag2id.values()).index(tag_id)] if tag_id != -100 else 'PAD' for tag_id in tag_ids[example_idx]]
print(f"Tags:\n{tags}\n")




from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    'allegro/herbert-base-cased',
    num_labels=len(tag2id)
)
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

# Przygotowanie TensorDataset
train_dataset = TensorDataset(input_ids, attention_masks, tag_ids)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # Możesz dostosować w zależności od zasobów
    sampler=RandomSampler(train_dataset)  # Mieszanie danych
)
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# Przenieś model na odpowiednie urządzenie (GPU, jeśli dostępne)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Pętla treningowa
model.train()
for epoch in range(3):  # Liczba epok
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attention_mask, b_labels = batch
        
        # Reset gradientów
        model.zero_grad()
        
        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        
        # Backward pass i optymalizacja
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")





# Zapisz model
model.save_pretrained('./saved_model')









#%%
def tag_tokens_with_ner_labels(data):
    tagged_data = []
    for text, annotations in data:
        # Założenie: tokens to lista tokenów z tekstu, np. za pomocą tokenizer.tokenize(text)
        tokens = text.split(' ')  # Przykład rozdzielenia na tokeny (uproszczony)
        
        # Inicjalizacja listy tagów jako 'O' dla każdego tokenu
        ner_tags = ['O'] * len(tokens)
        
        for start, end, label in annotations['entities']:
            if start == end:  # Przypadek dla jednotokenowych encji
                ner_tags[start] = f'B-{label}'
            else:
                ner_tags[start] = f'B-{label}'
                for i in range(start + 1, end):
                    ner_tags[i] = f'I-{label}'
        
        tagged_data.append((tokens, ner_tags))
    
    return tagged_data

tagged_data_train = tag_tokens_with_ner_labels(adjusted_data_train)
tagged_data_eval = tag_tokens_with_ner_labels(adjusted_data_eval)
import torch


def preprocess_for_model(tagged_data, tokenizer, label_map, max_length=512):
    input_ids = []
    attention_masks = []
    label_ids = []

    for tokens, labels in tagged_data:
        # Zakoduj tokeny do input_ids, dodając specjalne tokeny
        encoded = tokenizer.encode_plus(tokens, 
                                        is_split_into_words=True,
                                        add_special_tokens=True, 
                                        max_length=max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_attention_mask=True,
                                        return_tensors="pt")
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

        # Zakoduj etykiety
        labels_encoded = [label_map[label] for label in labels]
        # Uzupełnij etykiety dla specjalnych tokenów i paddingu
        labels_padded = labels_encoded + [-100] * (max_length - len(labels_encoded))
        label_ids.append(torch.tensor(labels_padded))

    # Konwersja list na tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    label_ids = torch.stack(label_ids)

    return input_ids, attention_masks, label_ids


input_ids_train, attention_masks_train, label_ids_train = preprocess_for_model(tagged_data_train, tokenizer, label_map)
input_ids_eval, attention_masks_eval, label_ids_eval = preprocess_for_model(tagged_data_eval, tokenizer, label_map)

sample_index = 0  # Możesz zmienić indeks, aby zobaczyć inne przykłady

# Konwersja input_ids z powrotem na tokeny
tokens = tokenizer.convert_ids_to_tokens(input_ids_train[sample_index])

# Przygotowanie tekstu etykiet dla wydruku
labels_text = [list(label_map.keys())[list(label_map.values()).index(label_id)] if label_id in label_map.values() else 'special_or_padding'
               for label_id in label_ids_train[sample_index].tolist()]

print("Tokeny:")
print(tokens)
print("\nMaski uwagi:")
print(attention_masks_train[sample_index].tolist())
print("\nIdentyfikatory etykiet:")
print(label_ids_train[sample_index].tolist())
print("\nTekst etykiet:")
print(labels_text)




from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(
    'allegro/herbert-base-cased',
    num_labels=len(label_map)  # liczba unikalnych etykiet w twoim zadaniu NER
)
train_dataset = NERDataset(input_ids_train, attention_masks_train, label_ids_train)
eval_dataset = NERDataset(input_ids_eval, attention_masks_eval, label_ids_eval)
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Ignoruj indeksy etykiet, które mają wartość -100 (specjalne tokeny i padding)
    valid_indices = (labels != -100)

    labels = labels[valid_indices]
    preds = preds[valid_indices]

    # Obliczanie metryk dla każdej etykiety
    label_metrics = {}
    for label in label_map.keys():
        label_indices = (labels == label_map[label])
        label_preds = preds[label_indices]
        label_labels = labels[label_indices]

        label_metrics[label + '_precision'] = precision_score(label_labels, label_preds, average='weighted', zero_division=1)
        label_metrics[label + '_recall'] = recall_score(label_labels, label_preds, average='weighted', zero_division=1)
        label_metrics[label + '_f1'] = f1_score(label_labels, label_preds, average='weighted')

    # Obliczanie metryk dla wszystkich etykiet
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_f1 = f1_score(labels, preds, average='micro')

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        **label_metrics
    }


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# funkcja do skracania danych

def find_nearest_acceptable_split_point(pos, text, total_length):
    """
    Finds the nearest split point that does not split words or sentences,
    preferring to split at punctuation followed by space or at natural sentence boundaries.
    """
    if pos <= 0:
        return 0
    if pos >= total_length:
        return total_length
    
    for offset in range(1, min(50, pos, total_length - pos) + 1):
        left = pos - offset
        right = pos + offset

        if left > 0 and text[left - 1] in '.?!' and text[left] == ' ':
            return left + 1

        if right < total_length and text[right - 1] in '.?!' and text[right] == ' ':
            return right + 1
    
    for offset in range(1, min(50, pos, total_length - pos) + 1):
        left = pos - offset
        right = pos + offset

        if left > 0 and text[left] == ' ':
            return left + 1

        if right < total_length and text[right] == ' ':
            return right + 1

    return pos

def split_text_around_entities_adjusted_for_four_parts(data_list):
    split_data = []
    
    for text, annotation in data_list:
        entities = sorted(annotation['entities'], key=lambda e: e[0])
        total_length = len(text)
        ideal_part_length = total_length // 5  # Adjusted for four parts
        
        split_points = [0]
        current_split = 0
        
        for _ in range(4):  # Adjusted to perform three splits for four parts
            proposed_split = current_split + ideal_part_length
            if proposed_split >= total_length:
                break
            
            adjusted_split = find_nearest_acceptable_split_point(proposed_split, text, total_length)
            
            for start, end, _ in entities:
                if adjusted_split > start and adjusted_split < end:
                    adjusted_split = end
                    break
            
            if adjusted_split != current_split:
                split_points.append(adjusted_split)
                current_split = adjusted_split
        
        split_points.append(total_length)
        
        last_split = 0
        for split in split_points[1:]:
            part_text = text[last_split:split].strip()
            part_entities = [(start - last_split, end - last_split, label) for start, end, label in entities if start >= last_split and end <= split]
            split_data.append((part_text, {'entities': part_entities}))
            last_split = split

    return split_data
transformed_data = split_text_around_entities_adjusted_for_four_parts(transformed_data)



#%%

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch


label_map = {'O': 0, 'B-PERSON': 1, 'I-PERSON': 2, 'B-LOC': 3}
max_length = 512  # Przykładowa maksymalna długość sekwencji

input_ids = []
attention_masks = []
label_ids = []

for tokens, labels in tagged_data:
    # Konwersja tokenów na identyfikatory tokenów z uwzględnieniem maksymalnej długości
    encoded_dict = tokenizer(tokens,
                              is_split_into_words=True,
                              add_special_tokens=True,
                              max_length=512,  # Maksymalna długość sekwencji
                              padding='max_length',
                              truncation=True,
                              return_attention_mask=True,
                              return_tensors='pt')
    
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])
    
    # Konwersja etykiet na indeksy
    label_index = [label_map[label] for label in labels] + [label_map['O']] * (max_length - len(labels))  # Dodajemy 'O' dla paddingu
    label_ids.append(torch.tensor(label_index))

# Konwersja list na tensor PyTorch
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
label_ids = torch.stack(label_ids)

example_index = 0  # Indeks przykładowych danych, które chcemy wyświetlić

# Dekodowanie identyfikatorów tokenów z powrotem na tokeny
tokens = tokenizer.convert_ids_to_tokens(input_ids[example_index])

# Przygotowanie listy etykiet na podstawie indeksów
inverse_label_map = {v: k for k, v in label_map.items()}  # Odwrócona mapa etykiet
labels = [inverse_label_map[label_id.item()] for label_id in label_ids[example_index]]

# Wyświetlenie danych
print("Tokeny:", tokens)
print("Maska uwagi:", attention_masks[example_index].tolist())
print("Etykiety:", labels)
#%%

def adjust_data_for_ner(data, tokenizer, max_length=512):
    adjusted_data = []

    for text, annotation in data:
        tokens = tokenizer.tokenize(text)
        entities = annotation['entities']
        current_chunk = []
        current_entities = []
        chunk_size = 0

        for entity in entities:
            start, end, label = entity
            entity_tokens = tokenizer.tokenize(text[start:end])
            entity_start = chunk_size + len(tokenizer.tokenize(text[:start]))
            entity_end = entity_start + len(entity_tokens)

            if entity_end <= max_length - 1:  # -1 to account for special tokens
                current_entities.append((entity_start, entity_end, label))
            else:
                # If the entity doesn't fit, start a new chunk
                if current_chunk:
                    adjusted_data.append((' '.join(current_chunk), {'entities': current_entities}))
                current_chunk = tokens[chunk_size:entity_start] + entity_tokens
                current_entities = [(0, len(entity_tokens), label)]
                chunk_size = entity_start

        if current_chunk or (chunk_size < len(tokens)):
            remaining_tokens = tokens[chunk_size:max_length - 1]
            adjusted_data.append((' '.join(current_chunk + remaining_tokens), {'entities': current_entities}))

    return adjusted_data

# Adjusting the data
adjusted_data = adjust_data_for_ner(transformed_data , tokenizer)

# Example output
for sample in adjusted_data[:2]:  # Just showing the first two samples for brevity
    print(sample)



def measure_token_count(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

# Example usage with your TRAIN_DATA before converting to BERT format
token_count_max=[]
for text, _ in transformed_data:
    token_count = measure_token_count(text, tokenizer)
    if token_count>500:
        token_count_max.append(token_count)
    print(f"Token count: {token_count}")
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, annotations = self.data[idx]
        tokens = self.tokenizer.tokenize(text)
        labels = [0] * len(tokens)  # Assuming 'O' (outside) is mapped to 0

        entities = annotations['entities']
        for start, end, label in entities:
            for i in range(start, end):
                if i < len(labels):
                    labels[i] = self.label_map[label]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(token_ids)

        # Padding
        padding_length = self.max_len - len(token_ids)
        token_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        labels += [0] * padding_length  # Assuming 'O' (outside) is mapped to 0

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Initialize your tokenizer


# Assuming your label map is something like this:
label_map = {'SPEKTAKL': 1, 'O': 0}  # Add all your labels here

# Create the dataset
dataset = NERDataset(adjusted_data, tokenizer, label_map)

# Example DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)



#%%
def adjust_entities(text, entities, tokenizer):
    tokenized_input = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = tokenized_input["offset_mapping"]
    
    adjusted_entities = []
    for start, end, label in entities['entities']:
        token_start = None
        token_end = None
        for idx, (start_pos, end_pos) in enumerate(offset_mapping):
            if start_pos == start and token_start is None:
                token_start = idx
            if end_pos == end:
                token_end = idx + 1  # Adjusting to include the end token
                break
        if token_start is not None and token_end is not None:
            adjusted_entities.append((token_start, token_end, label))
    
    return {"entities": adjusted_entities}

# Adjusting entities to align with token positions






def find_nearest_acceptable_split_point(pos, text, total_length):
    """
    Finds the nearest split point that does not split words or sentences,
    preferring to split at punctuation followed by space or at natural sentence boundaries.
    """
    if pos <= 0:
        return 0
    if pos >= total_length:
        return total_length
    
    for offset in range(1, min(50, pos, total_length - pos) + 1):
        left = pos - offset
        right = pos + offset

        if left > 0 and text[left - 1] in '.?!' and text[left] == ' ':
            return left + 1

        if right < total_length and text[right - 1] in '.?!' and text[right] == ' ':
            return right + 1
    
    for offset in range(1, min(50, pos, total_length - pos) + 1):
        left = pos - offset
        right = pos + offset

        if left > 0 and text[left] == ' ':
            return left + 1

        if right < total_length and text[right] == ' ':
            return right + 1

    return pos

def split_text_around_entities_adjusted_for_four_parts(data_list):
    split_data = []
    
    for text, annotation in data_list:
        entities = sorted(annotation['entities'], key=lambda e: e[0])
        total_length = len(text)
        ideal_part_length = total_length // 4  # Adjusted for four parts
        
        split_points = [0]
        current_split = 0
        
        for _ in range(3):  # Adjusted to perform three splits for four parts
            proposed_split = current_split + ideal_part_length
            if proposed_split >= total_length:
                break
            
            adjusted_split = find_nearest_acceptable_split_point(proposed_split, text, total_length)
            
            for start, end, _ in entities:
                if adjusted_split > start and adjusted_split < end:
                    adjusted_split = end
                    break
            
            if adjusted_split != current_split:
                split_points.append(adjusted_split)
                current_split = adjusted_split
        
        split_points.append(total_length)
        
        last_split = 0
        for split in split_points[1:]:
            part_text = text[last_split:split].strip()
            part_entities = [(start - last_split, end - last_split, label) for start, end, label in entities if start >= last_split and end <= split]
            split_data.append((part_text, {'entities': part_entities}))
            last_split = split

    return split_data

# Example usage with your transformed_data
transformed_data = split_text_around_entities_adjusted_for_four_parts(transformed_data)

def spacy_to_bert(data, tokenizer):
    bert_data = []
    for text, annotation in data:
        # Tokenize the text and get the offset mappings
        tokenized_input = tokenizer(text, return_offsets_mapping=True, is_split_into_words=False)
        tokens = tokenized_input.tokens()
        offsets = tokenized_input['offset_mapping']

        # Initialize labels for each token as 'O' (Outside)
        labels = ['O'] * len(tokens)

        # Iterate over the entity annotations
        for start, end, label in annotation['entities']:
            entity_started = False
            for idx, (start_offset, end_offset) in enumerate(offsets):
                # Skip special tokens
                if tokens[idx].startswith('<') and tokens[idx].endswith('>'):
                    continue
                
                # Check if the current token is within the entity range
                if start_offset >= start and end_offset <= end + 1:  # Adjust end position by one
                    if not entity_started:
                        if idx == 0:  # Check if the entity starts at position 0
                            labels[idx] = 'B-' + label  # Mark the beginning of an entity with 'B-'
                        else:
                            labels[idx] = 'I-' + label  # Mark tokens inside the entity with 'I-'
                        entity_started = True
                    else:
                        labels[idx] = 'I-' + label  # Mark tokens inside the entity with 'I-'
                elif entity_started:
                    break  # Break the loop once we have processed all tokens within the current entity

        bert_data.append((tokens, labels))
    return bert_data







# Convert the data
bert_ready_data[0]= spacy_to_bert(transformed_data[0], tokenizer)

for tokens, labels in bert_ready_data:
    print(tokens, labels)




import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Assuming bert_ready_data is already loaded as shown

class CustomDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_token_len=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        # Encoding the text
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_token_len, return_tensors="pt")
        
        # Your label processing here, depending on how they need to be adjusted to match token positions etc.
        labels = [int(label) for label in labels]  # Example conversion, adjust as needed
        
        # Here you need to ensure that `labels` matches the length of the input ids, which might involve expanding or truncating the labels list.
        # This step is crucial and depends on how your labels align with your tokens.

        item = {key: torch.tensor(val[0]) for key, val in encoding.items()}
        item['labels'] = torch.LongTensor(labels)  # Ensure labels are correctly aligned and formatted

        return item

# Initialize tokenizer


# Prepare the dataset
texts, labels = zip(*bert_ready_data)  # Assuming bert_ready_data is structured as [(text, label), ...]
dataset = CustomDataset(tokenizer, texts, labels)

# DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)  # Adjust `num_labels` as needed

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(loader))

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):  # Define num_epochs
    model.train()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Adjust the above code for any specific details of your dataset, such as custom label processing.







#%% FOR LONG TEXT 



from torch.utils.data import Dataset, DataLoader
import torch

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_length):
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.chunked_data = self.chunk_data(data)

    def chunk_data(self, data):
        chunked_data = []
        for text, original_labels in data:
            # Tokenize text with return_offsets_mapping to align labels later
            encoding = self.tokenizer(text, return_offsets_mapping=True, truncation=False, padding=False)
            tokens = encoding.tokens()

            labels = [self.label_to_id['O']] * len(tokens)
            for start_char, end_char, label in original_labels:
                for i, (start, end) in enumerate(encoding['offset_mapping']):
                    if start >= start_char and end <= end_char:
                        labels[i] = self.label_to_id[label]

            # Split the encoding into chunks
            input_ids_chunks = [encoding['input_ids'][i:i + self.max_length] for i in range(0, len(encoding['input_ids']), self.max_length)]
            attention_mask_chunks = [encoding['attention_mask'][i:i + self.max_length] for i in range(0, len(encoding['attention_mask']), self.max_length)]
            labels_chunks = [labels[i:i + self.max_length] for i in range(0, len(labels), self.max_length)]

            for input_ids_chunk, attention_mask_chunk, labels_chunk in zip(input_ids_chunks, attention_mask_chunks, labels_chunks):
                chunked_data.append((input_ids_chunk, attention_mask_chunk, labels_chunk))
        
        return chunked_data

    def __len__(self):
        return len(self.chunked_data)

    def __getitem__(self, idx):
        input_ids, attention_mask, labels = self.chunked_data[idx]
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Pad the sequences if they are shorter than `max_length`
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids_tensor = torch.cat([input_ids_tensor, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask_tensor = torch.cat([attention_mask_tensor, torch.zeros(padding_length, dtype=torch.long)])
            labels_tensor = torch.cat([labels_tensor, torch.full((padding_length,), self.label_to_id['O'], dtype=torch.long)])

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'labels': labels_tensor
        }    
    
    
    

#%%FOR SHORT TEXT   
from torch.utils.data import Dataset, DataLoader
import torch

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, labels = self.data[idx]
        # Tokenize text and map labels to IDs
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = [self.label_to_id[label] for label in labels]
        # Padding labels to match input ids length
        labels += [self.label_to_id['O']] * (self.max_length - len(labels))
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
# Example label mapping
label_to_id = {'O': 0, 'B-PERSON': 1, 'I-PERSON': 2, 'B-LOC': 3, 'I-LOC': 4}
# You should include all labels present in your dataset
max_length = 128  # Adjust based on your data
dataset = NERDataset(bert_ready_data, tokenizer, label_to_id, max_length)
from torch.utils.data import DataLoader

batch_size = 4
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
