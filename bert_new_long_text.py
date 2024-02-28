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
    marked_text = re.sub(title_pattern, r"[SPEKTAKL]\g<0>[/SPEKTAKL]", text, flags=re.IGNORECASE)
    return marked_text

df2.dropna(subset=['Tytuł spektaklu'], inplace=True)

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
# Function to convert spaCy entity format to token-level labels for BERT

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
transformed_data = split_text_around_entities_adjusted_for_four_parts(transformed_data)
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

def measure_token_count(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

# Example usage with your TRAIN_DATA before converting to BERT format
for text, _ in transformed_data:
    token_count = measure_token_count(text, tokenizer)
    print(f"Token count: {token_count}")


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
