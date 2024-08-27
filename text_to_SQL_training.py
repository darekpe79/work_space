# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:38:43 2024

@author: dariu
"""

import pandas as pd
from pymarc import MARCReader
from tqdm import tqdm
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import random


question_forms_author = [
    "jakie książki napisał",
    "jakie książki stworzył",
    "jakich książek jest autorem",
]


question_forms_subject = [
    "jakie książki dotyczą",
    "jakie książki są o tematyce",
    "jakie książki opisują"
]


def extract_data_from_marc(file_path):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for record in tqdm(reader, desc="Processing MARC records", unit="record"):
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else None
            subjects = record.get_fields('650')  # Zbieramy wszystkie pola 650
            
            # Generowanie pytań o autorów
            if author:
                for question_form in question_forms_author:
                    question = f"{question_form} {author}?"
                    answer = f"SELECT title FROM books WHERE author = '{author}';"
                    records.append((question, answer))
            
            # Generowanie pytań o tematykę
            for subject in subjects:
                subject_text = subject['a'] if 'a' in subject else None
                if subject_text:
                    #print (subject_text)
                    for question_form in question_forms_subject:
                        question = f"{question_form} {subject_text}?"
                        answer = f"SELECT title FROM books WHERE subject = '{subject_text}';"
                        records.append((question, answer))
                        
    return records


records = extract_data_from_marc('C:/Users/dariu/trained_modelQA/libri_marc_bn_books_2023-08-07.mrc')

# Tworzenie DataFrame
df = pd.DataFrame(records, columns=['question', 'answer'])


df = df.head(10000)

print(f"Liczba rekordów: {len(df)}")

# Tworzenie kolumny input_text
df['input_text'] = df['question']

# Tworzenie kolumny target_text
df['target_text'] = df['answer']

# Wyświetlenie pierwszych kilku wierszy
print("Dane wejściowe po przetworzeniu:")
print(df[['input_text', 'target_text']].head())

# Krok 1: Tokenizacja danych
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

def tokenize_and_encode(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Debugowanie: Sprawdzenie pustych rekordów
    for i in range(len(examples['input_text'])):
        if len(model_inputs['input_ids'][i]) == 0 or len(model_inputs['labels'][i]) == 0:
            print(f"Empty input or label found at index {i}")
            print(f"Input: {examples['input_text'][i]}")
            print(f"Label: {examples['target_text'][i]}")
    
    return model_inputs

# Konwersja DataFrame do Dataset
dataset = Dataset.from_pandas(df)

# Tokenizacja datasetu
tokenized_dataset = dataset.map(tokenize_and_encode, batched=True)

# Sprawdzenie liczby rekordów po tokenizacji
print(f"Liczba rekordów po tokenizacji: {len(tokenized_dataset)}")

# Krok 2: Przygotowanie datasetu do trenowania
# Podział na zbiór treningowy i walidacyjny
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Krok 3: Trenowanie modelu z małym learning rate
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,  # Zwiększona liczba epok
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=1e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Krok 4: Zapisywanie modelu
model.save_pretrained("C:/Users/User/Desktop/trained_modelQA")
tokenizer.save_pretrained("C:/Users/User/Desktop/trained_modelQA")
