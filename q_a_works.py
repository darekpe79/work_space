# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:45:36 2024

@author: dariu
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Załaduj model i tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_answer(question, context):
    # Utwórz tekst wejściowy dla modelu
    input_text = f"question: {question} context: {context}"
    
    # Tokenizacja
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    # Generowanie odpowiedzi
    outputs = model.generate(inputs.input_ids, max_length=50, num_beams=4, early_stopping=True)
    
    # Dekodowanie odpowiedzi
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# Przykładowe pytanie i kontekst
question = "Kto jest stwórcą 'Pana Tadeusza'?"
context = "Pan Tadeusz, epopeja narodowa, Książka jest epopeją narodową napisaną przez Adama Mickiewicza."

# Generowanie odpowiedzi
answer = generate_answer(question, context)
print(f"Pytanie: {question}\nOdpowiedź: {answer}\n")

#"dajemy answer w zapytanie Marcina"
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Załaduj model i tokenizer
model_name = "google/mt5-base"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)

def generate_answer(question, context):
    # Utwórz tekst wejściowy dla modelu
    input_text = f"question: {question} context: {context}"
    
    # Tokenizacja
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    # Generowanie odpowiedzi
    outputs = model.generate(inputs.input_ids, max_length=50, num_beams=4, early_stopping=True)
    
    # Dekodowanie odpowiedzi
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# Przykładowe pytanie i kontekst
question = "Kto jest stwórcą 'Pana Tadeusza'?"
context = "Pan Tadeusz, epopeja narodowa, Książka jest epopeją narodową napisaną przez Adama Mickiewicza."

# Generowanie odpowiedzi
answer = generate_answer(question, context)
print(f"Pytanie: {question}\nOdpowiedź: {answer}\n")

from pymarc import MARCReader
import pandas as pd
from tqdm import tqdm

# Funkcja do czyszczenia tytułu
def clean_title(title):
    return title.rstrip(' /:')  # Usuwa znaki '/', ':', ' ' (spacja) z końca tytułu

# Funkcja do ekstrakcji danych z rekordów MARC21 z licznikiem tqdm
def extract_data_from_marc(file_path):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        reader_list = list(reader)
        for record in tqdm(reader_list, desc="Processing MARC records"):
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else None
            if title and author:
                clean_title_text = clean_title(title)
                question = f"Kto napisał {clean_title_text}?"
                context = f"napisał {author}"
                answer = author
                records.append((question, context, answer))
    return records

# Przykładowe użycie
records = extract_data_from_marc('D:/Nowa_praca/08082023-Czarek_BN_update/libri_marc_bn_books_2023-08-07.mrc')

# Tworzenie DataFrame
df = pd.DataFrame(records, columns=['question', 'context', 'answer'])

# Tworzenie kolumny input_text
df['input_text'] = df['question'] + " context: " + df['context']

# Tworzenie kolumny target_text
df['target_text'] = df['answer']

# Wyświetlenie pierwszych kilku wierszy
print(df.head())

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Krok 1: Tokenizacja danych
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def tokenize_and_encode(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Konwersja DataFrame do Dataset
dataset = Dataset.from_pandas(df)

# Tokenizacja datasetu
tokenized_dataset = dataset.map(tokenize_and_encode, batched=True)

# Krok 2: Przygotowanie datasetu do trenowania
# Podział na zbiór treningowy i walidacyjny
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Krok 3: Trenowanie modelu
model = T5ForConditionalGeneration.from_pretrained("t5-base")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Krok 4: Zapisywanie i testowanie modelu
model.save_pretrained("./trained_model_DARULEX")
tokenizer.save_pretrained("./trained_model_DARULEX")

# Funkcja do generowania odpowiedzi
def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(inputs.input_ids, max_length=50, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Przykładowe pytanie do testowania
question = "Kto napisał Pan Tadeusz?"
context = "napisał Adam Mickiewicz"

answer = generate_answer(question, context)
print(f"Pytanie: {question}\nOdpowiedź: {answer}\n")

