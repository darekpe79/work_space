# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:06:01 2025

@author: darek
"""
from pymarc import MARCReader
import pandas as pd
from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, TrainingArguments, Trainer
from tqdm import tqdm

# Ścieżka do pliku MARC21
MARC_FILE_PATH = "D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc"

# Funkcja do ekstrakcji danych z MARC21
def extract_marc_data(file_path, limit=100):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Processing MARC records", unit="record")):
            if i >= limit:
                break
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            author = record['100']['a'] if '100' in record and 'a' in record['100'] else "Nieznany autor"
            if title:
                records.append({"title": title.strip(), "subjects": subjects, "author": author})
    return records

# Wczytanie danych z MARC21
marc_records = extract_marc_data(MARC_FILE_PATH, limit=100)

# Generowanie pytań i odpowiedzi
def generate_questions_and_answers(records):
    question_answer_pairs = []

    for record in records:
        title = record["title"]
        subjects = record["subjects"]
        author = record["author"]

        # Pytania o autora
        question_answer_pairs.append({
            "input_text": f"Kto napisał książkę '{title}'?",
            "target_text": f"[TITLE] Książka '{title}' została napisana przez autora: {author}."   #model szuka w GDZIE indeksie tytułóW- CZEGO string tytułu NER?, CO ZWRACA -autora 
        })

        # Pytania o dostępność książki
        question_answer_pairs.append({
            "input_text": f"Czy książka '{title}' jest dostępna w naszej bazie?",
            "target_text": f"[TITLE] Tak, książka '{title}' jest dostępna w naszej bibliotece." #GDZIE TITLE czego string "tytul" output true false
        })

        # Pytania o tematykę
        for subject in subjects:
            question_answer_pairs.append({
                "input_text": f"Jakie książki dotyczą tematyki {subject}?",
                "target_text": f"[SUBJECT] W naszej bazie znajdują się książki, które dotyczą tematyki '{subject}'."
            })
            question_answer_pairs.append({
                "input_text": f"Czy macie książki o tematyce {subject}?",
                "target_text": f"[SUBJECT] Tak, posiadamy książki o tematyce '{subject}'."
            })

    return question_answer_pairs

# Generowanie danych treningowych
training_data = generate_questions_and_answers(marc_records)

# Konwersja do DataFrame
training_df = pd.DataFrame(training_data)
print(training_df.head())

# Inicjalizacja tokenizera
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large")

def tokenize_and_encode(examples):
    model_inputs = tokenizer(
        examples['input_text'], max_length=512, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['target_text'], max_length=512, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Konwersja danych do Dataset
dataset = Dataset.from_pandas(training_df)

# Tokenizacja
tokenized_dataset = dataset.map(tokenize_and_encode, batched=True)

# Podział na zbiór treningowy i walidacyjny
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"Rekordy w zbiorze treningowym: {len(train_dataset)}")
print(f"Rekordy w zbiorze walidacyjnym: {len(eval_dataset)}")

# Inicjalizacja modelu
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")

# Ustawienia treningu
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    save_total_limit=3
)

# Trenowanie modelu
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# Zapisanie modelu i tokenizera
trainer.save_model("./saved_model")
tokenizer.save_pretrained("./saved_model")

