# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:41:52 2024

@author: darek
"""

from pymarc import MARCReader
import pandas as pd
from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, TrainingArguments, Trainer
from tqdm import tqdm

# Ścieżka do pliku MARC21
MARC_FILE_PATH =  "D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc"

def extract_marc_data(file_path, limit=100):
    records = []
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for i, record in enumerate(tqdm(reader, desc="Processing MARC records", unit="record")):
            if i >= limit:
                break
            title = record['245']['a'] if '245' in record and 'a' in record['245'] else None
            subjects = [field['a'] for field in record.get_fields('650') if 'a' in field]
            if title:
                records.append({"title": title.strip(), "subjects": subjects})
    return records

# Wczytanie danych z MARC21
marc_records = extract_marc_data(MARC_FILE_PATH, limit=100)

def generate_questions_and_answers(records):
    question_answer_pairs = []

    for record in records:
        title = record["title"]
        subjects = record["subjects"]

        # Pytania i odpowiedzi dla tytułów
        question_answer_pairs.append({
            "input_text": f"Kto napisał książkę '{title}'?",
            "target_text": f"Książka '{title}' została napisana przez autora wymienionego w naszych danych."
        })
        question_answer_pairs.append({
            "input_text": f"Czy książka '{title}' jest dostępna w naszej bibliotece?",
            "target_text": f"Tak, książka '{title}' jest dostępna w naszej bibliotece."
        })

        # Pytania i odpowiedzi dla tematów
        for subject in subjects:
            question_answer_pairs.append({
                "input_text": f"Jakie książki dotyczą tematyki {subject}?",
                "target_text": f"W naszej bazie znajdują się książki, które dotyczą tematyki '{subject}'."
            })
            question_answer_pairs.append({
                "input_text": f"Czy macie książki o tematyce {subject}?",
                "target_text": f"Tak, posiadamy książki o tematyce '{subject}'."
            })

    return question_answer_pairs

# Generowanie pytań i odpowiedzi
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
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, TrainerCallback, MT5ForConditionalGeneration

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

# Obliczenie liczby kroków
steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
total_steps = steps_per_epoch * training_args.num_train_epochs

# Callback z tqdm
class TqdmProgressCallback(TrainerCallback):
    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=self.total_steps, desc="Training Progress", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()

# Trenowanie modelu z callbackiem
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.add_callback(TqdmProgressCallback(total_steps=total_steps))

# Rozpoczęcie treningu
trainer.train()


# Zapisanie modelu i tokenizera
trainer.save_model("./saved_model")
tokenizer.save_pretrained("./saved_model")
