# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:48:49 2024

@author: darek
"""

import os
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import HerbertTokenizerFast, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
# Ustawienia dla tokenizerów
tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-large-cased')
device = torch.device("cpu")  # Wymuszenie użycia CPU

# Ładowanie danych JSON
json_files_dir = 'E:/Python/json_trening/'
json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]

labels_to_remove ={'WYDAWNICTWO'}
transformed_data = []

# Wczytanie danych i przetworzenie adnotacji
for json_file in json_files:
    file_path = os.path.join(json_files_dir, json_file)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
        for item in json_data['annotations']:
            text = item[0]
            entities = item[1]['entities']
            
            filtered_entities = [
                (start, end, label) 
                for start, end, label in entities 
                if label not in labels_to_remove
            ]
            
            if filtered_entities:
                transformed_data.append((text, {'entities': filtered_entities}))

# Statystyki encji
entity_counts = defaultdict(int)
for _, annotation in transformed_data:
    for _, _, label in annotation['entities']:
        entity_counts[label] += 1     
entity_stats = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)

# Podział na zbiory treningowe i ewaluacyjne
transformed_data_train, transformed_data_eval = train_test_split(transformed_data, test_size=0.1, random_state=42)

# Funkcja przetwarzająca dane do formatu `datasets`
def prepare_data(data, tokenizer, tag2id, max_length=514):
    texts = [text for text, _ in data]
    annotations = [annotation['entities'] for _, annotation in data]

    # Tokenizowanie całej listy tekstów jednocześnie
    tokenized_inputs = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    input_ids = tokenized_inputs["input_ids"]
    attention_masks = tokenized_inputs["attention_mask"]
    offset_mappings = tokenized_inputs["offset_mapping"]

    labels = []

    for offsets, entities in zip(offset_mappings, annotations):
        offsets = offsets.tolist()[1:-1]
        sequence_labels = ['O'] * len(offsets)

        for start, end, label in entities:
            entity_start_index = None
            entity_end_index = None

            for idx, (offset_start, offset_end) in enumerate(offsets):
                if start == offset_start or (start > offset_start and start < offset_end):
                    entity_start_index = idx
                if end > offset_start and end <= offset_end:
                    entity_end_index = idx
                    break

            if entity_start_index is not None and entity_end_index is not None:
                sequence_labels[entity_start_index] = f'B-{label}'
                for i in range(entity_start_index + 1, entity_end_index + 1):
                    sequence_labels[i] = f'I-{label}'

        full_sequence_labels = ['O'] + sequence_labels + ['O'] * (max_length - len(sequence_labels) - 1)
        label_ids = [tag2id.get(label, tag2id['O']) for label in full_sequence_labels]
        labels.append(label_ids)

    return input_ids, attention_masks, torch.tensor(labels, dtype=torch.long)
def find_longest_text(data, tokenizer):
    max_length = 0
    longest_text = ""
    for text, _ in data:
        token_length = len(tokenizer(text, truncation=False, return_tensors="pt")["input_ids"][0])
        if token_length > max_length:
            max_length = token_length
            longest_text = text
    return longest_text, max_length

longest_text, max_length_tokens = find_longest_text(transformed_data, tokenizer)

# Liczba znaków w najdłuższym tekście
text_length_characters = len(longest_text)

# Wyświetlenie wyników
print(f"Tekst o maksymalnej liczbie tokenów ({max_length_tokens}):")
print(longest_text)
print(f"Liczba znaków: {text_length_characters}")
# Przygotowanie mapowania tagów
tag2id = {
    'O': 0,
    'B-MIEJSCE WYDANIA': 1, 'I-MIEJSCE WYDANIA': 2,
    'B-TYTUŁ': 3, 'I-TYTUŁ': 4,
    'B-DATA': 5, 'I-DATA': 6,
    'B-STRONY': 7, 'I-STRONY': 8,
    'B-WSPÓŁTWÓRCA': 9, 'I-WSPÓŁTWÓRCA': 10,
    'B-AUTOR': 11, 'I-AUTOR': 12,
    'B-FUNKCJA WSPÓŁTWÓRSTWA': 13, 'I-FUNKCJA WSPÓŁTWÓRSTWA': 14,
    'B-DOPISEK BIBLIOGRAFICZNY': 15, 'I-DOPISEK BIBLIOGRAFICZNY': 16, 'B-TOM':17,'I-TOM':18
}
id2tag = {v: k for k, v in tag2id.items()}

# Przygotowanie zbiorów danych
def convert_to_dataset(data, tokenizer, tag2id, max_length=514):
    inputs, masks, labels = prepare_data(data, tokenizer, tag2id, max_length)
    return Dataset.from_dict({
        "input_ids": inputs.tolist(),
        "attention_mask": masks.tolist(),
        "labels": labels.tolist()
    })

train_dataset = convert_to_dataset(transformed_data_train, tokenizer, tag2id)
eval_dataset = convert_to_dataset(transformed_data_eval, tokenizer, tag2id)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": eval_dataset
})

def print_example_from_dataset(dataset, tokenizer, example_index=0):
    input_ids = dataset[example_index]["input_ids"]
    attention_mask = dataset[example_index]["attention_mask"]
    labels = dataset[example_index]["labels"]

    # Konwersja input_ids do tokenów
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Mapowanie etykiet do nazw
    readable_labels = [id2tag[label] if label in id2tag else "PAD" for label in labels]

    print("\n=== Przykład danych ===")
    print("Pełny tekst (zrekonstruowany):")
    reconstructed_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    print(reconstructed_text)
    print("\nTokeny, etykiety i maski uwagi:")
    print(f"{'Token':<15} {'Label':<25} {'Attention Mask':<15}")
    print("-" * 55)

    for token, label, mask in zip(tokens, readable_labels, attention_mask):
        if mask == 1:  # Wyświetlaj tylko te tokeny, które mają maskę uwagi równą 1
            print(f"{token:<15} {label:<25} {mask:<15}")

# Wydruk przykładowych danych z train_dataset
print_example_from_dataset(train_dataset, tokenizer, example_index=0)

# Przygotowanie modelu
model = AutoModelForTokenClassification.from_pretrained(
    'allegro/herbert-large-cased',
    num_labels=len(tag2id)
)
model.to(device)  # Przeniesienie modelu na CPU
# Argumenty treningowe
training_args = TrainingArguments(
    output_dir="/model/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    weight_decay=1e-2,
    per_device_train_batch_size=8,  # W razie potrzeby zmniejsz batch size dla CPU
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="/content/model/",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_avg_f1",
    report_to="none",
    no_cuda=True  # Wymuszenie CPU w TrainingArguments
)
# Funkcja metryk
from seqeval.metrics import classification_report

def compute_metrics_per_entity(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=2)

    true_labels = [
        [id2tag[label] for label in sentence if label != tag2id['O']] 
        for sentence in labels
    ]
    pred_labels = [
        [id2tag[pred] for pred, label in zip(sentence, sentence_labels) if label != tag2id['O']]
        for sentence, sentence_labels in zip(preds, labels)
    ]

    # Generowanie szczegółowego raportu
    report = classification_report(true_labels, pred_labels, output_dict=True)

    # Wydrukowanie wyników dla każdego typu encji
    print("\n=== Metryki dla każdego typu encji ===")
    for entity, scores in report.items():
        if entity in ['macro avg', 'weighted avg']:
            continue
        print(f"{entity}:")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall: {scores['recall']:.4f}")
        print(f"  F1-score: {scores['f1-score']:.4f}")
        print(f"  Support: {scores['support']}")

    # Dodanie prefiksu eval_ do monitorowanej metryki
    return {
        "eval_macro_avg_f1": report["macro avg"]["f1-score"]
    }

from transformers import EarlyStoppingCallback
# Parametry early stopping
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Liczba epok bez poprawy
    early_stopping_threshold=0.01  # Minimalna zmiana metryki
)

# Trener z callbackiem
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_per_entity,  # Nowa funkcja metryk
    callbacks=[early_stopping_callback]
)

# Trening
trainer.train()
model.config.label2id = tag2id
model.config.id2label = {v: k for k, v in tag2id.items()}
# Zapisanie modelu, tokenizera i mapowania tagów

trainer.save_model("/model/caly/")
tokenizer.save_pretrained("/model/caly/")

with open("/model/caly/tag2id.json", 'w') as f:
    json.dump(tag2id, f)
import transformers
print(transformers.__version__)