# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:24:31 2024

@author: dariu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:16:30 2024

@author: User
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import HerbertTokenizerFast, AutoModelForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from datasets import Dataset

# Initialize the tokenizer with the Polish model
tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-large-cased')

# Load JSON data and preprocess
json_files_dir = 'D:/Nowa_praca/adnotacje_spubi/anotowane/'
json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]
labels_to_remove = {'TOM', 'WYDAWNICTWO'}

transformed_data = []
for json_file in json_files:
    file_path = os.path.join(json_files_dir, json_file)
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        for item in json_data['annotations']:
            text = item[0]
            entities = [
                (start, end, label) for start, end, label in item[1]['entities'] if label not in labels_to_remove
            ]
            if entities:
                transformed_data.append((text, {'entities': entities}))

# Define entity statistics
entity_counts = defaultdict(int)
for _, annotation in transformed_data:
    for _, _, label in annotation['entities']:
        entity_counts[label] += 1
entity_stats = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)

# Split data into train and eval
transformed_data_train, transformed_data_eval = train_test_split(transformed_data, test_size=0.1, random_state=42)

# Prepare data function
def prepare_data(data, tokenizer, tag2id, max_length=256):
    input_ids, attention_masks, labels = [], [], []

    for text, annotation in data:
        tokenized_input = tokenizer.encode_plus(
            text, max_length=max_length, padding='max_length', truncation=True,
            return_offsets_mapping=True, return_tensors="pt"
        )
        input_id = tokenized_input['input_ids'].squeeze().tolist()
        attention_mask = tokenized_input['attention_mask'].squeeze().tolist()
        offset_mapping = tokenized_input['offset_mapping'].squeeze().tolist()

        sequence_labels = ['O'] * len(offset_mapping)
        for start, end, label in annotation['entities']:
            for idx, (offset_start, offset_end) in enumerate(offset_mapping):
                if idx == 0 or idx == len(offset_mapping) - 1:
                    continue
                if start == offset_start or (start > offset_start and start < offset_end):
                    sequence_labels[idx] = f'B-{label}'
                if end > offset_start and end <= offset_end:
                    sequence_labels[idx] = f'I-{label}'

        label_ids = [
            tag2id.get(lbl, tag2id['O']) if attention_mask[i] == 1 else -100
            for i, lbl in enumerate(sequence_labels)
        ]

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label_ids)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# Define tag mappings
tag2id = {
    'O': 0,
    'B-MIEJSCE WYDANIA': 1, 'I-MIEJSCE WYDANIA': 2,
    'B-TYTUŁ': 3, 'I-TYTUŁ': 4,
    'B-DATA': 5, 'I-DATA': 6,
    'B-STRONY': 7, 'I-STRONY': 8,
    'B-WSPÓŁTWÓRCA': 9, 'I-WSPÓŁTWÓRCA': 10,
    'B-AUTOR': 11, 'I-AUTOR': 12,
    'B-FUNKCJA WSPÓŁTWÓRSTWA': 13, 'I-FUNKCJA WSPÓŁTWÓRSTWA': 14,
    'B-DOPISEK BIBLIOGRAFICZNY': 15, 'I-DOPISEK BIBLIOGRAFICZNY': 16
}
id2tag = {v: k for k, v in tag2id.items()}

# Prepare datasets
input_ids, attention_masks, labels = prepare_data(transformed_data_train, tokenizer, tag2id)
input_ids_eval, attention_masks_eval, labels_eval = prepare_data(transformed_data_eval, tokenizer, tag2id)

texts_train = [data[0] for data in transformed_data_train]
texts_eval = [data[0] for data in transformed_data_eval]

train_dataset = Dataset.from_dict({
    'input_ids': input_ids.tolist(),
    'attention_mask': attention_masks.tolist(),
    'labels': labels.tolist(),
    'text': texts_train
})
eval_dataset = Dataset.from_dict({
    'input_ids': input_ids_eval.tolist(),
    'attention_mask': attention_masks_eval.tolist(),
    'labels': labels_eval.tolist(),
    'text': texts_eval
})

# Define model
model = AutoModelForTokenClassification.from_pretrained(
    'allegro/herbert-large-cased',
    num_labels=len(tag2id)
).to('cuda' if torch.cuda.is_available() else 'cpu')

# Class weights
class_weights = np.array([
    1.0 / (entity_counts[label] + 1e-6) if label in entity_counts else 1.0
    for label in tag2id.keys()
])
class_weights = torch.tensor(class_weights / class_weights.sum() * len(tag2id), dtype=torch.float)

# Custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# Define Trainer arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    save_total_limit=2
)

# Trainer
trainer = CustomTrainer(
    class_weights=class_weights.to('cuda' if torch.cuda.is_available() else 'cpu'),
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=lambda p: {
        'precision': precision_score(*[list(x) for x in zip(*[
            ([id2tag.get(label, 'O') for label in label_ids if label != -100],
             [id2tag.get(pred, 'O') for pred in predictions])
            for predictions, label_ids in zip(
                np.argmax(p.predictions, axis=2), p.label_ids
            )
        ])]),
        'recall': recall_score(*[list(x) for x in zip(*[
            ([id2tag.get(label, 'O') for label in label_ids if label != -100],
             [id2tag.get(pred, 'O') for pred in predictions])
            for predictions, label_ids in zip(
                np.argmax(p.predictions, axis=2), p.label_ids
            )
        ])]),
        'f1': f1_score(*[list(x) for x in zip(*[
            ([id2tag.get(label, 'O') for label in label_ids if label != -100],
             [id2tag.get(pred, 'O') for pred in predictions])
            for predictions, label_ids in zip(
                np.argmax(p.predictions, axis=2), p.label_ids
            )
        ])])
    }
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("D:/Nowa_praca/adnotacje_spubi/model/")
tokenizer.save_pretrained("D:/Nowa_praca/adnotacje_spubi/model/")
