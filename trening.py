# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:03:18 2024

@author: dariu
"""

from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer
texts = ["To był zwykły dzień, pełen rutynowych zadań i małych radości.", 
         "Gwiazdy mrugały na nocnym niebie, a księżyc w pełni świecił jasno.","Czarek to ulubiony kolega Marysi"]
genres = ["proza", "poezja","proza"]


# Inicjalizacja tokenizera
from transformers import HerbertTokenizerFast

# Initialize the tokenizer with the Polish model
tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-large-cased')

# Kodowanie etykiet
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(genres)  # Zakodowane etykiety: np. [0, 1]
num_labels = len(label_encoder.classes_) 
# Tokenizacja tekstu i przygotowanie tensorów danych wejściowych i etykiet
input_ids = []
attention_masks = []
for text in texts:
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    input_ids.append(inputs['input_ids'])
    attention_masks.append(inputs['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(encoded_labels)

# Teraz mamy input_ids, attention_masks i labels gotowe do użycia w treningu
from torch.utils.data import TensorDataset, DataLoader

# Tworzenie TensorDataset z input_ids, attention_masks, i labels
dataset = TensorDataset(input_ids, attention_masks, labels)

batch_size = 8  # Możesz dostosować rozmiar batcha do swoich potrzeb

# Tworzenie DataLoadera
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
from transformers import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-base-cased",
    num_labels=num_labels,  # Przyjmijmy, że `label_map` to twoje mapowanie etykiet
    problem_type="single_label_classification"
)

# Przykład konfiguracji optymalizatora
optimizer = AdamW(model.parameters(), lr=5e-5)

# Przenieś model na odpowiednie urządzenie (GPU lub CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Pętla treningowa
model.train()  # Ustaw model w tryb treningowy
for epoch in range(num_epochs):  # num_epochs to liczba epok, którą chcesz przeprowadzić
    for batch in data_loader:
        # Wyciągnięcie danych z batcha i przeniesienie na urządzenie
        b_input_ids, b_attention_mask, b_labels = tuple(t.to(device) for t in batch)
        
        # Zerowanie gradientów modelu
        optimizer.zero_grad()
        
        # Przeprowadzenie forward pass (obliczenie logitów przez model)
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        
        # Obliczenie straty
        loss = outputs.loss
        
        # Backpropagation (obliczenie gradientów)
        loss.backward()
        
        # Aktualizacja wag modelu
        optimizer.step()
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


from datasets import Dataset

texts = ["To był zwykły dzień, pełen rutynowych zadań i małych radości.", 
         "Gwiazdy mrugały na nocnym niebie, a księżyc w pełni świecił jasno.","Czarek to ulubiony kolega Marysi"]
genres = ["proza", "poezja","proza"]

# Inicjalizacja tokenizera
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Kodowanie etykiet
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(genres)  # Zakodowane etykiety: np. [0, 1]
num_labels = len(label_encoder.classes_) 

# Tworzenie DataFrame
data = {"text": texts, "labels": encoded_labels}

# Tworzenie obiektu Dataset
dataset = Dataset.from_dict(data)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Inicjalizacja tokenizatora
tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-large-cased')

# Inicjalizacja modelu
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-large-cased",
    num_labels=num_labels,  # Przyjmijmy, że `label_map` to twoje mapowanie etykiet
    problem_type="single_label_classification"
)

# Funkcja do tokenizacji
def tokenize_and_encode(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=20)

# Tokenizacja i kodowanie danych
dataset = dataset.map(tokenize_and_encode, batched=True)  # batched true - większe partie danych wchodzą do mapowania

# Podział na zbiór treningowy i walidacyjny (przykład: 80% trening, 20% walidacja)
train_test_dataset = dataset.train_test_split(test_size=0.2, seed=23)


# Mamy teraz obiekty dataset gotowe do treningu i walidacji
first_record = dataset[0]
input_ids = first_record['input_ids']
attention_mask = first_record['attention_mask']
labels = first_record['labels']
print("Stokenizowany tekst:", tokenizer.decode(input_ids))

from transformers import Trainer, TrainingArguments

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    no_cuda=True  # Opcja zmuszająca do korzystania z CPU zamiast GPU
)
from sklearn.metrics import accuracy_score
# Inicjalizacja trenera z dodatkową funkcją obliczającą metrykę dokładności
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_dataset['train'],
    eval_dataset=train_test_dataset['test'],
    compute_metrics=compute_metrics
)

# Rozpoczęcie procesu treningowego
trainer.train()

# Ewaluacja modelu na zbiorze walidacyjnym
results = trainer.evaluate()

# Wypisanie wyników ewaluacji
print(results)



from transformers import AutoTokenizer

# Załadowanie tokenizera dla konkretnego modelu
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")

# Przykładowy tekst
text = "Here is some example text to be tokenized."

# Tokenizacja i kodowanie tekstu
encoded_input = tokenizer.encode_plus(
    text,
    max_length=25,
    padding='max_length',
    truncation=True,
    return_offsets_mapping=True,
    return_tensors="pt"
)

print(encoded_input)

offset_mapping = encoded_input["offset_mapping"]

tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])

# Drukowanie tokenów
print("\nTokeny:")
for token in tokens[:15]:
    print(token)

# Usuwamy tensor i przekształcamy go na listę par (początek, koniec) dla każdego tokenu
offsets = offset_mapping.squeeze().tolist()

# Drukowanie oryginalnego tekstu z podziałem na tokeny
print("Stokenizowany tekst z podziałem na początek i koniec słowa:")
for i, (start, end) in enumerate(offsets):
    # Dekodowanie tokenu na tekst; pomijamy specjalne tokeny
    if start == end:  # Dla tokenów specjalnych, np. [CLS], [SEP], [PAD]
        continue
    token_str = text[start:end]
    print(f"Token {i}: {token_str} (Start: {start}, Koniec: {end})")