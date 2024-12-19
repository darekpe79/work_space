# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:50:57 2024

@author: dariu
"""

from datasets import load_dataset

# Załadowanie zbioru danych Open-Orca
dataset = load_dataset("Open-Orca", "OpenOrca")

# Wyświetlenie przykładu danych
print(dataset['train'][0])



import pandas as pd

# Przykładowe dane
data = {
    'Pytanie': ["Kto napisał 'Pan Tadeusz'?", "Jaki jest główny temat 'Dziadów'?"],
    'Tytuł': ["Pan Tadeusz", "Dziady"],
    'Hasła przedmiotowe': ["epopeja narodowa", "dramat romantyczny"],
    'Opis': ["Książka jest epopeją narodową napisaną przez Adama Mickiewicza.",
             "Dramat romantyczny Adama Mickiewicza, który opowiada o duchach i historii Polski."],
    'Odpowiedź': ["Adam Mickiewicz", "Główny temat to walka o wolność i duchowe zmartwychwstanie narodu."]
}

df = pd.DataFrame(data)

# Tworzenie kolumny input_text
df['input_text'] = "question: " + df['Pytanie'] + " context: " + df['Tytuł'] + ", " + df['Hasła przedmiotowe'] + ", " + df['Opis']

# Tworzenie kolumny target_text
df['target_text'] = df['Odpowiedź']

# Wyświetlenie przygotowanych danych
print(df[['input_text', 'target_text']])
from transformers import T5Tokenizer
from datasets import Dataset

tokenizer = T5Tokenizer.from_pretrained("t5-base")

def tokenize_and_encode(examples):
    # Tokenizacja tekstu wejściowego
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    
    # Tokenizacja odpowiedzi i przypisanie jako 'labels'
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]  # Przypisanie zakodowanej odpowiedzi do 'labels' w danych wejściowych
    return model_inputs


# Przetwarzanie DataFrame do Dataset i tokenizacja
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_encode, batched=True)
#%% PARAFRAZY AUGMENTACJA
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Wczytanie modelu
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tekst wejściowy
text = "What is the best way to train a machine learning model?"

# Tokenizacja
input_ids = tokenizer("paraphrase: " + text, return_tensors="pt").input_ids

# Generowanie z poprawioną długością
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    num_beams=5,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)


# Dekodowanie
paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(paraphrased_text)

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Wczytanie modelu
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

text = "Question: What is the best way to train a machine learning model? Answer:"

inputs = tokenizer(text, return_tensors="pt", padding=True)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,
    num_beams=5,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    no_repeat_ngram_size=2
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)


from transformers import T5ForConditionalGeneration, T5Tokenizer

# Model FLAN-T5-XL
model_name = "google/flan-t5-xl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Prompt wejściowy
text = (
    "You are an expert in machine learning. "
    "Please provide a detailed step-by-step guide for training a machine learning model, "
    "starting from data collection and preprocessing, and ending with evaluation and deployment."
)

# Tokenizacja
inputs = tokenizer(text, return_tensors="pt", padding=True)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Generowanie odpowiedzi
outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=500,  # Więcej tokenów dla dłuższej odpowiedzi
    num_beams=7,         # Większa liczba wiązek dla spójności
    temperature=0.3,     # Mniejsza losowość
    top_p=0.9,           # Zrównoważona różnorodność
    no_repeat_ngram_size=3  # Zapobieganie powtórzeniom
)

# Dekodowanie i wyświetlenie wyniku
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)





from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence = "This is something which i cannot understand at all"

text =  "paraphrase: " + sentence + " </s>"

encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    do_sample=True,
    top_k=200,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=5
)

for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(line)
