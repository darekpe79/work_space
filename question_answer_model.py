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

model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = "What is the best way to train a model?"

input_ids = tokenizer("paraphrase: " + text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(paraphrased_text)


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
