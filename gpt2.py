# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:45:37 2024

@author: dariu
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Wybór modelu GPT-2
model_name = 'gpt2'  # Możesz również użyć 'gpt2-medium', 'gpt2-large', 'gpt2-xl' w zależności od potrzeb

# Ładowanie modelu i tokenizer'a
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Funkcja generująca tekst
def generate_text(prompt, max_length=50):
    # Tokenizacja wejściowego tekstu
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generowanie tekstu
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    
    # Dekodowanie wygenerowanego tekstu
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return text

# Przykład użycia
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

# Wybór modelu GPT-2
model_name = 'gpt2'

# Ładowanie modelu i tokenizer'a
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Funkcja generująca tekst
def generate_text(prompt, max_length=50):
    # Tokenizacja wejściowego tekstu
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generowanie tekstu z odpowiednimi ustawieniami maski i pad token
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Dekodowanie wygenerowanego tekstu
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return text

# Funkcja przetwarzająca pytanie użytkownika i generująca zapytanie SQL
def process_user_question(question):
    # Wyrażenie regularne do wyodrębnienia nazwiska autora
    match = re.search(r"książki których autorem jest (.+)", question, re.IGNORECASE)
    if match:
        author_name = match.group(1).strip()
        prompt = f"Generate an SQL query to select book titles where the author's name is '{author_name}':\nSELECT title FROM books WHERE author = '{author_name}';"
        return generate_text(prompt, max_length=50)
    else:
        return "Could not understand the question. Please ask about books by a specific author."

# Przykład użycia
user_question = "Chcę książki których autorem jest Adam Mickiewicz"
sql_query = process_user_question(user_question)
print(sql_query)


from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import re

# Wybór modelu GPT-Neo
model_name = 'EleutherAI/gpt-neo-2.7B'

# Ładowanie modelu i tokenizer'a
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Funkcja generująca tekst
def generate_text(prompt, max_length=50):
    # Tokenizacja wejściowego tekstu
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generowanie tekstu z odpowiednimi ustawieniami maski i pad token
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Dekodowanie wygenerowanego tekstu
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return text

# Funkcja przetwarzająca pytanie użytkownika i generująca zapytanie SQL
def process_user_question(question):
    # Wyrażenie regularne do wyodrębnienia nazwiska autora
    match = re.search(r"książki których autorem jest (.+)", question, re.IGNORECASE)
    if match:
        author_name = match.group(1).strip()
        prompt = f"Generate an SQL query to select book titles where the author's name is '{author_name}':\nSELECT title FROM books WHERE author = '{author_name}';"
        return generate_text(prompt, max_length=50)
    else:
        return "Could not understand the question. Please ask about books by a specific author."

# Przykład użycia
user_question = "Chcę książki których autorem jest Adam Mickiewicz"
sql_query = process_user_question(user_question)
print(sql_query)
