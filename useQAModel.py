# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:12:54 2024

@author: dariu
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Ładowanie wytrenowanego modelu i tokenizatora
model = T5ForConditionalGeneration.from_pretrained("./trained_model")
tokenizer = T5Tokenizer.from_pretrained("./trained_model")

# Funkcja do generowania odpowiedzi
def generate_answer(question):
    input_text = f"question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    outputs = model.generate(inputs.input_ids, max_length=50, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

question = "Kto napisał Bajki o dinozaurach?"
answer = generate_answer(question)
print(f"Pytanie: {question}\nOdpowiedź: {answer}\n")