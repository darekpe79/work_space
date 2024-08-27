# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:55:07 2024

@author: dariu
"""

from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# Załaduj zapisany model i tokenizer
model = MT5ForConditionalGeneration.from_pretrained("C:/Users/dariu/trained_modelQA/")
tokenizer = MT5Tokenizer.from_pretrained("C:/Users/dariu/trained_modelQA/")
# Przykładowe pytanie
input_question = "jakie książki tworzył Adam Mickiewicz?"

# Tokenizacja pytania
input_ids = tokenizer.encode(input_question, return_tensors="pt", max_length=512, truncation=True)
# Generowanie odpowiedzi
output = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)

# Dekodowanie wygenerowanego tekstu
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print("Odpowiedź:", answer)

# question_forms = [
#     "jakie książki napisał",
#     "jakie książki stworzył",
#     "jakich książek jest autorem",
# ]
