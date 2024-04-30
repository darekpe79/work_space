# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:30:16 2024

@author: dariu
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "tuner007/pegasus_paraphrase"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "What is the best way to train a model?"

# Przygotowanie inputu dla modelu
input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids

# Generowanie parafrazy
outputs = model.generate(input_ids, max_length=60, num_beams=10, num_return_sequences=1, temperature=1.5)
paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(paraphrased_text)