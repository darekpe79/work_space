# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:50:47 2025

@author: darek
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Nazwa modelu na Hugging Face
model_name = "lzw1008/ConspEmoLLM-7b"

# Ładowanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Ładowanie modelu z automatycznym przypisaniem do GPU/
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # Automatyczne przypisanie do GPU/CPU
    torch_dtype=torch.float16, # Oszczędza pamięć, wymaga GPU z obsługą float16
    trust_remote_code=True   # Wymagane dla modeli z niestandardowym kodem
)

# Jeśli chcesz użyć kwantyzacji 8-bitowej (opcja dla oszczędności pamięci VRAM)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     load_in_8bit=True,      # Włączenie kwantyzacji 8-bitowej
#     trust_remote_code=True
# )

# Tworzenie pipeline do generacji tekstu
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Funkcja do generowania odpowiedzi
def generate_text(prompt, max_length=512):
    output = generator(
        prompt,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,  # Kontrola kreatywności
        top_p=0.9        # Metoda nucleus sampling
    )
    return output[0]["generated_text"]

if __name__ == "__main__":
    # Testowe pytanie
    prompt = "Describe the emotions evoked by a conspiracy theory about UFOs."
    result = generate_text(prompt)
    print("Wynik:")
    print(result)