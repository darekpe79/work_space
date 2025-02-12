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
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",       # Automatyczne przypisanie do GPU/CPU
#     torch_dtype=torch.float16, # Oszczędza pamięć, wymaga GPU z obsługą float16
#     trust_remote_code=True   # Wymagane dla modeli z niestandardowym kodem
# )

# Jeśli chcesz użyć kwantyzacji 8-bitowej (opcja dla oszczędności pamięci VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,      # Włączenie kwantyzacji 8-bitowej
    trust_remote_code=True
)

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
        temperature=1.0,  # Kontrola kreatywności
        top_p=0.9        # Metoda nucleus sampling
    )
    return output[0]["generated_text"]

if __name__ == "__main__":
    # Testowe pytanie
    prompt = ''''check whether there are conspiracy themes in the content of the attached letter? answer true or false and explain decision "
An Uncle to his Nephew.
[? 1607], July 18.	Honest nephew, I writ to you last by my cousin James what good success your business hath in Court. And now having opportunity to send unto you safely by our cousin Thomas, I send you a bee in a box, out of which you and all England may gather honey, if it be rightly handled. These articles were showed me by a dear friend, and with much ado writ them out myself, because I durst not trust any in this place. You are acquainted with my fist, and therefore it will serve between you and me. Impart them warily to our trusty friends, and let copies of them be dispersed secretly among them, that upon occasion they may show themselves true Englishmen, lovers of their country's liberty and the welfare of their posterity. Your plate cost 23l. 10s. 9d., your new suit 5l. 3s. I will fetch the money at Michaelmas, for I desire to see you and our other friends and to confer about many things. Away with Scots and Danes and English atheists, their complices, or woe to England for ever. Tib and my daughter Ann greet you well.—My house near London, 18 July.
Holograph. Signature illegible. ½ p. (121. 138.)

"'''
    result = generate_text(prompt)
    print("Wynik:")
    print(result)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Nazwa modelu na Hugging Face
model_name = "lzw1008/ConspEmoLLM-7b"

# Ładowanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Ładowanie modelu na CPU
device = torch.device("cpu")  # Wymuszenie CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
).to(device)  # Przeniesienie modelu na CPU

# Tworzenie pipeline do generacji tekstu
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # Wskazuje użycie CPU w pipeline
)

# Funkcja do generowania odpowiedzi
def generate_text(prompt, max_length=512):
    output = generator(
        prompt,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return output[0]["generated_text"]

if __name__ == "__main__":
    # Testowe pytanie
    prompt = "Describe the emotions evoked by a conspiracy theory about UFOs."
    result = generate_text(prompt)
    print("Wynik:")
    print(result)
