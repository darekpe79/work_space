# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:24:53 2025

@author: darek
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- ustawienia
assert torch.cuda.is_available(), "Brak GPU CUDA."
device = torch.device("cuda:0")
model_name = "speakleash/Bielik-4.5B-v3.0-Instruct"
hf_token = os.getenv("HF_TOKEN")

# (opcjonalnie: lepsza wydajność na RTX)
torch.backends.cuda.matmul.allow_tf32 = True

# --- tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,                 # w starszych transformers: use_auth_token=hf_token
)

# --- model: FP16 + cały na GPU (bez device_map="auto")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch.float16,      # FP16 (BF16 nie na 4070TiS)
    low_cpu_mem_usage=True,
)
model.to(device)                    # <- kluczowe: wszystko na GPU

# awaryjnie ustaw pad_token_id, jeśli brak
if model.generation_config.pad_token_id is None:
    model.generation_config.pad_token_id = tokenizer.eos_token_id

# --- wiadomości (ChatML)
messages = [
    {"role": "system", "content": "Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim."},
    {"role": "user", "content": "Jakie mamy pory roku w Polsce?"},
    {"role": "assistant", "content": "W Polsce mamy 4 pory roku: wiosna, lato, jesień i zima."},
    {"role": "user", "content": "Która jest najcieplejsza?"},
    {"role": "assistant", "content": ""}   # sygnał: tutaj ma pisać
]

# Tokenizacja promptu
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# Odcinamy część promptu
new_tokens = output_ids[0][input_ids.shape[-1]:]
generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
print("Odpowiedź modelu:", generated_text.strip())



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

device = "cuda"
model_name = "speakleash/Bielik-4.5B-v3.0-Instruct"
hf_token = os.getenv("HF_TOKEN")

# Ładowanie modelu i tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# Prosty prompt bez ChatML
prompt = """Odpowiadaj krótko, precyzyjnie i wyłącznie w języku polskim.
Pytanie: Jakie mamy pory roku w Polsce?
Odpowiedź:"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# Dekodowanie nowo wygenerowanej części
generated_text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Odpowiedź modelu:", generated_text.strip())


import torch
from transformers import AutoTokenizer

model_name = "speakleash/Bielik-4.5B-v3.0-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sprawdzamy identyfikatory tokenów
true_tokens = tokenizer.encode(" True", add_special_tokens=False)
false_tokens = tokenizer.encode(" False", add_special_tokens=False)

print("True tokens:", true_tokens)
print("False tokens:", false_tokens)

# Opcjonalnie: pokazujemy, co dekoduje się z pierwszego tokenu
print("Decode True first token:", tokenizer.decode([true_tokens[0]]))
print("Decode False first token:", tokenizer.decode([false_tokens[0]]))

import torch
from transformers import AutoTokenizer

# PLLuM zamiast Bielika
model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

true_tokens = tokenizer.encode(" True", add_special_tokens=False)
false_tokens = tokenizer.encode(" False", add_special_tokens=False)

print("True tokens:", true_tokens)
print("False tokens:", false_tokens)

print("Decode True first token:", tokenizer.decode([true_tokens[0]]))
print("Decode False first token:", tokenizer.decode([false_tokens[0]]))


