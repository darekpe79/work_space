# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 11:18:19 2025

@author: darek
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"

# jeśli chcesz 8-bit, użyj load_in_8bit=True
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)

prompt = """### Instrukcja: Oceń, czy poniższy tekst kwalifikuje się do Polskiej Bibliografii Literackiej. 
Odpowiedz jednym słowem: True lub False.
### Wejście: To jest recenzja nowego tomu wierszy współczesnego poety, wraz z analizą jego twórczości.
### Odpowiedź:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )
    
odp=tokenizer.decode(outputs[0])#, skip_special_tokens=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "speakleash/Bielik-4.5B-v3.0-Instruct"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)

prompt = (
    "### Instrukcja: Oceń, czy poniższy tekst kwalifikuje się do Polskiej Bibliografii Literackiej.\n"
    "Odpowiedz jednym słowem: True lub False.\n"
    "### Wejście: To jest recenzja nowego tomu wierszy współczesnego poety, wraz z analizą jego twórczości.\n"
    "### Odpowiedź: "   # UWAGA: spacja na końcu
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# with torch.inference_mode():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=20,
#         do_sample=False
#     )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=4,        # tylko 1 token, czyste True/False
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# dekodujemy tylko wygenerowaną część
gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
print("ODP:", gen)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "speakleash/Bielik-4.5B-v3.0-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=quantization_config
)

prompt = (
    "### Instrukcja: Oceń, czy poniższy tekst kwalifikuje się do Polskiej Bibliografii Literackiej.\n"
    "Odpowiedz jednym słowem: True lub False.\n"
    "### Wejście: To jest recenzja nowego tomu wierszy współczesnego poety, wraz z analizą jego twórczości.\n"
    "### Odpowiedź: "   # <- ważna SPACJA!
)

# 1) Zobacz, jak tnie ' True' i ' False'
ids_true  = tok.encode(" True",  add_special_tokens=False)
ids_false = tok.encode(" False", add_special_tokens=False)
print(" True ->", ids_true,  [tok.decode([i]) for i in ids_true])
print(" False->", ids_false, [tok.decode([i]) for i in ids_false])

# 2) Generacja i podgląd token-po-tokenie
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=8,      # daj 6–8 żeby zobaczyć kroki
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

gen_ids = out.sequences[0][inputs.input_ids.shape[1]:]  # tylko nowa część
pieces = [tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False) for tid in gen_ids.tolist()]
print("TOKENS:", gen_ids.tolist())
print("PIECES:", pieces)
print("FULL  :", tok.decode(gen_ids, skip_special_tokens=True).strip())
