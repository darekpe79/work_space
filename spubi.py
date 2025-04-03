# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:02:45 2025

@author: darek
"""


import pandas as pd
import json
from tqdm import tqdm 

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Nazwa modelu to dziala
model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"
# model_name ="CYFRAGOVPL/PLLuM-12B-nc-instruct"

# Konfiguracja kwantyzacji
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,                         # Włącz kwantyzację 8-bitową
    llm_int8_enable_fp32_cpu_offload=True      # Offload części modelu na CPU
)

# Wczytanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Wczytanie modelu z konfiguracją kwantyzacji
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                         # Automatyczne przypisanie GPU/CPU
    quantization_config=quant_config,         # Nowa metoda dla 8-bit
    trust_remote_code=True
)


# kweantyzacja 4bit proby - tutaj nie działa

model_name = "CYFRAGOVPL/PLLuM-12B-nc-instruct"

# Konfiguracja dla 4-bitowej kwantyzacji
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Włącz kwantyzację 4-bitową
    bnb_4bit_compute_dtype="float16",      # Precyzja obliczeń (float16 działa dobrze na większości GPU)
    bnb_4bit_quant_type="nf4",             # Użyj Normalized Float 4 (NF4) - lepsza jakość
    llm_int8_enable_fp32_cpu_offload=True  # Offload części obliczeń na CPU
)

# Załaduj model z kwantyzacją
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"  # Automatyczny przydział GPU/CPU
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
# Pipeline do generowania tekstu
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# Funkcja do zadawania pytań
# def ask_llama_pllum(prompt: str, max_new_tokens=1000) -> str:
#     output = generator(
#         prompt,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.8,
#         repetition_penalty=1.2
#     )
#     return output[0]["generated_text"]
import torch 

# # Ustaw ziarno dla całego skryptu
# torch.manual_seed(42)

# torch.backends.cudnn.deterministic = True  # Zwiększa determinizm
# torch.backends.cudnn.benchmark = False     # Wyłącza optymalizacje, które mogą wprowadzać losowość


# 2. Zmodyfikowana funkcja generowania
def ask_llama_pllum(prompt: str, max_new_tokens=1000) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.7,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "speakleash/Bielik-11B-v2.3-Instruct"

# Ładujemy tokenizer i model Bielika w float16 na GPU:
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# ).cuda()

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True
)
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
    #device=0  # GPU
)

#%% Działajacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "speakleash/Bielik-11B-v2.3-Instruct"

# 1. Ładujemy tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 2. Ładujemy model w 8-bitach (o ile VRAM na to pozwala; ewentualnie float16)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,  # Zamiast load_in_8bit możesz też użyć BitsAndBytesConfig 4-bit.
    trust_remote_code=True
)

# 3. Definiujemy funkcję do generacji
def ask_bielik(prompt: str, max_new_tokens=10000) -> str:
    # Tokenizujemy prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generujemy odpowiedź
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            # top_k=40,  # można dodać, jeśli model zbyt powtarza
            # brak stop tokens, ale można dodać
        )
    # Dekodujemy do stringa, pomijając special tokens
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ---- PRZYKŁADOWE UŻYCIE ----



prompt=''' Jesteś profesjonalnym bibliografem, specjalistą w dziedzinie analizowania i wyodrębniania tytułów utworów z adnotacji bibliograficznych. Twoim zadaniem jest czytelne wylistowanie tytułów utworów z podanego tekstu.

Aby ułatwić Ci pracę, przedstawiam dwa przykłady, w których dokonano analogicznego wydobycia tytułów. Dzięki nim zobaczysz, jakie schematy stosujemy:

Przykład 1
Oto adnotacja:
“Chory dom: Chory dom (1-6). Pamięci księdza Jerzego Popiełuszki. - Garść ognia: Głos. Boże Narodzenie 1981. Czarna Maria ze wsi Sochy. Psalm - Wygnanie (1-8). Apokalipsa. Zarzynanie gęsi. Lęki. Psalm - Mieszkanka Północy. Psalm z zimna. Posag. Psalm - Biały ranek. Ogród. Psalm z ognia (1-7). Wrzesień 1984 (1-3). - Garść wody: Kobieta i morze. Plaża w Dębkach. Psalm ognia i wody. Kobieta i mężczyzna. Poranek. Łabędź. Garść wody. Modlitwa wieczorna. Połów. Dom wody. Nokturn. Późny wieczór.”

Zauważ, że mamy trzy tytuły „rozdziałów”:

Chory dom:
Garść ognia:
Garść wody:
Pod nimi znajdują się tytuły konkretnych utworów, np. „Chory dom (1-6).”, „Pamięci księdza Jerzego Popiełuszki.”, „Głos.”, „Boże Narodzenie 1981.”, „Czarna Maria ze wsi Sochy.” itp.

Z adnotacji wyodrębniamy same tytuły utworów, ignorujemy tytuły rozdziałów i inne zbędne elementy. Efekt końcowy to lista wszystkich tytułów (np. ponumerowana lub wypunktowana), zachowująca ich oryginalną pisownię i kolejność.

Przykład 2
Oto adnotacja:
“Sztychy i akwarele: Krakowski koncert. Rynek Główny. Brama Floriańska. Katedra Wawelska. Pomnik wieszcza. Adam Mickiewicz. Komnaty królewskie na Wawelu. Pomnik Stanisława Wyspiańskiego. Arrasy wawelskie. Ulica Kanonicza. Hejnał z wieży mariackiej. U fryzjera w Pasażu Bielaka. Sukiennice. Do krakowskich gołębi. Imieniny Krakowa. Pomnik Grunwaldzki. W Muzeum Narodowym. Pochód rektorski. Na Kazimierzu cień Estery... Orkiestra Małego Władzia. W Muzeum Żup Krakowskich w Wieliczce. Do ZOO układanka dla syna. Na krakowskim dworcu PKS. U racjonalistów w klubie „Euhemer”. Bractwa Kurkowego przesłanie. Robotnicy. - Opowiedz mi: Opowiedz mi teraz... We dwoje w Parku Jordana. W kawiarni Literackiej. Podróż z K.I. Gałczyńskim. Wołam Cię, Eurydyko. W knajpie „Na Stawach” (J. Harasymowiczowi). Na Rynku. Kwiaty krakowskie. Szczęscie. Niewidomy skrzypek z ulicy Wiślnej. Władek Kapciuch. Sprzątanie w świetlicy dworcowej. Odpływanie. - Smog: Smok wawelski. Dwa mosty. Ulica Wiślna latem. Requiem. Czarna mgła. Malowanka na szkle dla syna Jerzego. Rozmowa na serio z wnukiem w r. 2000.”

Ponownie, mamy tytuły „rozdziałów” („Sztychy i akwarele:”, „Opowiedz mi:”, „Smog:”); nie są one właściwymi tytułami utworów. W tej adnotacji pojawiają się też liczne tytuły utworów, np. „Krakowski koncert.”, „Rynek Główny.”, „Brama Floriańska.”, „Adam Mickiewicz.”, „Komnaty królewskie na Wawelu.”, „Opowiedz mi teraz…”, „We dwoje w Parku Jordana.”, „W kawiarni Literackiej.”, „Wołam Cię, Eurydyko.”, „Na Rynku.”, „Smok wawelski.”, „Dwa mosty.” itd.

Tak jak w pierwszym przykładzie — ignorujemy tytuły rozdziałów, a wyciągamy jedynie tytuły utworów w oryginalnej kolejności i pisowni, pomijając wszystkie dodatkowe informacje czy notatki.

Twoje aktualne zadanie:

Otrzymujesz następujący tekst do analizy i wydobycia z niego tytułów (tylko tytułów!):

"On. Sława i rzeczy. Dzień twoich narodzin. Arkadia. Jabłko. Oni są tak gościnni. Sami skazali się. List miłosny Alfreda Prufrocka (Fragment). Kamień. Piosenka dla Marylin Monroe. Ostatni wieczór autorski. Przebudzenie. Miłość. Jesień (K. Korpolewskiej). Noc. Ulica piękna (J. Z. Cichockim). Oda do młodości (E. i J. Olszewskim). Piosenka dla dziewczyny z chmur. Piosenka o duszach (Dla M. C. Patel). Pieśń błazeńska. Pieśń obłąkania. Pieśń wieczorna. Pieśń zmartwychwstania. Pieśń o rzece. Pieśń dla poety. Oda do nie przeczytanych książek. Oda do domu którego nie mam. Oda do dziurawego buta. Oda do szkolnych lektur. Wiosna, twój ptasi powrót do nie twojej Itaki (Z. Atemborskiemu). Kropla, kropelka (W. Chętnickiemu). Tłum. Nie. Możesz. Sportowe życie. Proces. Siedem serc. Odkrycie. Wiersz dla Urszuli Ambroziewicz. Elegia dla G.P. Pieśń z pustelni (M. Kobzdejowi). Zabawki. Krasnoludek. Wszystko już powiedziałem (K. Tarasiewicz). - [Not. i fot. na okł.]."

Nie zwracaj żadnych innych informacji (nie przywołuj przykładów, nie wyjaśniaj, nie komentuj).

Nie wymieniaj tytułów rozdziałów (jeśli by się pojawiły), tylko tytuły utworów.

Zachowaj oryginalną pisownię i kolejność.

W odpowiedzi mają pojawić się wyłącznie same tytuły — np. jeden wiersz za drugim lub w formie numerowanej listy'''

interp = interpret_prompt.format(user_question=user_query)


interpretation = ask_llama_pllum(prompt)
interpretation = ask_bielik(prompt)
print(interpretation)


#%%

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

model_name = "speakleash/Bielik-11B-v2.3-Instruct"

# Konfiguracja 4-bit (NF4)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # obliczenia w float16
    bnb_4bit_quant_type="nf4",            # NormalFloat4 (lepsza od FP4)
    llm_int8_enable_fp32_cpu_offload=True # opcjonalnie, jeśli chcesz offload na CPU
)

# 1. Ładujemy tokenizer (trust_remote_code=True – jeśli Bielik ma niestandardowe pliki modelu)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 2. Ładujemy model w 4-bitach
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,  # kluczowy fragment do 4-bit
    device_map="auto",
    trust_remote_code=True
)

# 3. Definiujemy funkcję generującą
def ask_bielik(prompt: str, max_new_tokens=512) -> str:
    # Tokenizujemy prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generujemy odpowiedź
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2 , # ogranicza powtarzanie n-gramów
            # top_k=40,               # opcjonalnie, jeśli chcesz dodatkowo ograniczyć wybory
        )
    # Dekodujemy do stringa
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Jeśli model mimo wszystko cytuje prompt, możemy to wyciąć:
    # (usunięcie pierwszego wystąpienia promptu z odpowiedzi)
    if prompt in answer:
        answer = answer.replace(prompt, "").strip()
    
    return answer

# 4. Przykładowe użycie
prompt = """Jesteś profesjonalnym bibliografem...
[Tutaj wklej długi tekst do analizy i wyodrębnienia tytułów]
"""

wynik = ask_bielik(prompt, max_new_tokens=1024)
print(wynik)
