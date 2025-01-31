# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:57:18 2025

@author: darek
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Wybierz model, np. BLOOM
model_name = "bigscience/bloom-7b1"

# Pobierz tokenizer i model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Funkcja do generowania tekstu
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Przykładowe zapytanie
prompt = "What is the future of AI?"
generated_text = generate_text(prompt)
print(generated_text)


from transformers import pipeline

# Możesz wybrać inny model niż google/flan-t5-large,
# np. bigscience/bloomz-7b1 albo google/flan-t5-xxl (choć jest większy).
model_name = "google/flan-t5-large"

# Konfigurujemy pipeline do generowania tekstu
qa_pipeline = pipeline("text2text-generation", model=model_name)

def answer_question(question: str) -> str:
    """
    Zwraca odpowiedź na zadane pytanie w formie tekstu.
    """
    prompt = (
        f"You are a helpful and knowledgeable assistant. "
        f"Please answer the following question as accurately as possible.\n\n"
        f"Question: {question}\nAnswer:"
    )

    # Wykonujemy generowanie
    response = qa_pipeline(
        prompt,
        max_length=250,       # maksymalna długość odpowiedzi
        num_return_sequences=1,
        temperature=0.7,      # od 0.0 (bardzo konserwatywne) do ~1.0 (bardziej kreatywne)
        top_p=0.9
    )
    # Pipeline zwraca listę odpowiedzi, bierzemy pierwszą
    return response[0]["generated_text"]

# Przykładowe pytanie
if __name__ == "__main__":
    question = "What is the future of AI?"
    generated_answer = answer_question(question)
    print("Question:", question)
    print("Answer:", generated_answer)


#%% bloomz
from transformers import pipeline

def generate_answer(prompt: str, max_new_tokens=100) -> str:
    model_name = "bigscience/bloomz-7b1"
    
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=0, #jeśli masz GPU,
        return_full_text=False  # <-- kluczowe, by NIE dołączał promptu do wygenerowanego tekstu
    )
    
    output = generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=1,
        top_p=1,
        repetition_penalty=1.2,
        #num_return_sequences=1
    )
    # Zwracamy tylko generowany fragment (bez promptu)
    return output[0]["generated_text"]

if __name__ == "__main__":
    prompt = (
        "You are a helpful, knowledgeable assistant. "
        "Please answer the following question in detail:\n\n"
        "Question: What is the future of AI?\n\n"
        "Answer:"
    )
    response = generate_answer(prompt)
    print("Prompt:", prompt)
    print("Response:", response)
    
    
 
            
#%% falcon            
from transformers import pipeline

model_name = "tiiuae/falcon-7b-instruct"
generator = pipeline(
    "text-generation",
    model=model_name,
    device=0,
    return_full_text=False
)

def ask_falcon(question: str, max_new_tokens=200) -> str:
    """
    Zadaj pytanie modelowi Falcon 7B i zwróć odpowiedź.
    """
    prompt = (
        "You are a helpful, knowledgeable assistant. "
        "Please answer in detail:\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"].strip()


q="What is the future of AI?"
response = ask_falcon(q, max_new_tokens=300)

if __name__ == "__main__":
    # Przykładowe pytania (po angielsku i po polsku)
    questions = [
        "What is the future of AI?",
                "How does reinforcement learning differ from supervised learning?"
    ]

    for i, q in enumerate(questions, 1):
        response = ask_falcon(q, max_new_tokens=200)
        print(f"\n--- [Pytanie {i}] ---")
        print("Question:", q)
        print("Answer:", response)

#%% TLUMACZENIE
from transformers import pipeline

def translate_pl_to_en(text_pl: str) -> str:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-pl-en")
    # translator może przetwarzać listy, więc w przypadku pojedynczego str:
    result = translator(text_pl, max_length=512)
    return result[0]["translation_text"]

if __name__ == "__main__":
    text_pl = '''skarpetki miały być wielokolorowe, a są białe'''
    translation = translate_pl_to_en(text_pl)
    print("ORYGINAŁ (PL):", text_pl)
    print("TŁUMACZENIE (EN):", translation)

#%%
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/mt5-large"  # możesz też spróbować "google/mt5-base" / "google/mt5-xl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_polish_text(text: str, max_new_tokens=100) -> str:
    """
    Streszcza (skraca) tekst w j. polskim.
    mT5 oryginalnie nie jest 'fine-tuned' do streszczania,
    ale możemy dać prefix: 'summarize in Polish: ...'
    """
    # Tworzymy prompt w stylu T5
    prompt = f"summarize in Polish: {text}"

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # ile tokenów ma wygenerować
            do_sample=False                 # greedy (stabilniejsze, krótsze)
        )
    # Dekodowanie z usunięciem tokenów specjalnych
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    article_pl = (
        "Sztuczna inteligencja staje się coraz bardziej popularna na całym świecie. "
        "Wiele firm inwestuje w rozwój systemów uczenia maszynowego. "
        "Pojawia się też wiele pytań o etykę i bezpieczeństwo. "
        "W niniejszym artykule przyjrzymy się perspektywom rozwoju SI w najbliższych latach."
    )
    summary = summarize_polish_text(article_pl, max_new_tokens=80)
    print("ORYGINALNY TEKST (PL):", article_pl)
    print("STRESZCZENIE (mT5):", summary)




#%%BIELIK
'''1. max_new_tokens

    Określa, ile maksymalnie „nowych” tokenów (słów/fraz) model może wygenerować ponad liczbę tokenów już podanych w prompt.
    Zakres:
        Typowo od kilkudziesięciu do kilkuset.
        W codziennych zastosowaniach:
            20–50 tokenów da bardzo krótką, zwięzłą odpowiedź,
            100–300 to standard do średniej długości wypowiedzi,
            1000+ dla szczególnie długich odpowiedzi (np. opowiadania).
    Im większa wartość, tym model może tworzyć dłuższe wyjaśnienia (ale też wolniej się kończy generacja).
    Przy zbyt małej wartości może przerwać wypowiedź w połowie zdania.

2. do_sample

    Jeśli do_sample=True, model korzysta z losowego próbkowania (zgodnie z rozkładem prawdopodobieństw słów), zamiast deterministycznego (greedy) wybierania najbardziej prawdopodobnego kolejnego tokenu.
    Zalety do_sample=True:
        Bardziej kreatywne i różnorodne odpowiedzi,
        Można sterować temperaturą i top_p.
    Jeśli do_sample=False:
        Model generuje deterministyczne (greedy) odpowiedzi.
        Zwykle bardziej spójne, mniej losowe, ale czasem monotonne.

3. temperature

    Steruje losowością w generacji, gdy do_sample=True.
    Wzór (w uproszczeniu): model podnosi rozkład prawdopodobieństw słów do potęgi 1/temperature.
        temperature=1.0: brak modyfikacji (model generuje tokeny wg oryginalnych prawdopodobieństw).
        temperature>1.0: bardziej „płaski” rozkład — model staje się bardziej kreatywny i nieprzewidywalny,
        temperature<1.0: rozkład bardziej „skupiony” — model częściej wybiera token o najwyższym prawdopodobieństwie, staje się mniej kreatywny i bardziej zachowawczy.
    Zakres:
        Zwykle od 0.1 do 2.0.
        0.1–0.3 to prawie deterministyczne,
        0.7–1.0 to popularny zakres dla względnie kreatywnych, ale sensownych odpowiedzi,
        >1.2 może być mocno chaotyczne.

4. top_p (nucleus sampling)

    Nucleus sampling: model bierze pod uwagę tylko tokeny, które łącznie mają prawdopodobieństwo p (np. p=0.9).
    Przykład: jeśli top_p=0.9, to model odcina długi „ogon” rzadkich tokenów i próbuje wybierać tylko z puli, której łączne prawdopodobieństwo to 0.9.
    Im mniejsze top_p, tym węższa pula tokenów możliwych do generacji (odrzucasz bardziej egzotyczne słowa).
    Typowe wartości: 0.8–0.95.
    Możesz łączyć top_p z temperature w zależności od stylu, jaki chcesz uzyskać.

5. repetition_penalty

    Kara za powtarzanie: zwiększa lub zmniejsza prawdopodobieństwo słów, które już się pojawiły.
    Wartość >1.0 powoduje, że model będzie unikał powtarzania słów/zwrotów, bo za każdy tok, który już się pojawił, będzie „płacił” karę.
    Typowy zakres: 1.0–1.3.
        1.0 wyłączona kara,
        1.1–1.3 wymusza większą różnorodność słownictwa,
        powyżej 1.3 może czasem szkodzić spójności, bo model zacznie unikać naturalnych powtórzeń.

Zalecane praktyczne wartości

    max_new_tokens:
        Jeśli chcesz szybkich, krótkich odpowiedzi → ~50–150,
        Dłuższe teksty → 200–500 (zwiększa czas i zasoby).
    do_sample=True:
        Pozostaw, jeśli chcesz, by odpowiedzi były mniej schematyczne.
    temperature=0.7–1.0:
        Daje najczęściej zbalansowaną kreatywność i spójność.
        1.0 jest wartością „neutralną”.
        0.6–0.8 jest często wybierane do Q&A, by generować bardziej rzeczowe, mniej rozpraszające odpowiedzi.
    top_p=0.9:
        Dość standardowa wartość. Dobrze sprawdza się w wielu zadaniach, „ucina” skrajne tokeny.
        Możesz eksperymentować z 0.8 lub 0.95.
    repetition_penalty=1.2:
        Sprawdza się, by uniknąć nadużywania powtórzeń.
        Możesz próbować 1.1 lub 1.25, zależy od obserwowanego stylu.'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "speakleash/Bielik-11B-v2.3-Instruct"

# Ładujemy tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Ładujemy model z automatycznym mapowaniem na GPU (wymaga accelerate)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # automatycznie wgra model na GPU/CPU (o ile jest accelerate)
    torch_dtype=torch.float16, # wymaga GPU z obsługą float16
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
#Kwantyzajca 8 bitowa
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True
)
# Tworzymy pipeline:
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    #device=0  # to zapewnia GPU (nr 0)
)






def ask_bielik(prompt: str, max_new_tokens=512) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]

if __name__ == "__main__":
    # Polski prompt w stylu "instruktażowym":
    question = "kim jest adam mickiewicz?"
    bielik_prompt = f"""Jesteś polskim modelem asystującym. Napisz szczegółową, krótką wypowiedź:.

Pytanie: {question}

Odpowiedź:"""

    answer = ask_bielik(bielik_prompt)
    print("Prompt:", bielik_prompt)
    print("Answer:", answer)



# z szczegółową, długą wypowiedź, omawiając wiele aspektów


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "speakleash/Bielik-11B-v2.3-Instruct"

# Ładujemy tokenizer i model Bielika w float16 na GPU:
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # GPU
)

def ask_bielik(prompt: str, max_new_tokens=300) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,   # możesz zmienić
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]

# --------------------
# 1) Prompt interpretacyjny
# --------------------
interpret_prompt = """Jesteś asystentem bibliotecznym, który odbiera pytania użytkowników 
i potrafi przygotować zapytania do systemu wyszukiwania RAG.

Twoim zadaniem jest:
1. Zrozumieć, o co pyta użytkownik.
2. Określić, czy szuka książek po tytule, autorze, czy tematyce.
3. Wyodrębnij kluczowe hasło (np. jeśli to tematyka – wyodrębnij temat; jeśli to autor – nazwisko; jeśli to tytuł – tytuł).
4. Wypisz to w formacie:
   [subject: "coś"], [author: "nazwisko"], albo [title: "tytuł"]
5. Dodaj krótkie wyjaśnienie w języku naturalnym.

Przykład:
Pytanie: "Czy znajdę książki o grach komputerowych?"
Odpowiedź:
[subject: "gry komputerowe"]
Użytkownik szuka książek związanych z grami komputerowymi.

Zachowaj tę konwencję w odpowiedzi.
"""

# --------------------
# 2) Prompt finalnej odpowiedzi
# --------------------
def create_final_prompt(user_query, results_text):
    return f"""Użytkownik pytał: {user_query}

Oto wyniki wyszukiwania z bazy biblioteki:
{results_text}

Napisz zwięzłą i przyjazną odpowiedź dla użytkownika, uwzględniając powyższe pozycje.
"""

# --------------------
# TEST
# --------------------
if __name__ == "__main__":
    # Przykładowe zapytanie:
    user_query = "Czy znajdę książki o grach komputerowych?"
    
    # 1) Wywołanie promptu interpretacyjnego
    interp = interpret_prompt + f"\nPytanie użytkownika: {user_query}\n"
    interpretation = ask_bielik(interp, max_new_tokens=200)
    print("=== INTERPRETACJA ===")
    print(interpretation)
    
    # 2) Załóżmy, że "wyniki" to 2 przykładowe książki (normalnie tu byłoby z FAISS):
    mock_results_text = """Znalezione pozycje:
1) Tytuł: "Historia gier komputerowych", Autor: Jan Kowalski
2) Tytuł: "Gry komputerowe. Wpływ na kulturę masową", Autor: Anna Nowak
"""
    # Teraz tworzymy prompt finalny
    final_prompt = create_final_prompt(user_query, mock_results_text)
    
    final_answer = ask_bielik(final_prompt, max_new_tokens=300)
    print("\n=== FINALNA ODPOWIEDŹ ===")
    print(final_answer)


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Nazwa modelu na Hugging Face
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uruchamianie na urządzeniu: {device}")

# Ładowanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

# Ładowanie modelu z kwantyzacją 8-bitową dla oszczędności pamięci
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # Automatyczny podział na GPU i CPU
    load_in_8bit=True,          # 8-bitowa kwantyzacja (znacznie zmniejsza użycie VRAM)
    torch_dtype=torch.float16,  # Obliczenia w float16 dla wydajności
)

# Tworzenie pipeline do generacji tekstu
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # GPU
)

# Testowe generowanie tekstu
input_text = "Explain the principles of quantum mechanics."
output = generator(input_text, max_length=100, temperature=0.7, top_k=50)
print("Generated text:")
print(output[0]["generated_text"])

#%%Model czarka

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Nazwa modelu na Hugging Face
model_name = "lzw1008/ConspEmoLLM-7b"

# Ładowanie tokenizera
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Ładowanie modelu z automatycznym przypisaniem do GPU/CPU i kwantyzacją 8-bitową
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


