# -*- coding: utf-8 -*-
import pandas as pd
import json
from tqdm import tqdm 
csv_file = "C:/Users/darek/Downloads/500 a.csv"  # Ścieżka do pliku

# Wczytanie pliku i ustawienie nowych nagłówków
df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip", quotechar='"', header=0, names=["id", "field500"])
df_filtered = df[df["field500"].astype(str).str.startswith("Tytuł oryginału:")]
# Wyświetlenie danych
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Nazwa modelu
model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"

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
model.eval()
# Pipeline do generowania tekstu
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# Funkcja do zadawania pytań
def ask_llama_pllum(prompt: str, max_new_tokens=1000) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]



# 2. Dopiero potem tworzysz f-string, który z niej korzysta:
# 1. Szablon promptu
interpret_prompt = """Jesteś specjalistą bibliografem. Chcesz wydobyć tytuły oryginalnych utworów z pola 500a w formacie MARC21. 
Tytuły zwykle zaczynają się od słów "Tytuł oryginału:", a następnie może występować sam tytuł, numer woluminu lub tytuł cyklu.

Oto przykłady:
- "Tytuł oryginału: La disparition. Rok wydania na podstawie strony internetowej wydawcy." 
  → wydobywasz "La disparition." (bez roku)
- "Tytuł oryginału: The winds of Dune. Stanowi kontynuację sagi Kroniki Diuny Franka Herberta."
  → wydobywasz "The winds of Dune."
- "Tytuł oryginału: Noir burlesque. 2"
  → wydobywasz "Noir burlesque. 2"
- "Tytuł oryginału: Bloody Mary. 3. Numeracja stron od prawej do lewej."
  → wydobywasz "Bloody Mary. 3."
- "Tytuł oryginału: D.O.G.S. Kontynuacja powieści pt:. S.T.A.G.S."
  → wydobywasz "D.O.G.S."
- "Tytuł oryginału: Memoria, 2023. Tytuł oryginału cyklu: En Rekke/Vargas deckare."
  → wydobywasz "Memoria" w kluczu "title", "2023" w kluczu "year", a "En Rekke/Vargas deckare" w kluczu "series_title" – nie powtarzaj tego w "title"!

**Format odpowiedzi** musi być wyłącznie w postaci JSON, na przykład:

{{
  "title": ["A prison diary.", "Vol. 1, Belmarsh: hell", "Vol. 2, Wayland: purgatory", "Vol. 3, North Sea Camp: heaven"],
  "year": "1991",
  "series_title": "The vampire diaries"
}}

### Zasady:
1. **Zawsze zwróć klucz `"title"`** – nawet jeśli tytułów nie da się ustalić, zwróć `"title": []`.
2. Jeśli występuje `"Tytuł oryginału cyklu:"`, zapisz wartość w `"series_title"`, **nie** umieszczaj tego w `"title"`.
3. Jeśli w tekście pojawia się rok (np. po tytule, "Memoria, 2023."), zapisz go w `"year"` – ale **nie** w kluczu `"title"`.
4. Nigdy nie zwracaj `null`. Jeśli jakaś informacja nie występuje w tekście, **pomijaj** ten klucz.
5. Nie dodawaj żadnych innych informacji czy wyjaśnień – **ma być samo JSON**.
6. Jeśli masz "Tytuł oryginału:" i coś dalej, zawsze to wydobądź (z pominięciem słowa "Tytuł oryginału:").
7. **Nie** twórz osobnych kluczy dla słów typu "Kontynuacja powieści" czy "Rok wydania" – te informacje ignoruj, jeśli nie wnoszą nowych tytułów.

Teraz zadanie dla Ciebie: Skup się tylko na tym – nie wracaj do przykładów.  
Poniżej masz treść, na podstawie której masz wygenerować odpowiedź:

{user_question}

Odpowiedź:

"""


# 2. Przykładowy "user_question
user_query = "Tytuł oryginału: Memoria, 2023. Tytuł oryginału cyklu: En Rekke/Vargas deckare. "


# 3. Wstawianie treści do promptu
interp = interpret_prompt.format(user_question=user_query)
interpretation = ask_llama_pllum(interp, max_new_tokens=500)
# 4. Sprawdź efekt
print(interpretation)

df_small = df_filtered.head(100)

results = []

for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Przetwarzanie rekordów"):
    # 4a. Wstawiamy aktualny 'field500' do promptu
    user_query = row["field500"]
    print(user_query)
    interp = interpret_prompt.format(user_question=user_query)

    # 4b. Wywołujemy model
    interpretation = ask_llama_pllum(interp, max_new_tokens=500)
    
    # 4c. Dodajemy do listy wyników np. w formacie:
    # {"id": 123, "interpretation": {...}}
    results.append({
        "id": row["id"],
        "field 500": user_query, 
        "interpretation": interpretation
    })

df_results = pd.DataFrame(results)

# 1. Zapis do JSON
for item in results:
    interpretation_str = item.get("interpretation", "")
    print(interpretation_str)
    if interpretation_str:
        try:
            # Zamiana surowego tekstu JSON w Pythonowy słownik
            item["interpretation"] = json.loads(interpretation_str)
        except json.JSONDecodeError:
            # Jeśli łańcuch jest nieprawidłowy, można ustawić None lub inny fallback
            item["interpretation"] = None

json_output_file = "C:/Users/darek/Downloads/interpretation_results_fixed.json"
with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Zapisano wyniki w formacie JSON do: {json_output_file}")

# 2. Zapis do Excela
excel_output_file = "C:/Users/darek/Downloads/interpretation_results.xlsx"
df_results.to_excel(excel_output_file, index=False)

print(f"Zapisano wyniki do pliku {json_output_file} i {excel_output_file}")
