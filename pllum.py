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

def ask_bielik(prompt: str, max_new_tokens=1000) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,   # możesz zmienić
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]

prompt = "opowiedz mi o mickiewiczu adamie"
response = ask_bielik(prompt)
print(response)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Nazwa modelu
model_name = "CYFRAGOVPL/PLLuM-12B-nc-instruct"

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

# Pipeline do generowania tekstu
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

# Funkcja do zadawania pytań
def ask_pllum(prompt: str, max_new_tokens=500) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=0.7,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]

# Przykład użycia
prompt = "Wyjaśnij zasadę działania silnika spalinowego."
response = ask_pllum(prompt)
print(response)

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
        temperature=0.8,
        top_p=0.7,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]

# Przykład użycia
prompt = "opowiedz mi o mickiewiczu adamie"
response = ask_llama_pllum(prompt)
print(response)

import datetime

# Oblicz bieżący rok
current_year = datetime.datetime.now().year

# Dynamically insert the current year into the prompt:
interpret_prompt = f"""Jesteś asystentem bibliotecznym.

Masz wyłącznie ustalić, czy w pytaniu użytkownika występuje:
- tytuł,
- autor,
- tematyka,
- zakres dat (year range),
- pojedyncza data (year),
- miejsce wydania (place),
- wydawca (publisher),
- język (language),
- typ dokumentu (document_type),
lub jakiekolwiek kombinacje powyższych.

Wynik przedstaw **wyłącznie** w formacie:
[author: "nazwisko"], [subject: "temat"], [title: "tytuł"], [year: "RRRR"], [year_range: "RRRR-RRRR"], [place: "miejsce wydania"], [publisher: "wydawca"], [language: "język"], [document_type: "typ dokumentu"]

Jeśli pytanie zawiera wiele aspektów, zwróć je osobno, np.:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"], [publisher: "Wydawnictwo Naukowe PWN"], [language: "polski"], [document_type: "książka"]

Nie powtarzaj pytania użytkownika, nie dodawaj komentarzy, nie zmieniaj nazwisk, tytułów, lat, miejsc wydania, wydawcy, języka ani typu dokumentu.
Nie wpisuj kluczy, których nie ma w pytaniu.

Dodatkowe zasady interpretacji dat:
1. Jeśli w pytaniu jest „ostatnich X lat”, przyjmij, że obecny rok to {current_year}, więc zapisz [year_range: "{current_year} - X + 1-{current_year}"].
2. Jeśli w pytaniu jest „przed rokiem YYYY”, przyjmij [year_range: "0000-(YYYY-1)"].
3. Jeśli w pytaniu jest „po roku YYYY”, przyjmij [year_range: "(YYYY+1)-9999"].
4. Jeśli w pytaniu jest „w roku YYYY”, przyjmij [year: "YYYY"].
5. Jeśli w pytaniu jest „od roku YYYY do roku ZZZZ”, przyjmij [year_range: "YYYY-ZZZZ"].
6. Jeśli nie da się jednoznacznie ustalić, nie wpisuj year ani year_range.

Przykłady:

1. Pytanie użytkownika: "Czy znajdę książki Mickiewicza o przyrodzie z lat 1830-1840 wydane w Warszawie?"
Odpowiedź:
[author: "mickiewicz"], [subject: "przyroda"], [year_range: "1830-1840"], [place: "warszawa"]

2. Pytanie użytkownika: "Szukam czasopism o astronomii wydanych w Krakowie."
Odpowiedź:
[subject: "astronomia"], [place: "Kraków"], [document_type: "czasopismo"]

3. Pytanie użytkownika: "Czy macie artykuły naukowe o sztucznej inteligencji z lat 2010-2020?"
Odpowiedź:
[subject: "sztuczna inteligencja"], [year_range: "2010-2020"], [document_type: "artykuł naukowy"]

4. Pytanie użytkownika: "Gdzie mogę znaleźć ebooki Stephena Kinga w języku angielskim?"
Odpowiedź:
[author: "Stephen King"], [language: "angielski"], [document_type: "ebook"]

5. Pytanie użytkownika: "Czy macie raporty o zmianach klimatycznych wydane przez ONZ?"
Odpowiedź:
[subject: "zmiany klimatyczne"], [publisher: "ONZ"], [document_type: "raport"]

6. Pytanie użytkownika: "Szukam audiobooków z powieściami kryminalnymi."
Odpowiedź:
[subject: "powieści kryminalne"], [document_type: "audiobook"]

7. Pytanie użytkownika: "Czy macie raporty ekonomiczne wydane przez Bank Światowy oraz Forum Ekonomiczne?"
Odpowiedź:
[subject: "ekonomia"], [publisher: "Bank Światowy"], [publisher: "Forum Ekonomiczne"], [document_type: "raport"]

8. Pytanie użytkownika: "Szukam książek o historii Polski lub historii Europy wydanych w latach 2000-2010."
Odpowiedź:
[subject: "historia Polski"], [subject: "historia Europy"], [year_range: "2000-2010"], [document_type: "książka"]

9. Pytanie użytkownika: "Czy macie artykuły naukowe o medycynie oraz raporty o zdrowiu publicznym?"
Odpowiedź:
[subject: "medycyna"], [document_type: "artykuł naukowy"], [subject: "zdrowie publiczne"], [document_type: "raport"]

10. Pytanie użytkownika: "Czy macie książki Stephena Kinga wydane przed 2000 rokiem?"
Odpowiedź:
[author: "Stephen King"], [year_range: "0000-1999"], [document_type: "książka"]

11. Pytanie użytkownika: "Czy macie artykuły naukowe o psychologii z ostatnich 5 lat?"
Odpowiedź:
[subject: "psychologia"], [year_range: "{current_year - 4}-{current_year}"], [document_type: "artykuł naukowy"]

Pytanie użytkownika, interesuje Ci tylko ono, nic od siebie nie dodawaj, żadnych pytań, jeśli użytkownik używa imienia i nazwiska używaj ich obu, wszystko zawsze w mianowniku, pamiętaj o zasadach dotyczących dat i obecnym roku {current_year}!: {{user_question}}
Odpowiedź:
"""


if __name__ == "__main__":
    #user_query = "Czy znajdę książki o grach komputerowych?"
    user_query = "„Czy macie książki napisane przez Bolesława Prusa, wydane w Warszawie w roku 1890?"
    user_query ="Szukam książek o historii Polski w języku angielskim, wydanych przez wydawnictwo Penguin."
    user_query ="Czy macie czasopisma o astronomii wydane w Krakowie?"
    user_query ="Czy macie artykuły naukowe o medycynie oraz raporty o zdrowiu publicznym?"
    user_query ="Czy macie raporty ekonomiczne wydane przez Bank Światowy oraz Forum Europejskie?"
    user_query ="Czy macie czasopisma o technologii oraz artykuły naukowe o sztucznej inteligencji?"
    user_query ="Czy macie książki Stephena Kinga wydane przed 2000 rokiem oraz artykuły naukowe o psychologii z ostatnich 5 lat?"
    user_query ="Czy znajdę ebooki w języku angielskim autorstwa J.K. Rowling?"
    interp = interpret_prompt.format(user_question=user_query)

    print(">>> interpret_prompt:")
    print(interp)

    interpretation = ask_llama_pllum(interp, max_new_tokens=150)
  #  interpretation = ask_pllum(interp, max_new_tokens=150)
    print("=== INTERPRETACJA ===")
    print(interpretation)
    
    
    
    
    
from ctransformers import AutoModelForCausalLM

# Nazwa modelu
model_path = "speakleash/Bielik-11B-v2.3-Instruct-GGUF"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type="llama",
    gpu_layers=55,
    max_context_size=2048  # jeżeli biblioteka to wspiera
)

# Wczytanie modelu bez ponownej kwantyzacji
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    
    gpu_layers=55
)

# Funkcja do zadawania pytań
def ask_bielik(prompt: str, max_new_tokens=600) -> str:
    return model(
        prompt,
        max_new_tokens=max_new_tokens,

        top_p=0.7,
        repetition_penalty=1.2
    )

# Przykład użycia
prompt = "opowiedz mi o mickiewiczu adamie"
response = ask_bielik(prompt)
print(response)


if __name__ == "__main__":
    #user_query = "Czy znajdę książki o grach komputerowych?"
    user_query = "„Czy macie książki napisane przez Bolesława Prusa, wydane w Warszawie w roku 1890?"
    user_query ="Szukam książek o historii Polski w języku angielskim, wydanych przez wydawnictwo Penguin."
    user_query ="Czy macie czasopisma o astronomii wydane w Krakowie?"
    user_query ="Czy macie artykuły naukowe o medycynie oraz raporty o zdrowiu publicznym?"
    user_query ="Czy macie raporty ekonomiczne wydane przez Bank Światowy oraz Forum Europejskie?"
    user_query ="Czy macie czasopisma o technologii oraz artykuły naukowe o sztucznej inteligencji?"
    user_query ="Czy macie książki Stephena Kinga wydane przed 2000 rokiem oraz artykuły naukowe o psychologii z ostatnich 5 lat?"
    user_query ="Czy znajdę ebooki w języku angielskim autorstwa J.K. Rowling?"
    interp = interpret_prompt.format(user_question=user_query)

    print(">>> interpret_prompt:")
    print(interp)

    interpretation = ask_bielik(interp, max_new_tokens=150)
    
#%% FALCON   
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "tiiuae/Falcon3-10B-Instruct"

# Ładujemy tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Ładujemy model w trybie 8-bitowym na GPU (oszczędność VRAM)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True
)

# Tworzymy pipeline do generowania tekstu
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

def ask_falcon(prompt: str, max_new_tokens=150) -> str:
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,   # Możesz dostosować
        top_p=0.9,
        repetition_penalty=1.2
    )
    return output[0]["generated_text"]

# Przykładowe zapytanie do modelu:
prompt = "who is mickiewicz?"
response = ask_falcon(prompt)
print(response)
import datetime

# Calculate the current year
current_year = datetime.datetime.now().year

# Dynamically insert the current year into the prompt:
interpret_prompt = f"""You are a library assistant.

Your task is solely to determine whether the user's question contains:
- a title,
- an author,
- a subject,
- a date range (year range),
- a single year (year),
- a place of publication (place),
- a publisher,
- a language,
- a document type,
or any combination of the above.

Present the result **only** in the following format:
[author: "last name"], [subject: "topic"], [title: "title"], [year: "YYYY"], [year_range: "YYYY-YYYY"], [place: "place of publication"], [publisher: "publisher"], [language: "language"], [document_type: "document type"]

If the question contains multiple aspects, list them separately, for example:
[author: "mickiewicz"], [subject: "nature"], [year_range: "1830-1840"], [place: "Warsaw"], [publisher: "Scientific Publishing House PWN"], [language: "Polish"], [document_type: "book"]

Do not repeat the user’s question, do not add any comments, and do not change names, titles, years, places of publication, publishers, languages, or document types.
Do not include keys that are not explicitly mentioned in the question.

### **Additional date interpretation rules:**
1. If the question mentions "the last X years," assume the current year is {current_year}, and record it as [year_range: "{current_year} - X + 1-{current_year}"].
2. If the question asks for "before year YYYY," record it as [year_range: "0000-(YYYY-1)"].
3. If the question asks for "after year YYYY," record it as [year_range: "(YYYY+1)-9999"].
4. If the question asks for "in the year YYYY," record it as [year: "YYYY"].
5. If the question asks for "from year YYYY to year ZZZZ," record it as [year_range: "YYYY-ZZZZ"].
6. If it is not possible to determine the year or range unambiguously, do not include year or year_range.

### **Examples:**

1. **User question:** "Can I find books by Mickiewicz about nature from 1830-1840 published in Warsaw?"
   **Answer:**
   [author: "mickiewicz"], [subject: "nature"], [year_range: "1830-1840"], [place: "Warsaw"]

2. **User question:** "I'm looking for journals about astronomy published in Kraków."
   **Answer:**
   [subject: "astronomy"], [place: "Kraków"], [document_type: "journal"]

3. **User question:** "Do you have scientific articles on artificial intelligence from 2010-2020?"
   **Answer:**
   [subject: "artificial intelligence"], [year_range: "2010-2020"], [document_type: "scientific article"]

4. **User question:** "Where can I find ebooks by Stephen King in English?"
   **Answer:**
   [author: "Stephen King"], [language: "English"], [document_type: "ebook"]

5. **User question:** "Do you have reports on climate change published by the UN?"
   **Answer:**
   [subject: "climate change"], [publisher: "UN"], [document_type: "report"]

6. **User question:** "I'm looking for audiobooks with crime novels."
   **Answer:**
   [subject: "crime novels"], [document_type: "audiobook"]

7. **User question:** "Do you have economic reports published by the World Bank and the Economic Forum?"
   **Answer:**
   [subject: "economy"], [publisher: "World Bank"], [publisher: "Economic Forum"], [document_type: "report"]

8. **User question:** "I'm looking for books on the history of Poland or the history of Europe published between 2000-2010."
   **Answer:**
   [subject: "history of Poland"], [subject: "history of Europe"], [year_range: "2000-2010"], [document_type: "book"]

9. **User question:** "Do you have scientific articles on medicine and reports on public health?"
   **Answer:**
   [subject: "medicine"], [document_type: "scientific article"], [subject: "public health"], [document_type: "report"]

10. **User question:** "Do you have books by Stephen King published before the year 2000?"
    **Answer:**
    [author: "Stephen King"], [year_range: "0000-1999"], [document_type: "book"]

11. **User question:** "Do you have scientific articles on psychology from the last 5 years?"
    **Answer:**
    [subject: "psychology"], [year_range: "{current_year - 4}-{current_year}"], [document_type: "scientific article"]

The user’s question is of primary importance. Do not add anything extra, do not ask any questions. If the user provides both a first and last name, use both. Always use the nominative case. Remember the date interpretation rules and the current year {current_year}!  

User question: {{user_question}}

Answer:
"""

interpret_prompt = """Extract relevant metadata from the user's query.

Format:
[author: "author name"], [subject: "topic"], [title: "title"], [year: "YYYY"], 
[year_range: "YYYY-YYYY"], [place: "publication place"], [publisher: "publisher"], 
[language: "language"], [document_type: "document type"]

User question: {user_question}
Answer:
"""






if __name__ == "__main__":
    user_query = "Do you have books written by Bolesław Prus, published in Warsaw in the year 1890?"
    user_query = "I'm looking for books on the history of Poland in English, published by Penguin."
    user_query = "Do you have journals about astronomy published in Kraków?"
    user_query = "Do you have scientific articles on medicine and reports on public health?"
    user_query = "Do you have economic reports published by the World Bank and the European Forum?"
    user_query = "Do you have journals on technology and scientific articles on artificial intelligence?"
    user_query = "Do you have books by Stephen King published before the year 2000 and scientific articles on psychology from the last 5 years?"
    user_query = "Can I find ebooks in English by J.K. Rowling?"
    interp = interpret_prompt.format(user_question=user_query)

    print(">>> interpret_prompt:")
    print(interp)
    response = ask_falcon(interp)
  