# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:49:17 2025

@author: darek
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch, json, re
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

ROOT_DIR   = Path(r"C:/pdf_llm_do_roboty/")
device = "cuda" # the device to load the model onto
MAX_TOK = 1300
OVERLAP = 150

# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2-7B-Instruct",
#     torch_dtype="auto",
#     device_map="auto"
# )
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")



bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

attention_mask = model_inputs["attention_mask"]

generated_ids = model.generate(
    input_ids=model_inputs["input_ids"],
    attention_mask=attention_mask,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.0,
    pad_token_id=tokenizer.eos_token_id  # ważne!
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

tok = tokenizer

PROMPT_TMPL = """
You are a strict technical tool extractor for English academic texts. Follow these rules carefully.

🚫 HARD EXCLUSIONS — IMMEDIATE REJECTION IF:
- Contains personal names (any capitalized name like Smith or Kowalska)
- Literary/humanities content without clear technical tools !!!
- Journal or magazine names (e.g., “Journal”, “Review”, “Studies”)
- Book or article titles, literary works, or publishers (e.g., “Middlemarch”, “Penguin”)
- Conference names or acronyms (e.g., CLARIN, ACL, LREC)
- Anything that looks like a citation (e.g., “Heuser_2016”, “Morselli_1975”)!

✅ ONLY EXTRACT:
1. Clearly named software tools or code libraries (e.g., "QGIS" in text → "QGIS"; "spaCy" in text → "spaCy") – Extract ONLY terms that literally appear in the text!
2. Clearly named NLP/ML models (e.g., "BERT" in text → "BERT"; "RoBERTa" in text → "RoBERTa") – Extract ONLY terms that literally appear in the text!
3. Clearly named datasets or corpora (e.g., "NKJP" in text → "NKJP"; "MultiEmo" in text → "MultiEmo") – Extract ONLY terms that literally appear in the text!
4. Clearly named data formats or markup standards (e.g., "XML" in text → "XML"; "JSON" in text → "JSON") – Extract ONLY terms that literally appear in the text!
5. Clearly named repositories or portals (e.g., "Wikipedia" in text → "Wikipedia"; "Polona" in text → "Polona") – Extract ONLY terms that literally appear in the text!

🛡️ VALIDATION RULES:
1. Extract ONLY terms that **literally appear in the text** — NO interpretation, NO guesswork!!!!!!!!
2. Do NOT reuse or copy terms from the examples unless they are present in the current input
3. Reject any term that looks like a person name (Firstname Lastname or similar)
4. If a term is unclear or looks suspicious, exclude it

‼️ STRICT BAN ON HALLUCINATION:
1. Do not invent software names under any circumstances
2. **NO HALLUCINATIONS**: If unsure → EXCLUDE. 
3. **NO EXAMPLES**: Never copy "QGIS", "spaCy", etc., unless **LITERALLY** in the text.
4. **NO GENERALIZATIONS**: "XML parser" → only "XML" if "XML" is standalone in text.
5. **CASE-SENSITIVE**: "Space" ≠ "spaCy", "Python" ≠ "python".

📌 EXAMPLES!!! (these are just illustrative!! — DO NOT COPY unless they appear in the actual input!):
text: We analyzed data using XML and spaCy 3.1  
→ ["XML", "spaCy"]  here rememebr, "SPACE" is not "spaCy"!!!!

text: Compared Proust and Morante with help from the Oxford edition  
→ []

text: Results were presented at the CLARIN conference  
→ []

text: Metadata was sourced from Wikipedia and Polona  
→ ["Wikipedia", "Polona"]
text: The space between words was analyzed  
→ []  ← "SPACE" is not "spaCy"!!! Reject!

💻 OUTPUT FORMAT:
- Return ONE valid JSON list (e.g., ["QGIS", "XML"]- this example if there is no XML or spaCy in text don't give me XML or spaCy!!!!!!)
- Do not include duplicates or explanations
- Return [] if no valid TOOLS or SOFTWARE are clearly mentioned Extract ONLY terms that **literally appear in the text** — NO interpretation, NO guesswork don't invent!
- Do NOT output common NLP names like ‘spaCy’, ‘BERT’, etc., unless they explicitly appear in the input
- NO interpretation, NO guesswork DON'T invent! Extract ONLY terms that **literally appear in the text**!
### TASK:
text:
{chunk}
software:
"""


'''LAST CHANGE 08.07.2025: ‼️ STRICT BAN ON HALLUCINATION:
- Do not invent software names under any circumstances
- **NO EXAMPLES**: Never copy "QGIS", "spaCy", etc., unless **LITERALLY** in the text.
- Do not generalize or "guess" based on context
- Do not include example tools unless they are present in the input!!!'''


STOP: set[str] = {
    # zmienne / abstrakty
    "city", "village", "author_gender", "publication_year", "emotions",
    "crowdsourcing", "sentiment_analysis", "symbolic_models", "text", "software",

}

# ---------- (LUŹNY) REGEX ------------------------------------------
# – pozwala na litery/cyfry/_ . – i -, min 2 znaki
RE_NAME = re.compile(r'^[A-Za-z0-9_.\-]{2,}$')
def pdf_to_text(pdf: Path) -> str:
    try:
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages)
    except Exception as e:
        print(f"Błąd wczytywania {pdf.name}: {e}")
        return ""

# --------------- CHUNK ----------------------
def chunk(txt: str, max_tok=MAX_TOK, overlap=OVERLAP):
    buf, t = [], 0
    for w in txt.split():
        t += len(tok(w, add_special_tokens=False).input_ids)
        buf.append(w)
        if t >= max_tok:
            yield " ".join(buf)
            buf, t = buf[-overlap:], sum(len(tok(x, add_special_tokens=False).input_ids) for x in buf)
    if buf:
        yield " ".join(buf)
# ---------- FILTR ---------------------------------------------------
def is_software(name: str) -> bool:
    """
    True  → zachowaj nazwę
    False → odrzuć
    """
    n = name.strip()
    l = n.lower()

    # 1) słowa z listy blokującej
    if l in STOP:
        return False

    # 2) nagie lata (np. "1975")
    if re.fullmatch(r'\d{3,4}', l):
        return False

    # 3) wzorzec nazwisko+rok (np. "herrmann2019")
    if re.fullmatch(r'[a-z]+[0-9]{4}', l):
        return False

    # ► nic nie zablokowało – zachowaj
    return True


def parse(reply: str) -> set[str]:
    m = re.search(r'\[[^\[\]]+\]', reply)
    if not m: return set()
    try:
        lst = json.loads(m.group(0))
    except Exception:
        return set()
    return {x.strip() for x in lst if isinstance(x, str) and is_software(x.strip())}

# ------------- CALL LLM ---------------------
# def extract_names(chunk_txt: str) -> set[str]:
#     prompt = PROMPT_TMPL.format(chunk=chunk_txt)
#     ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
#                                   add_generation_prompt=True,
#                                   return_tensors="pt").to(model.device)
#     gen = model.generate(ids, attention_mask=torch.ones_like(ids),
#                          max_new_tokens=200)
#     return parse(tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True))

def extract_names(chunk_txt: str) -> set[str]:
    prompt = PROMPT_TMPL.format(chunk=chunk_txt)

    # 1. Utwórz tekst promptu (z template)
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False
    )

    # 2. Ztokenizuj prompt ręcznie (żeby dostać też attention_mask)
    model_inputs = tokenizer(chat_text, return_tensors="pt", padding=True).to(model.device)

    # 3. Generacja z poprawnymi danymi
    gen = model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=200,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    # 4. Odcinanie promptu od outputu
    decoded = tokenizer.decode(
        gen[0][model_inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return parse(decoded)

# ------------- PIPELINE ---------------------
def process_pdf(pdf: Path) -> dict:
    text = pdf_to_text(pdf)
    chs = list(chunk(text))

    if chs:
        print(f"\n--- {pdf.name} ⋅ first chunk preview ---")
        print(chs[0][:200].replace("\n", " ") + " …\n")

    names = set()
    for ch in chs:
        names |= extract_names(ch)

    return {
        "text_id": str(pdf),  # pełna ścieżka do pliku
        "software_detected": bool(names),
        "list_of_software": sorted(names)
    }

def pdf_files(root: Path):
    return [root] if root.is_file() else sorted(root.rglob("*.pdf"))

    # sprawdzenie, ile PDF-ów zostało zebranych
files = pdf_files(ROOT_DIR)
print(f"Znaleziono {len(files)} plików PDF w {ROOT_DIR}:")
for p in files:
    print("  ", p)
    
import shutil

BAD_PDF_DIR = Path("C:/pdf_zrobione/błedne_do sprawdzenia_dhq/")
BAD_PDF_DIR.mkdir(parents=True, exist_ok=True)

clean_files = []
bad_count = 0
ok_count = 0

for i, pdf_path in enumerate(files, 1):
    move = False
    try:
        reader = PdfReader(str(pdf_path))
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"

        num_G = len(re.findall(r'/G\d{2,3}', full_text))
        num_slash = full_text.count("/")

        # 🔁 Łagodniejsze warunki
        if not full_text.strip():
            move = True
        elif num_G > 150 and num_G > 0.9 * num_slash:  # podniesione progi
            move = True
        elif num_slash > 300:  # wyższy limit ukośników
            move = True

    except Exception as e:
        print(f"❌ Błąd czytania {pdf_path.name}: {e}")
        move = True

    if move:
        target = BAD_PDF_DIR / pdf_path.name
        shutil.move(str(pdf_path), str(target))
        print(f"[{i}] 🚫 TREFNY: {pdf_path.name}")
        bad_count += 1
    else:
        clean_files.append(pdf_path)
        print(f"[{i}] ✓ OK: {pdf_path.name}")
        ok_count += 1
        
files = clean_files
print(f"\n✓ Pozostało {len(files)} poprawnych PDF-ów do analizy.")
print(f"✓ Łącznie poprawnych: {ok_count}, odrzuconych: {bad_count}") 
#results = [process_pdf(p) for p in tqdm(pdf_files(ROOT_DIR), desc="PDFs")]
results = []
for p in tqdm(files, desc="PDFs"):
    rec = process_pdf(p)
    print(rec)
    results.append(rec)
        # od razu dopisujemy do pliku, żeby nie stracić