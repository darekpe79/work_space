from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch, json, re
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

# ------------------- KONFIG -------------------
ROOT_DIR   = Path(r"D:/Nowa_praca/pdfy_oprogramowanie")
OUT_FILE   = ROOT_DIR / "Facta Universitatis, Series_ Linguistics.json"

MODEL_ID   = "tiiuae/Falcon3-10B-Instruct" #"mistralai/Mistral-7B-Instruct-v0.3"#
USE_INT8   = True
TEMPERATURE = 0.0
MAX_TOK    = 1300
OVERLAP    = 150
# ----------------- LOGIN ---------------------
 # ← Twój token prywatny

# ----------------- MODEL ---------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)  #, trust_remote_code=True

if USE_INT8:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto"
    ) #, trust_remote_code=True
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )

# model.generation_config.update(
#     temperature=TEMPERATURE, top_p=0.9, do_sample=True,
#     pad_token_id=tok.eos_token_id
# )
model.generation_config.update(
    temperature=0.0,  # determinizm
    do_sample=False,  # bez samplowania
    pad_token_id=tok.eos_token_id
)


# --------------- PDF -> TEXT -----------------
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

# ---------------- PROMPT ---------------------
# PROMPT_TMPL = """
# You are extracting software names from scientific or technical texts.

# ✓ Return only proper software names:
#  • software tools, standalone programs, code libraries
#  • APIs, NLP models, GIS/OCR tools
#  • scientific or linguistic processing packages

# ✘ Do NOT return:
#  • person names (e.g., Proust, Morante, Ginzburg, Herrmann, Gius)
#  • book or article titles (e.g., Dissipatio H.G., Proustisme)
#  • citations with years (e.g., Morselli_1975, Heuser_et_al_2016)
#  • datasets or corpora (e.g., ELTeC, NKJP, Dariah-PL, MultiEmo)
#  • data formats or markup types (e.g., TEI, XML, CSV)
#  • abstract terms (e.g., narrative, city, status, emotions)
#  • publishers, projects, conferences (e.g., CLARIN, ACL, LREC)
#  • internal variables or concept labels (e.g., cities, partition, sentiment)
#  • any invented or guessed names — only extract names clearly present in the text!!!
#  • INVENTED / guessed names ⇨ absolutely forbid hallucinations

# ❗ If the passage is **not primarily in English**, skip it and return [].
# Only extract actual software, models, or tools used for processing.

# Return exactly ONE valid JSON list of unique software names (no duplicates, no comments).
# If nothing is found, return: []

# ### good example
# text:
# We used spaCy 3, QGIS 3.28 and the PolDeepNer2 model.
# software:
# ["PolDeepNer2", "QGIS", "spaCy"]

# ### bad example 1
# text:
# City and publication_year are metadata variables.
# software:
# []

# ### bad example 2
# text:
# Herrmann 2019 processes the ELTeC corpus stored in XML.
# software:
# []

# ### task
# text:
# {chunk}
# software:
# """   
# PROMPT_TMPL = """
# You are extracting **software names** from **English** scientific or technical texts.

# ❗ If the passage is **not primarily in English**, return [] immediately.

# ✓ Return only actual digital tools or resources:
#   • software tools, standalone programs, code libraries  
#   • APIs, NLP models, GIS/OCR tools  
#   • scientific or linguistic processing packages  
#   • datasets and corpora used in processing  
#   • repositories or data portals (e.g., Wikipedia, Geonames, Polona)  
#   • data formats or markup types (e.g., TEI, XML, CSV)  

# ✘ Do NOT return — we don't want!:
#   • we don't want! person names (e.g., Proust, Morante, Ginzburg, James Smith, Anna Kowalska, Maria Curie)  
#   • we don't want! book or article titles, literary works, or publishers (e.g., *Middlemarch*, *Shakespeare*, *Penguin Books*)  
#   • we don't want! journal or magazine names (e.g., *Nineteenth-Century Gender Studies*, *Contemporary Review*, *Girl's Own Paper*)  
#   • we don't want! citations with years (e.g., *Morselli_1975*, *Heuser_et_al_2016*)  
#   • we don't want! abstract terms (e.g., narrative, city, status, emotions)  
#   • we don't want! publishers, funding bodies, or academic journals (e.g., De Gruyter, Springer, *Frontiers in Digital Humanities*)  
#   • we don't want! conferences or conference acronyms (e.g., CLARIN, ACL, **LREC**) — these are venues, not tools or datasets  
#   • we don't want! any invented or guessed names — **absolutely no hallucinations!** Only return names that clearly appear in the text

# ✅ Return exactly ONE JSON list of **unique** software/dataset names (no duplicates, no comments).
# If nothing is found, return: []

# ### good example 1
# text:
# We used spaCy 3, QGIS 3.28 and the PolDeepNer2 model.
# software:
# ["PolDeepNer2", "QGIS", "spaCy"]

# ### good example 2
# text:
# We applied our approach to the ELTeC corpus and exported data in TEI-XML format.
# software:
# ["ELTeC", "TEI", "XML"]

# ### good example 3
# text:
# The dataset was enriched with Wikipedia and Geonames metadata.
# software:
# ["Wikipedia", "Geonames"]

# ### bad example 1 — we don't want! conference + journal
# text:
# The project is supported by the CLARIN infrastructure and published in the *Frontiers in Digital Humanities* journal.
# software:
# []

# ### bad example 2 — we don't want! literary titles
# text:
# We analyzed *Middlemarch* using methods inspired by *Shakespeare* studies.
# software:
# []

# ### bad example 3 — we don't want! publisher
# text:
# The Penguin edition was preferred for annotation clarity.
# software:
# []

# ### bad example 4 — we don't want! conference name
# text:
# Results were presented at LREC 2022.
# software:
# []

# ### bad example 5 — we don't want! journals and dictionaries
# text:
# We used data from the *Contemporary Review* and consulted the *Oxford Dictionary of National Biography*.
# software:
# []

# ### bad example 6 — we don't want! magazine names
# text:
# This approach was described in *Girl's Own Paper* and *Atalanta* magazine.
# software:
# []

# ### bad example 7 — we don't want! journal name
# text:
# The article appeared in *Nineteenth-Century Gender Studies*.
# software:
# []

# ### task
# text:
# {chunk}
# software:
# """

# PROMPT_TMPL = """
# You are a strict technical tool extractor for English academic texts. Follow these rules:

# 🚨 IMMEDIATE REJECTION IF:
# - Contains personal names (any capitalized name like Smith or Kowalska)
# - Literary/humanities content without clear technical tools
# - Journal or magazine titles (e.g., "Journal", "Studies", "Review")
# - Publisher names (e.g., "Penguin", "Oxford Press")

# ✅ ONLY EXTRACT:
# 1. Known software tools or code libraries (e.g., only examples- Python, R, QGIS)
# 2. Datasets with suffixes or known formats (.csv, .json, _v2, _dataset)
# 3. Technical data formats or markups (e.g., XML, TEI, CSV)
# 4. Repositories or data portals (e.g., Wikipedia, Geonames, Polona)

# 🔍 VALIDATION STEPS:
# 1. Term must appear **clearly in the input text** in a technical context (e.g., "using X", "analyzed with Y")
# 2. Absolutely DO NOT hallucinate or guess software names — extract only what is **explicitly present**
# 3. Reject if term resembles a person name (Firstname Lastname)
# 4. Reject standalone capitalized words unless clearly technical or data-related

# ❌ UNDER NO CIRCUMSTANCES:
# - Do not invent names !!! NEVER INVENT!!!
# - Do not copy from the examples unless they appear in the actual input
# - Do not return anything unless it is 100% clearly mentioned in the passage

# 📌 EXAMPLES (for illustration only — do not reuse unless they appear in the text!!!!):
# text: Analyzed with NLTK 3.7 and StanfordNLP  
# → ["NLTK", "StanfordNLP"] - only example!

# text: Compared Proust and Morante using XML  
# → ["XML"] - example!

# text: Data from Christopher Tilley's archive  
# → [] - example!

# text: Published by Oxford University Press  
# → [] - example!

# 💻 OUTPUT FORMAT:
# - One valid JSON list of software/tool names from the passage (e.g., ["spaCy", "XML"])
# - NO duplicates, NO explanations
# - Return an empty list [] if you are not certain or nothing matches

# text:
# {chunk}
# software:
# """
# PROMPT_TMPL = """
# You are a PRECISION EXTRACTOR for technical tools in English texts. Follow these RIGID rules:

# ⚠️ ABSOLUTE PROHIBITIONS:
# 1. NEVER invent tools - ONLY extract EXPLICITLY mentioned terms
# 2. NEVER use example terms ("NLTK"/"StanfordNLP") in real extractions
# 3. NEVER guess - require DIRECT textual evidence

# 🌐 LANGUAGE REQUIREMENTS:
# ▸ INSTANTLY RETURN [] IF:
#    - Non-English text dominates
#    - Technical context <50% English
#    - No operational verbs ("use"/"process") in English

# 🛠️ EXTRACTION CRITERIA:
# ✅ ONLY ACCEPT:
#    - Tools WITH version numbers (spaCy 3.2)
#    - Formats WITH suffixes (TEI-XML)
#    - Repositories (Geonames)
#    - APIs WITH names (GPT-3 API)

# 🚫 ALWAYS REJECT:
#    - Example terms (marked below)
#    - Partial matches ("Python" in "Python book")
#    - Unversioned tools ("some Python code")

# 🔬 VALIDATION PROCESS:
# 1. CONFIRM term appears verbatim
# 2. VERIFY technical context:
#    - Operational verbs ("analyzed using X")
#    - Version markers (v1.2, 3.4.1)
# 3. CROSS-CHECK against:
#    - Blacklist (names/publishers)
#    - Whitelist (known tools)

# 📘 DEMONSTRATION EXAMPLES:
# NOTE: These are ILLUSTRATIONS ONLY - NEVER use these terms in actual extraction!

# text: "Processed with [EXAMPLE]NLTK 3.7[/EXAMPLE] and [EXAMPLE]StanfordNLP[/EXAMPLE]"
# → ["NLTK", "StanfordNLP"] (EXAMPLE ONLY)

# text: "Exported to TEI format"
# → ["TEI"] (REAL extraction)

# text: "Data from Christopher's database"
# → [] (No tools mentioned)

# ⚙️ OUTPUT REQUIREMENTS:
# - STRICT JSON list ONLY
# - NO hypothetical terms
# - NO example terms
# - EMPTY list if:
#   * No explicit tools found
#   * Language uncertain
#   * Context ambiguous

# text:
# {chunk}
# software:
# """

# PROMPT_TMPL = """
# You are a strict technical tool extractor for English academic texts. Follow these rules carefully.

# 🚫 HARD EXCLUSIONS — IMMEDIATE REJECTION IF:
# - Contains personal names (any capitalized name like Smith or Kowalska)
# - Literary/humanities content without clear technical tools
# - Journal or magazine names (e.g., “Journal”, “Review”, “Studies”)
# - Book or article titles, literary works, or publishers (e.g., “Middlemarch”, “Penguin”)
# - Conference names or acronyms (e.g., CLARIN, ACL, LREC)
# - Anything that looks like a citation (e.g., “Heuser_2016”, “Morselli_1975”)

# ✅ ONLY EXTRACT:
# 1. Clearly named software tools or code libraries (e.g., QGIS, spaCy)
# 2. NLP/ML models (e.g., BERT, RoBERTa, LaBSE, PolDeepNer2)
# 3. Datasets or corpora (e.g., NKJP, ELTeC, MultiEmo, PolEval)
# 4. Data formats and markup standards (e.g., XML, TEI, CSV, JSON)
# 5. Repositories or portals (e.g., Wikipedia, Geonames, Polona)

# 🛡️ VALIDATION RULES:
# 1. Extract ONLY terms that **literally appear in the text** — NO interpretation, NO guesswork
# 2. Do NOT reuse or copy terms from the examples unless they are present in the current input
# 3. Reject any term that looks like a person name (Firstname Lastname or similar)
# 4. If a term is unclear or looks suspicious, exclude it

# ‼️ STRICT BAN ON HALLUCINATION:
# - Do not invent software names under any circumstances
# - Do not generalize or "guess" based on context
# - Do not include example tools unless they are present in the input

# 📌 EXAMPLES (these are just illustrative — DO NOT COPY unless they appear in the actual input!):
# text: We analyzed data using XML and spaCy 3.1  
# → ["XML", "spaCy"] this is spacy, not space!

# text: Compared Proust and Morante with help from the Oxford edition  
# → []

# text: Results were presented at the CLARIN conference  
# → []

# text: Metadata was sourced from Wikipedia and Polona  
# → ["Wikipedia", "Polona"]

# 💻 OUTPUT FORMAT:
# - Return ONE valid JSON list (e.g., ["QGIS", "XML"])
# - Do not include duplicates or explanations
# - Return [] if no valid tools or software are clearly mentioned

# ### TASK:
# text:
# {chunk}
# software:
# """
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
1. Clearly named software tools or code libraries (e.g., QGIS, spaCy)- Extract ONLY terms that **literally appear in the text**!
2. NLP/ML models (e.g., BERT, RoBERTa, LaBSE, PolDeepNer2)- Extract ONLY terms that **literally appear in the text**!
3. Datasets or corpora (e.g., NKJP, ELTeC, MultiEmo, PolEval)- Extract ONLY terms that **literally appear in the text**!
4. Data formats and markup standards (e.g., XML, TEI, CSV, JSON)- Extract ONLY terms that **literally appear in the text**!
5. Repositories or portals (e.g., Wikipedia, Geonames, Polona)- Extract ONLY terms that **literally appear in the text**!

🛡️ VALIDATION RULES:
1. Extract ONLY terms that **literally appear in the text** — NO interpretation, NO guesswork!!!!!!!!
2. Do NOT reuse or copy terms from the examples unless they are present in the current input
3. Reject any term that looks like a person name (Firstname Lastname or similar)
4. If a term is unclear or looks suspicious, exclude it

‼️ STRICT BAN ON HALLUCINATION:
- Do not invent software names under any circumstances
- Do not generalize or "guess" based on context
- Do not include example tools unless they are present in the input!!!

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



# Extract software names mentioned in the text below.

# ✓ Include only actual software, tools, models, libraries, APIs used for processing
# ✓ Examples: NLP tools, GIS/OCR programs, scientific packages

# ✘ Exclude:
#  • person names (e.g., Proust, Herrmann)
#  • book titles or citations (e.g., Heuser_2016, Dissipatio H.G.)
#  • datasets or formats (e.g., ELTeC, CSV, TEI, XML)
#  • abstract terms or variables (e.g., city, narrative, emotions, status)
#  • conferences or projects (e.g., CLARIN, LREC, ACL)
#  • invented or guessed names — only list those clearly mentioned!!!!

# Return one JSON list of unique software names (no duplicates, no explanation).
# If no software is found, return: []

# ### task
# text:
# {chunk}
# software:
# """



# ------------- PARSER -----------------------
# ---------- BLOKADY -------------------------------------------------


# ---------- REGEX ---------------------------------------------------
STOP: set[str] = {
    # zmienne / abstrakty
    "city", "village", "author_gender", "publication_year", "emotions",
    "crowdsourcing", "sentiment_analysis", "symbolic_models", "text", "software",

}

# ---------- (LUŹNY) REGEX ------------------------------------------
# – pozwala na litery/cyfry/_ . – i -, min 2 znaki
RE_NAME = re.compile(r'^[A-Za-z0-9_.\-]{2,}$')

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
def extract_names(chunk_txt: str) -> set[str]:
    prompt = PROMPT_TMPL.format(chunk=chunk_txt)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    gen = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=200)
    return parse(tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True))

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

BAD_PDF_DIR = Path("D:/Nowa_praca/pdfy_oprogramowanie_zrobione/bledne_pdfy")
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
       # print(full_text)

        num_G = len(re.findall(r'/G\d{2,3}', full_text))
        num_slash = full_text.count("/")

        # Warunki przenoszenia pliku
        if not full_text.strip():
            move = True
        elif num_G > 50 and num_G > 0.7 * num_slash:
            move = True
        elif num_slash > 100:
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
OUT_FILE.parent.mkdir(exist_ok=True)
OUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n✓ Zapisano {len(results)} rekordów → {OUT_FILE}")

#%% działania na przetworzonym jsonie, usuwanie zrobionych plikow 

import json
from pathlib import Path
import shutil

# === KONFIGURACJA ===
JSON_PATH = Path("D:/Nowa_praca/pdfy_oprogramowanie/IBERICA.json")
BASE_DIR  = Path("D:/Nowa_praca/pdfy_oprogramowanie")
TARGET_DIR = BASE_DIR / "DONE"

# Upewnij się, że katalog docelowy istnieje
TARGET_DIR.mkdir(exist_ok=True)

# Wczytaj listę przetworzonych plików PDF
with open(JSON_PATH, encoding="utf-8") as f:
    records = json.load(f)

# Iteruj po rekordach
for rec in records:
    pdf_path = Path(rec["text_id"])

    # Sprawdź, czy PDF istnieje
    if pdf_path.exists() and pdf_path.suffix.lower() == ".pdf":
        # Katalog docelowy z zachowaniem struktury
        rel_path = pdf_path.relative_to(BASE_DIR)
        new_path = TARGET_DIR / rel_path

        # Utwórz podkatalogi, jeśli trzeba
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Przenieś PDF
        shutil.move(str(pdf_path), str(new_path))
        print(f"✓ przeniesiono: {pdf_path} → {new_path}")
    else:
        print(f"⚠️ pominięto (nie znaleziono pliku): {pdf_path}")

# You must return **only software names** (stand-alone programs, libraries, APIs, NLP/GIS/OCR tools).

# ✘ IGNORE and NEVER list:
#  • person surnames (e.g. Herrmann, Gius, Morariu, Jacobs, Klinger)
#  • person-with-year citations (e.g. herrmann2019, bode2017, Herrmann_et_al_2022)
#  • datasets / text collections (e.g. ELTeC, NKJP, Dariah-PL)
#  • data / markup formats (e.g. XML, TEI, CSV)
#  • generic nouns or abstract concepts (city, emotions, narrative, status, love)
#  • publisher, project or conference acronyms (CLARIN, LREC, ACL)

# Return exactly ONE JSON list of unique names, no comments. If nothing → [].
# """

# GOOD_EX_IN  = "We used spaCy 3, QGIS 3.28 and the PolDeepNer2 model."
# GOOD_EX_OUT = '["PolDeepNer2","QGIS","spaCy"]'

# BAD1_IN = "City and publication_year are metadata variables."
# BAD2_IN = "Herrmann 2019 processes the ELTeC corpus stored in XML."

# PROMPT_TMPL = f"""
# ### system
# {SYSTEM}

# ### good example
# text:
# {GOOD_EX_IN}
# software:
# {GOOD_EX_OUT}

# ### bad example 1
# text:
# {BAD1_IN}
# software:
# []

# ### bad example 2
# text:
# {BAD2_IN}
# software:
# []

# ### task
# text:
# {{chunk}}
# software:
# """

# # ---- 4  CALL LLM (bez parsowania) ----
# def llm_reply(chunk_txt: str) -> str:
#     prompt = PROMPT_TMPL.format(chunk=chunk_txt)
#     ids = tok.apply_chat_template([{"role":"user","content":prompt}],
#                                   add_generation_prompt=True,
#                                   return_tensors="pt").to(model.device)
#     gen = model.generate(ids, attention_mask=torch.ones_like(ids),
#                          max_new_tokens=120)
#     return tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True)

# # ---- 5  PIPELINE ----
# def process_pdf(pdf: Path):
#     replies = []
#     for idx, ch in enumerate(chunk(pdf_to_text(pdf)), 1):
#         raw = llm_reply(ch)
#         replies.append({"chunk_id": f"{pdf.name}::chunk{idx}", "raw_reply": raw})
#     return replies

# def pdf_files(root: Path):
#     return [root] if root.is_file() else sorted(root.rglob("*.pdf"))

# all_raw = []
# for pdf in tqdm(pdf_files(ROOT_DIR), desc="PDFs"):
#     all_raw.extend(process_pdf(pdf))

# OUT_FILE.parent.mkdir(exist_ok=True)
# OUT_FILE.write_text(json.dumps(all_raw, indent=2, ensure_ascii=False), encoding="utf-8")
# print(f"\n✓ Zapisano {len(all_raw)} surowych odpowiedzi → {OUT_FILE}")
