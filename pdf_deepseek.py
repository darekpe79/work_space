

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, json, re
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

# -------------------  USTAWIENIA  -------------------
ROOT_DIR   = Path(r"D:/Nowa_praca/pdfy_oprogramowanie")
OUT_FILE   = ROOT_DIR / "software_in_pdfsdeepseek.json"

MODEL_ID   = model_name = "deepseek-ai/deepseek-llm-7b-chat" #"tiiuae/Falcon3-10B-Instruct"#"CYFRAGOVPL/Llama-PLLuM-8B-instruct" "deepseek-ai/deepseek-llm-7b-chat"   # zmień na 33b-chat gdy masz VRAM
USE_INT8   = False                              # True = 8-bit, False = fp16
TEMPERATURE = 0.25                                # 0.35 gdy model gubi nazwy
MAX_TOK    = 1000
OVERLAP    = 200
# ----------------------------------------------------

# ---- 1  MODEL -------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)

if USE_INT8:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto", trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )

model.generation_config.update(
    temperature=TEMPERATURE, top_p=0.8, do_sample=True,
    pad_token_id=tok.eos_token_id
)

# ---- 2  PDF ➜ TEXT --------------------------------
def pdf_to_text(pdf: Path) -> str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages)

# ---- 3  CHUNK -------------------------------------
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

# ---- 4  PROMPT ------------------------------------
SYSTEM = """
You must return **only software names** (stand-alone programs, libraries, APIs, NLP/GIS/OCR tools).

✘ IGNORE and NEVER list:
 • person surnames (e.g., Herrmann, Gius, Morariu, Jacobs, Klinger, Kowalski)
 • person-with-year citations!!! Don't take this!!! — (EXAMPLES- herrmann2019, bode2017, Herrmann_et_al_2022, Heuser_et_al_2016)
 • datasets / text collections (e.g., ELTeC, NKJP, Dariah-PL)
 • data / markup formats (e.g., XML, TEI, CSV)
 • generic nouns or abstract concepts (e.g., city, emotions, narrative, status, love)
 • publisher, project or conference acronyms (e.g., CLARIN, LREC, ACL)

Return exactly ONE JSON list of unique names, with no comments.  
If nothing is found → return [].
"""


GOOD_EX_IN  = "We used spaCy 3, QGIS 3.28 and the PolDeepNer2 model."
GOOD_EX_OUT = '["PolDeepNer2","QGIS","spaCy"]'

BAD1_IN = "City and publication_year are metadata variables."
BAD2_IN = "Herrmann 2019 processes the ELTeC corpus stored in XML."

PROMPT_TMPL = f"""
### system
{SYSTEM}

### good example
text:
{GOOD_EX_IN}
software:
{GOOD_EX_OUT}

### bad example 1
text:
{BAD1_IN}
software:
[]

### bad example 2
text:
{BAD2_IN}
software:
[]

### task
text:
{{chunk}}
software:
"""
print(PROMPT_TMPL)
# -------- 4  PARSER -----
STOP = {"city","village","author_gender","publication_year","emotions",
        "crowdsourcing","sentiment_analysis","symbolic_models","text","software"}
WHITELIST = {"geonames","polona","wikidata","wikipedia"}
RE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.\\-]{1,38}[A-Za-z0-9]$')

def is_software(s: str) -> bool:
    l = s.lower()
    if l in WHITELIST:                       return True
    if l in STOP:                            return False
    if re.fullmatch(r'\d{3,4}', l):          return False  # rok
    if re.fullmatch(r'[a-z]+[0-9]{4}', l):   return False  # nazwisko+rok
    return bool(RE_NAME.fullmatch(s.strip()))

def parse(reply: str) -> set[str]:
    m = re.search(r'\[[^\[\]]+\]', reply)
    if not m: return set()
    try:
        lst = json.loads(m.group(0))
    except Exception:
        return set()
    return {x.strip() for x in lst if isinstance(x, str) and is_software(x.strip())}

# -------- 5  LLM CALL ---
def extract_names(chunk_txt: str) -> set[str]:
    prompt = PROMPT_TMPL.format(chunk=chunk_txt)
    ids = tok.apply_chat_template([{"role":"user","content":prompt}],
                                  add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    gen = model.generate(ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=120)
    return parse(tok.decode(gen[0][ids.shape[-1]:], skip_special_tokens=True))

# -------- 6  PIPELINE ----
def process_pdf(pdf: Path) -> dict:
    names = set()
    for ch in chunk(pdf_to_text(pdf)):
        names |= extract_names(ch)
    return {
        "text_id": pdf.name,
        "software_detected": bool(names),
        "list_of_software": sorted(names)
    }

def pdf_files(root: Path):
    return [root] if root.is_file() else sorted(root.rglob("*.pdf"))

results = [process_pdf(p) for p in tqdm(pdf_files(ROOT_DIR), desc="PDFs")]

OUT_FILE.parent.mkdir(exist_ok=True)
OUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n✓ Zapisano {len(results)} rekordów → {OUT_FILE}")


#%%

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import torch, json, re
# from pathlib import Path
# from pypdf import PdfReader
# from tqdm import tqdm

# # ---------------  KONFIG  -----------------
# ROOT_DIR = Path(r"D:/Nowa_praca/pdfy_oprogramowanie")
# OUT_FILE = ROOT_DIR / "software_raw_chunks.json"

# MODEL_ID  = "tiiuae/Falcon3-10B-Instruct"  # lub "CYFRAGOVPL/Llama-PLLuM-8B-instruct"
# USE_INT8  = True                           # 8-bit (mniej VRAM) czy fp16
# TEMPERATURE = 0.25
# MAX_TOK   = 1000
# OVERLAP   = 200
# # ------------------------------------------

# # ---- 1  MODEL ----
# tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# if USE_INT8:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         quantization_config=BitsAndBytesConfig(load_in_8bit=True),
#         device_map="auto", trust_remote_code=True)
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.float16,
#         device_map="auto", trust_remote_code=True)

# model.generation_config.update(
#     temperature=TEMPERATURE, top_p=0.8, do_sample=True,
#     pad_token_id=tok.eos_token_id)

# # ---- 2  UTILS ----
# def pdf_to_text(path: Path) -> str:
#     return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)

# def chunk(txt, max_tok=MAX_TOK, overlap=OVERLAP):
#     buf, t = [], 0
#     for w in txt.split():
#         t += len(tok(w, add_special_tokens=False).input_ids)
#         buf.append(w)
#         if t >= max_tok:
#             yield " ".join(buf)
#             buf, t = buf[-overlap:], sum(len(tok(x, add_special_tokens=False).input_ids) for x in buf)
#     if buf: yield " ".join(buf)

# # ---- 3  PROMPT ----
# SYSTEM = """
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
