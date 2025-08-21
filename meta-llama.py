import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
# -------------------- KONFIGURACJA ---------------------
MODEL_ID   = "meta-llama/Llama-3.1-8B-Instruct" #"tiiuae/Falcon3-10B-Instruct"
USE_INT8   = True
TEMPERATURE = 0.0
MAX_TOK    = 1300
#OVERLAP    = 150
from huggingface_hub import login
token = os.getenv("HF_TOKEN")
# -------------------- TOKENIZER ------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# -------------------- MODEL ----------------------------
if USE_INT8:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

# -------------------- USTAWIENIA GENERACJI ----------------------------
model.generation_config.update(
    temperature=TEMPERATURE,
    do_sample=False,
    pad_token_id=tok.eos_token_id
)

# -------------------- PROMPT ----------------------------
prompt = '''You are a highly skilled assistant for processing book indexes and bibliographic data. Your task is to transform irregular index text into a strictly structured, semicolon-delimited CSV format.

Requirements:
- Use “;” as the field delimiter.
- Do not quote fields—even if they contain commas.
- Output columns in this order: Name;Description;Pages
- Each Pages field may contain one or multiple page numbers or ranges, separated by commas.

Here is an example of the desired output format:

Name;Description;Pages
Abbreviationes;Quid significant in hoc volumine;53
Acapulco;Portus;610
Acevedo, Ignatius de, S.I.;Borgia dat ei instructionem missionariam;120, 122, 124
Acosta, Iosephus de, S.I.;Missionarius per regionem de La Plata;629, 709
Agnus Dei;Ab eo recipiuntur;140
Aguado, Petrus de, O.F.M.;Scriptor;130

Now transform the following raw index text into that CSV form, one entry per line:

Abbreviationes in hoc volumine quid significent, 53.
Acapulco, portus, 610.
Acevedo, Ignatius de, S. I., Borgia dat ei instructionem missionariam, 120, 122, 124.
Acllas, 21s.
Acosta, Iosephus de, S.I., vita, 299; manifestat Borgiae suum desiderium Indias adeundi, 300–302; pro cleri educatione, 302; P. Ludovicum Guzmán ad missiones praesentat, 303; revelat P. Nadal voluntatem suam ad missiones indicas et varia de se ipso, 300, 301; conficit litteras annuas, Roma vel Burgos destinatus, 322; Peruae missionarius renuntiatus, 389, 390, 391, 371; nuntiat Borgiae praeludia itineris, 439–442; quaerit de P. Fonseca, 442; proponit ad Sacros Ordines F. D. Martínez, 442; Hispali versatur et Sanlúcar, 440; in insula S. Ioannis et S. Dominici, 443; eius iter a Sacchini narratur, 47s; Limam attigit, 36, 505; confessarius in collegio limensi et magister novitiorum, 505; concionator Limae, 703; missionarius per regionem de La Plata, 629, 709; eius missio per Peruam, 629, 706; ad Potosí, 709; quaestiones morales cum eo conferendae, 632; provincialis designatus Peruae, 37; eius actio in Congregationibus provincialibus, 38; scriptor, 321; informationes de eo, 301; laudatur, 505, 507, 509, 589.

'''

# -------------------- TOKENIZACJA I GENERACJA ----------------------------
inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOK).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=MAX_TOK  # wykorzystanie stałej
)

# -------------------- WYNIK ----------------------------
response = tok.decode(outputs[0], skip_special_tokens=True)
print("\nWygenerowana odpowiedź:\n")
print(response)

#%%

import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------- KONFIGURACJA ---------------------
MODEL_ID    = "tiiuae/Falcon3-10B-Instruct"
USE_INT8    = True
TEMPERATURE = 0.0
MAX_TOK     = 1300
PREVIEW_CHUNKS = 3

# -------------------- WCZYTAJ MODEL I TOKENIZER ---------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if USE_INT8:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
model.generation_config.update(
    temperature=TEMPERATURE,
    do_sample=False,
    pad_token_id=tok.eos_token_id
)

# -------------------- PROMPT (stała część) ---------------------
BASE_PROMPT = """You are a highly skilled assistant for processing book indexes and bibliographic data. Your task is to transform irregular index text into a strictly structured, semicolon-delimited CSV format.

Requirements:
- Use “;” as the field delimiter.
- Do not quote fields—even if they contain commas.
- Output columns in this order: Name;Description;Pages
- Each Pages field may contain one or multiple page numbers or ranges, separated by commas.

Here is an example of the desired output format:

Name;Description;Pages
Abbreviationes;Quid significant in hoc volumine;53
Acapulco;Portus;610
Acevedo, Ignatius de, S.I.;Borgia dat ei instructionem missionariam;120, 122, 124
Acosta, Iosephus de, S.I.;Missionarius per regionem de La Plata;629, 709
Agnus Dei;Ab eo recipiuntur;140
Aguado, Petrus de, O.F.M.;Scriptor;130

Now transform the following raw index text into that CSV form, one entry per line:
"""

# -------------------- FUNKCJE ---------------------
def parse_csv_response(text):
    entries = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("Name;"):
            continue
        parts = line.strip().split(";")
        if len(parts) >= 3:
            entries.append({
                "name": parts[0].strip(),
                "description": parts[1].strip(),
                "pages": parts[2].strip()
            })
    return entries

def chunk_text(entries, tokenizer, max_token_len):
    chunks = []
    current_chunk = ""
    for entry in entries:
        token_len = len(tokenizer(entry, return_tensors="pt").input_ids[0])
        current_len = len(tokenizer(current_chunk, return_tensors="pt").input_ids[0]) if current_chunk else 0
        if current_len + token_len > max_token_len:
            chunks.append(current_chunk.strip())
            current_chunk = entry
        else:
            current_chunk += " " + entry
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

# -------------------- WCZYTAJ I PRZYGOTUJ DANE ---------------------
with open("C:/pdf_llm_do_roboty/indkes.txt", encoding="utf-8") as f:
    raw_text = f.read()

# Uprość i wyczyść
cleaned_text = re.sub(r"\s+", " ", raw_text)
entries = re.split(r"(?=(?:[A-ZĄĆĘŁŃÓŚŹŻ][^,]{1,80},))", cleaned_text)
entries = [e.strip().strip(".") for e in entries if len(e.strip()) > 10]

# Podziel na chunk'i
chunks = chunk_text(entries, tok, MAX_TOK - 400)
print(f"\n🔹 Znaleziono {len(entries)} jednostek, podzielono na {len(chunks)} chunków\n")

# Pokaż kilka chunków do kontroli
for i in range(min(PREVIEW_CHUNKS, len(chunks))):
    print(f"🧩 CHUNK {i+1}:\n{'-'*40}\n{chunks[i][:700]}...\n{'-'*40}\n")

# -------------------- GENERUJEMY ---------------------
all_data = []
for idx, chunk in enumerate(chunks):
    prompt = BASE_PROMPT + chunk
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOK).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=MAX_TOK)
    decoded = tok.decode(outputs[0], skip_special_tokens=True)

    # Parsuj i dodaj
    parsed_entries = parse_csv_response(decoded)
    all_data.extend(parsed_entries)
    print(f"✅ Wygenerowano chunk {idx+1}/{len(chunks)} ({len(parsed_entries)} rekordów)")

# -------------------- ZAPIS DO JSON ---------------------
with open("index_parsed.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\n🎉 Gotowe! Zapisano {len(all_data)} rekordów do pliku index_parsed.json.")
