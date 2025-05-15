# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:21:03 2025
@author: darek
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, textwrap, json, re

# ── 1. MODEL ───────────────────────────────────────────────────────────
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tok   = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)

# ── 2. INPUT ───────────────────────────────────────────────────────────
input_json = """[
  {"SECTION":"Timing as a Cue for Emotional Expression",
   "CITATION":"1 See Hevner (1935) and Hevner (1937). 2 On the effect of rhythmic variation interacted with pitch, see Schellenberg et al., (2000). 3 Balkwill & Thompson (1999).",
   "REFERENCE":"Hevner (1935)"},
  {"SECTION":"Timing as a Cue for Emotional Expression",
   "CITATION":"1 See Hevner (1935) and Hevner (1937). 2 On the effect of rhythmic variation interacted with pitch, see Schellenberg et al., (2000). 3 Balkwill & Thompson (1999).",
   "REFERENCE":"Hevner (1937)"},
  {"SECTION":"Timing as a Cue for Emotional Expression",
   "CITATION":"1 See Hevner (1935) and Hevner (1937). 2 On the effect of rhythmic variation interacted with pitch, see Schellenberg et al., (2000). 3 Balkwill & Thompson (1999).",
   "REFERENCE":"Schellenberg et al., (2000)"}
]"""

# ── FEW-SHOT PRZYKŁAD ────────────────────────────────────────────────
example_in = """[
  { "SECTION": "Intro",
    "CITATION": "Dogs communicate using a rich repertoire (Smith, 2020).",
    "REFERENCE": "Smith, 2020" },
  { "SECTION": "Intro",
    "CITATION": "Dogs communicate using a rich repertoire (Smith, 2020).",
    "REFERENCE": "Smith, 2020" }
]"""

example_out = """[
  {
    "section": "Intro",
    "text": "Dogs communicate using a rich repertoire (Smith, 2020).",
    "citations": ["Smith, 2020"]
  }
]"""

# ── PROMPT ────────────────────────────────────────────────────────────
system_msg = (
    "You are a transformer that returns ONLY valid JSON between <START_JSON> and <END_JSON>. "
    "Never add explanations or markdown."
)

user_msg = textwrap.dedent(f"""
### EXAMPLE_INPUT
{example_in}

### EXAMPLE_OUTPUT
{example_out}

---

### TASK
Same transformation for the dataset below.

Rules (repeat):
1. Group by SECTION.
2. Keep the first occurrence of each distinct CITATION.
3. Concatenate those into 'text'.
4. Extract unique references in first-appearance order → 'citations'.

### INPUT
{input_json}

<START_JSON>""")

prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}"

# ── GENERATE (greedy) ────────────────────────────────────────────────
inputs = tok(prompt, return_tensors="pt").to("cuda")
out    = model.generate(**inputs,
                        max_new_tokens=512,
                        do_sample=False)

decoded = tok.decode(out[0], skip_special_tokens=True)

# ── WYCIĘCIE + WALIDACJA ─────────────────────────────────────────────
m = re.search(r"<START_JSON>(.*?)<END_JSON>", decoded, re.S)
if not m:
    raise ValueError("Brak bloków <START_JSON>/<END_JSON>:\n" + decoded)

json_block = m.group(1).strip()
try:
    data = json.loads(json_block)
    print("✔ JSON OK\n", json.dumps(data, indent=2, ensure_ascii=False))
except json.JSONDecodeError:
    print("✘ Niepoprawny JSON:\n", json_block)

#%%
# -*- coding: utf-8 -*-
"""
Synthetic citation aggregation – few-shot prompt
Created on Thu May 15 13:30:00 2025  |  @author: darek
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap, json, re, torch

# ───────────────────────── 1. MODEL (chat) ──────────────────────────
model_id = "deepseek-ai/deepseek-llm-7b-chat"
tok   = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)

# ───────────────────────── 2. FEW-SHOT EXAMPLES ─────────────────────
example_A_in = """[
  { "SECTION": "Intro",
    "CITATION": "Dogs communicate using a rich repertoire (Smith, 2020).",
    "REFERENCE": "Smith, 2020" },
  { "SECTION": "Intro",
    "CITATION": "Dogs communicate using a rich repertoire (Smith, 2020).",
    "REFERENCE": "Smith, 2020" }
]"""

example_A_out = """[
  {
    "section": "Intro",
    "text": "Dogs communicate using a rich repertoire (Smith, 2020).",
    "citations": ["Smith, 2020"]
  }
]"""

example_B_in = """[
  { "SECTION": "Background",
    "CITATION": "Language models improve rapidly (Jones & Ruiz 2023).",
    "REFERENCE": "Jones & Ruiz 2023" },
  { "SECTION": "Background",
    "CITATION": "1 See Hevner (1935) and Hevner (1937).",
    "REFERENCE": "Hevner, 1935" }
]"""

example_B_out = """[
  {
    "section": "Background",
    "text": "Language models improve rapidly (Jones & Ruiz 2023). 1 See Hevner (1935) and Hevner (1937).",
    "citations": ["Jones & Ruiz 2023", "Hevner (1935)", "Hevner (1937)"]
  }
]"""

# ───────────────────────── 3. TRUDNIEJSZE DANE TESTOWE ───────────────
input_json = """[
  {
    "SECTION": "Introduction",
    "CITATION": "Music can convey emotion (Gardner, 1993, p. 124).",
    "REFERENCE": "Gardner, 1993, p. 124"
  },
  {
    "SECTION": "Introduction",
    "CITATION": "Our attraction to music has long fascinated researchers (Juslin & Laukka 2004).",
    "REFERENCE": "Juslin & Laukka 2004"
  },
  {
    "SECTION": "Introduction",
    "CITATION": "Music can convey emotion (Gardner, 1993, p. 124).",
    "REFERENCE": "Gardner, 1993, p. 124"
  },
  {
    "SECTION": "Methods",
    "CITATION": "1 See Hevner (1935) and Hevner (1937).",
    "REFERENCE": "Hevner, 1935"
  },
  {
    "SECTION": "Methods",
    "CITATION": "1 See Hevner (1935) and Hevner (1937). 2 Schellenberg et al. (2000).",
    "REFERENCE": "Hevner, 1937"
  },
  {
    "SECTION": "Methods",
    "CITATION": "2 Schellenberg et al. (2000).",
    "REFERENCE": "Schellenberg et al., 2000"
  }
]"""

# ───────────────────────── 4. PROMPT  ────────────────────────────────
system_msg = (
    "You are a strict data transformer. "
    "Return ONLY valid JSON between <START_JSON> and <END_JSON>. "
    "Never output explanations, markdown or thoughts."
)

user_msg = textwrap.dedent(f"""
### EXAMPLE_INPUT
{example_A_in}

### EXAMPLE_OUTPUT
{example_A_out}

### EXAMPLE_INPUT
{example_B_in}

### EXAMPLE_OUTPUT
{example_B_out}

---

### TASK
Transform the dataset below using the same rules:
1. Group by SECTION.
2. Keep the first occurrence of each distinct CITATION.
3. Concatenate those citations into 'text'.
4. Extract unique references in first-appearance order → 'citations'.

### INPUT
{input_json}

<START_JSON>""")

prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}"

# ───────────────────────── 5. GENERATE (greedy) ──────────────────────
inputs  = tok(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
decoded = tok.decode(outputs[0], skip_special_tokens=True)
print(decoded)
# ───────────────────────── 6. CUT + VALIDATE ─────────────────────────
m = re.search(r"<START_JSON>\s*(\[.*?\])\s*<END_JSON>", decoded, re.S)
if not m:
    raise ValueError("Brak poprawnych tagów:\n" + decoded)

json_block = m.group(1)
try:
    data = json.loads(json_block)
    print("✔ JSON parsed OK\n", json.dumps(data, indent=2, ensure_ascii=False))
except json.JSONDecodeError as e:
    print("✘ Niepoprawny JSON:", e, "\n--- BLOCK ---\n", json_block)
