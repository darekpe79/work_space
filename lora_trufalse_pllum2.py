# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:28:02 2025

@author: darek
"""

"""
Klasyfikacja PBL (True/False) z użyciem PLLuM (CYFRAGOVPL/Llama-PLLuM-8B-instruct)
QLoRA 4-bit + dynamiczny budżet tokenów oparty na ID (bez ponownej tokenizacji).

W TYM SKRYPCIE:
- sanity-check: sprawdzamy, czy 1. niemaskowana etykieta to faktycznie 1. sub-token "True"/"False" (25 próbek),
- quick probe PRZED treningiem: logity + PEŁNA odpowiedź modelu (generate) dla 25 próbek,
- callback: w trakcie treningu drukujemy kilka predykcji + pełne odpowiedzi,
- eval na 100 próbkach (żeby szybko obejrzeć),
- quick probe PO treningu: ponownie drukujemy pełne odpowiedzi dla 25 próbek.

Kluczowe ustawienia:
- AFTER_INPUT = "\\n### Odpowiedź:\\n" (bez spacji, newline na końcu)
- Porównujemy 1. sub-tokeny "True"/"False" (bez spacji z przodu).
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# ========================= 1) ŁADOWANIE I SCALANIE DANYCH =========================
def load_and_merge_data(json_file_path, excel_file_path, common_column='Link',
                        selected_columns_list=['Tytuł artykułu','Tekst artykułu','do PBL','hasła przedmiotowe']):
    # Ładujemy JSON i Excel, scalając po kolumnie 'Link'. Zachowujemy kolejność z Excela.
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)[['Link','Tekst artykułu']]
    df_json['Tekst artykułu'] = df_json['Tekst artykułu'].astype(str)

    df_excel = pd.read_excel(excel_file_path)
    df_excel['original_order'] = df_excel.index

    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")
    merged_df = merged_df.sort_values(by='original_order')
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)

    # Ograniczamy do ostatniego wiersza, gdzie do PBL=True i hasła przedmiotowe niepuste
    filtered_df = merged_df[(merged_df['do PBL'] == True) & (merged_df['hasła przedmiotowe'].notna())]
    if not filtered_df.empty:
        last_true_filled_index = filtered_df.index[-1]
        merged_df = merged_df.loc[:last_true_filled_index]
    else:
        return pd.DataFrame(columns=selected_columns_list)

    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df[merged_df['do PBL'].isin([True, False])]
    selected_columns = merged_df[selected_columns_list]
    return selected_columns

# Ścieżki — dostosuj do swojego środowiska
base_dir = 'D:/Nowa_praca/dane_model_jezykowy/kopia_dla_UAM/'
json_dir = os.path.join(base_dir, 'Json')

# Zbieramy pary JSON/XLSX o wspólnych nazwach
json_files = {os.path.splitext(f)[0]: os.path.join(json_dir, f)
              for f in os.listdir(json_dir) if f.endswith('.json')}
excel_files = {os.path.splitext(f)[0]: os.path.join(base_dir, f)
               for f in os.listdir(base_dir) if f.endswith('.xlsx')}
common_files = set(json_files.keys()).intersection(excel_files.keys())

merged_dfs = []
for file_name in sorted(common_files):
    json_path = json_files[file_name]
    excel_path = excel_files[file_name]
    print(f"Przetwarzanie pary: JSON - {json_path}, Excel - {excel_path}")
    merged_df = load_and_merge_data(json_path, excel_path)
    if merged_df is not None and not merged_df.empty:
        merged_dfs.append(merged_df)

if not merged_dfs:
    raise RuntimeError("Brak złączonych danych. Sprawdź ścieżki/formaty.")

combined_df = pd.concat(merged_dfs, ignore_index=True)

# ========================= 2) BALANS, LABELS, COMBINED TEXT =========================
df = combined_df.dropna(subset=['do PBL']).copy()
df['do PBL'] = df['do PBL'].astype(str).replace({'0.0': 'False', '1.0': 'True'})
value_counts = df['do PBL'].value_counts()
print("Liczba wystąpień przed undersamplingiem:\n", value_counts)

desired_ratio = 1.3
count_true = value_counts.get('True', 0)
count_false = value_counts.get('False', 0)
keep_false_count = min(int(count_true * desired_ratio), count_false)

df_false = df[df['do PBL'] == 'False'].sample(n=keep_false_count, random_state=42) if keep_false_count > 0 else df[df['do PBL'] == 'False']
df_true  = df[df['do PBL'] == 'True']

df_adjusted = pd.concat([df_false, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Liczba wystąpień po dostosowaniu:\n", df_adjusted['do PBL'].value_counts())

df_adjusted['combined_text'] = df_adjusted['Tytuł artykułu'].astype(str) + " " + df_adjusted['Tekst artykułu'].astype(str)

# ========================= 3) DANE INSTRUKCYJNE =========================
INSTRUCTION = (
    "Oceń, czy poniższy tekst kwalifikuje się do Polskiej Bibliografii Literackiej. "
    "Odpowiedz jednym słowem: True lub False."
)

df_adjusted['output'] = df_adjusted['do PBL'].map(lambda x: 'True' if str(x) == 'True' else 'False')
df_adjusted['instruction'] = INSTRUCTION
df_adjusted['input'] = df_adjusted['combined_text']

df_pllum = df_adjusted[['instruction', 'input', 'output']].copy()

# Przycinamy do ~połowy przykładów (stratyfikacja), żeby szybciej iterować
frac_keep = 0.5
df_pllum = (
    df_pllum
      .groupby('output', group_keys=False)
      .apply(lambda g: g.sample(
          n=min(len(g), max(1, int(round(len(g)*frac_keep)))),
          random_state=42
      ))
      .reset_index(drop=True)
)

print("Po przycięciu:", len(df_pllum))
print(df_pllum['output'].value_counts())

# ========================= 4) MODEL + QLoRA (4-bit) =========================
model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map='auto',
    trust_remote_code=True
)

# Przygotowanie pod k-bit trening + LoRA
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()  # sanity-check

# ========================= 5) DYNAMICZNY BUDŻET TOKENÓW (ID) =========================
IGNORE_INDEX   = -100
ANSWER_PREFIX  = "### Odpowiedź:"
AFTER_INPUT    = f"\n{ANSWER_PREFIX}\n"   # BEZ spacji; newline na końcu

HARD_CAP       = 768
MODEL_LIMIT    = int(getattr(model.config, "max_position_embeddings", 131072))
BIG_NUMBER     = 2_000_000_000

def get_ctx_cap(tok, mdl, hard_cap=HARD_CAP, safety_margin=4):
    # Obliczamy efektywny budżet kontekstu biorąc min(model_limit, tokenizer_limit, hard_cap) - safety_margin
    cap_model = getattr(mdl.config, "max_position_embeddings", None)
    if cap_model is None or cap_model > BIG_NUMBER:
        cap_model = BIG_NUMBER
    cap_tok = getattr(tok, "model_max_length", None)
    if cap_tok is None or cap_tok > BIG_NUMBER:
        cap_tok = BIG_NUMBER
    cap = min(int(cap_model), int(cap_tok), int(hard_cap))
    return max(16, cap - safety_margin)

CTX_CAP = get_ctx_cap(tokenizer, model)
print("Efektywny budżet kontekstu (tokeny):", CTX_CAP)

# --- Pierwszy sub-token dla 'True'/'False' (bez spacji) ---
TRUE_TOKENS  = tokenizer.encode("True",  add_special_tokens=False)
FALSE_TOKENS = tokenizer.encode("False", add_special_tokens=False)
TRUE_ID0, FALSE_ID0 = TRUE_TOKENS[0], FALSE_TOKENS[0]
print("DEBUG first-token IDs:", TRUE_ID0, FALSE_ID0,
      "| decode:", repr(tokenizer.decode([TRUE_ID0])), repr(tokenizer.decode([FALSE_ID0])))
print("AFTER_INPUT repr:", repr(AFTER_INPUT))

def tokenize_row_dynamic_budgeted(example):
    # Budujemy prompt: [BOS] "### Instrukcja: ..." + "### Wejście: " + [INPUT] + "\n### Odpowiedź:\n" + [ANS] + [EOS]
    instruction = example["instruction"]
    input_text  = example["input"]
    output_text = example["output"]  # "True" lub "False"

    before_input = f"### Instrukcja: {instruction}\n### Wejście: "
    after_input  = AFTER_INPUT

    # 1) Tokenizujemy stałe segmenty (bez special tokens)
    ids_before = tokenizer(before_input, add_special_tokens=False)["input_ids"]
    ids_after  = tokenizer(after_input,  add_special_tokens=False)["input_ids"]

    # 2) Budżet dla inputu – odejmujemy narzut promptu + docelowej odpowiedzi (True/False)
    special_count = int(tokenizer.bos_token_id is not None) + int(tokenizer.eos_token_id is not None)
    soft_cap = max(16, min(CTX_CAP, MODEL_LIMIT) - special_count)

    # Odpowiedź BEZ spacji z przodu
    ids_ans = tokenizer(output_text, add_special_tokens=False)["input_ids"]
    overhead = len(ids_before) + len(ids_after) + len(ids_ans)
    budget_for_input = max(1, soft_cap - overhead)

    # 3) Pre-cięcie po znakach dla przyspieszenia (heurystyka 4 znaki/token)
    approx_chars_per_token = 4
    max_chars = budget_for_input * approx_chars_per_token
    if len(input_text) > max_chars:
        input_text = input_text[:max_chars]

    # 4) Tokenizacja inputu z twardym limitem
    ids_input = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        max_length=budget_for_input
    )["input_ids"]

    # 5) Składamy pełną sekwencję ID
    ids_prompt = []
    if tokenizer.bos_token_id is not None:
        ids_prompt.append(tokenizer.bos_token_id)
    ids_prompt += (ids_before + ids_input + ids_after)

    ids_full = ids_prompt + ids_ans
    if tokenizer.eos_token_id is not None:
        ids_full.append(tokenizer.eos_token_id)

    # 6) Twardy limit długości
    hard_cap = min(CTX_CAP, MODEL_LIMIT) + special_count
    if len(ids_full) > hard_cap:
        ids_full = ids_full[:hard_cap]
        if tokenizer.eos_token_id is not None:
            ids_full[-1] = tokenizer.eos_token_id

    # 7) Maskowanie promptu w labelach – uczymy tylko odpowiedź
    labels = ids_full.copy()
    labels[:len(ids_prompt)] = [IGNORE_INDEX] * len(ids_prompt)

    return {
        "input_ids": ids_full,
        "attention_mask": [1] * len(ids_full),
        "labels": labels,
        # surowe pola — pomocne w podglądzie
        "raw_instruction": instruction,
        "raw_input": input_text,
        "raw_output": output_text,
    }

_dataset = Dataset.from_pandas(df_pllum, preserve_index=False)
tokenized_dataset = _dataset.map(tokenize_row_dynamic_budgeted)

# ========================= 5.1) SANITY CHECK ETYKIET (25 próbek) =========================
def _inspect_labels(ds, true_id0, false_id0, sample_n=25, verbose=True):
    # Sprawdzamy pierwszy NIEMASKOWANY token w labels – powinien być równy
    # pierwszemu sub-tokenowi "True" albo "False".
    total = len(ds)
    ok_true = ok_false = other = 0
    print("\n=== SANITY CHECK: pierwsze nie-maskowane ID w labels ===")
    printed = 0
    for i in range(min(total, 5000)):
        labels = ds[i]["labels"]
        first = next((j for j, t in enumerate(labels) if t != IGNORE_INDEX), None)
        if first is None:
            other += 1
            continue
        t = int(labels[first])
        if t == int(true_id0):
            ok_true += 1
        elif t == int(false_id0):
            ok_false += 1
        else:
            other += 1

        if verbose and printed < sample_n:
            dec = tokenizer.decode([t])
            gold = ds[i].get("raw_output", "?")
            print(f"[{i:04d}] first_label_id={t} decode={repr(dec)} gold={gold}")
            printed += 1

    print(f"[LABEL CHECK] total={total}  TrueID0={ok_true}  FalseID0={ok_false}  OTHER={other} (OTHER powinno być ~0)\n")

_inspect_labels(tokenized_dataset, TRUE_ID0, FALSE_ID0, sample_n=25, verbose=True)

# ========================= 6) METRYKI (2-klasowe, 1. token odpowiedzi) =========================
def preprocess_logits_for_metrics(logits, labels):
    # Bierzemy logity dokładnie na pozycji pierwszego NIEMASKOWANEGO znacznika w labels,
    # a potem redukujemy tylko do dwóch wymiarów: logity dla ID0("True") i ID0("False").
    if isinstance(logits, tuple):
        logits = logits[0]
    with torch.no_grad():
        if labels is None:
            sel = logits[:, -1, :]
        else:
            pos_list = []
            for i in range(labels.shape[0]):
                pos = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=False)
                pos_list.append(pos[0].item() if len(pos) > 0 else logits.shape[1]-1)
            idx = torch.tensor(pos_list, device=logits.device)
            sel = logits[torch.arange(logits.size(0), device=logits.device), idx, :]
        return sel[:, [TRUE_ID0, FALSE_ID0]]

def compute_metrics_2class(eval_pred: Tuple[np.ndarray, np.ndarray]):
    # Z metryki wyrzucamy próbki, w których pierwszy niemaskowany token nie jest True_ID0/False_ID0 (nie powinno się zdarzyć).
    two_logits, labels = eval_pred
    two_logits = np.asarray(two_logits)
    labels = np.asarray(labels)

    preds_list = []
    golds_list = []
    for i in range(labels.shape[0]):
        pos = np.where(labels[i] != IGNORE_INDEX)[0]
        if len(pos) == 0:
            continue
        true_id = int(labels[i, pos[0]])
        if   true_id == int(TRUE_ID0):  golds_list.append(0)
        elif true_id == int(FALSE_ID0): golds_list.append(1)
        else:
            continue
        preds_list.append(int(two_logits[i].argmax(-1)))

    if not golds_list:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "skipped": int(labels.shape[0])}

    acc = accuracy_score(golds_list, preds_list)
    precision, recall, f1, _ = precision_recall_fscore_support(
        golds_list, preds_list, average="weighted", zero_division=0
    )
    skipped = int(labels.shape[0] - len(golds_list))
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "skipped": skipped}

# ========================= 7) COLLATOR z dynamicznym paddingiem =========================
@dataclass
class DataCollatorForCausalWithIgnore:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Padujemy input_ids i attention_mask do wspólnej długości; labels padujemy IGNORE_INDEX.
        batch_input = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }
        batch = self.tokenizer.pad(batch_input, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            lab = f["labels"]
            if len(lab) < max_len:
                lab = lab + [self.label_pad_token_id] * (max_len - len(lab))
            labels.append(lab)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

data_collator = DataCollatorForCausalWithIgnore(tokenizer=tokenizer)

# ========================= 8) SPLIT + LEKKI EVAL (100 próbek) =========================
split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset_full = split["test"]

# Losujemy i ograniczamy eval do 100 przykładów – szybka pętla i czytelne logi
rng = random.Random(42)
eval_indices = list(range(len(eval_dataset_full)))
rng.shuffle(eval_indices)
eval_indices = eval_indices[:min(len(eval_indices), 100)]
eval_dataset = eval_dataset_full.select(eval_indices)

# ========================= 8.1) FUNKCJE DRUKUJĄCE PEŁNE ODPOWIEDZI =========================
def _build_prompt_ids(instr: str, input_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Budujemy input_ids i attention_mask dla modelu identycznie jak w treningu
    before_input = f"### Instrukcja: {instr}\n### Wejście: "
    after_input  = AFTER_INPUT
    ids_before = tokenizer(before_input, add_special_tokens=False)["input_ids"]
    ids_after  = tokenizer(after_input,  add_special_tokens=False)["input_ids"]

    special_count = int(tokenizer.bos_token_id is not None) + int(tokenizer.eos_token_id is not None)
    overhead = len(ids_before) + len(ids_after) + 1  # miejsce na 1. token odpowiedzi
    budget_for_input = max(1, min(CTX_CAP, MODEL_LIMIT) - special_count - overhead)

    ids_input = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    if len(ids_input) > budget_for_input:
        ids_input = ids_input[:budget_for_input]

    ids_prompt = []
    if tokenizer.bos_token_id is not None:
        ids_prompt.append(tokenizer.bos_token_id)
    ids_prompt += (ids_before + ids_input + ids_after)

    input_ids = torch.tensor([ids_prompt], dtype=torch.long).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

def _classify_next_token(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[str, float]:
    # Klasyfikacja na podstawie logitów 1. wygenerowanego tokenu (True_ID0 vs False_ID0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = out.logits[:, -1, :]
        logp = torch.log_softmax(next_token_logits, dim=-1)
    logp_true  = float(logp[0, TRUE_ID0].item())
    logp_false = float(logp[0, FALSE_ID0].item())
    pred = "True" if logp_true > logp_false else "False"
    margin = logp_true - logp_false
    return pred, margin

def _generate_full_answer(input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int = 5) -> str:
    # Tu drukujemy REALNĄ odpowiedź modelu (to, co "mówi") – pełen tekst po separatorze
    with torch.inference_mode():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_suffix = gen[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(gen_suffix, skip_special_tokens=True).strip()
    return generated_text

def _quick_eval_probe(model, tokenizer, ds, k=25):
    # === PRZED/PO TRENINGU DRUK: pokazujemy gold, pred (logitowy), margin oraz generated (pełna odpowiedź)
    print("\n=== QUICK EVAL PROBE (pierwszych", k, "próbek) ===")
    model.eval()
    for i in range(min(k, len(ds))):
        item = ds[i]
        instruction = item.get("raw_instruction", INSTRUCTION)
        input_text  = item.get("raw_input", "")
        gold        = item.get("raw_output", "")

        input_ids, attention_mask = _build_prompt_ids(instruction, input_text)
        pred, margin = _classify_next_token(input_ids, attention_mask)
        generated_text = _generate_full_answer(input_ids, attention_mask, max_new_tokens=5)

        # Dodatkowo: pierwszy NIEMASKOWANY label – kontrola zgodności etykiety
        labels = item["labels"]
        pos = next((j for j, t in enumerate(labels) if t != IGNORE_INDEX), None)
        first_lab = None if pos is None else int(labels[pos])
        first_lab_dec = None if first_lab is None else tokenizer.decode([first_lab])

        print(f"[{i:02d}] gold={gold:5s} pred={pred:5s}  margin={margin:+.3f}  first_label_id={first_lab} decode={repr(first_lab_dec)}")
        print(f"     generated={repr(generated_text)}")
    print("=== KONIEC QUICK EVAL PROBE ===\n")

class EvalPrinterCallback(TrainerCallback):
    """
    Drukuje kilka przykładowych predykcji podczas każdej ewaluacji:
    - gold
    - pred (logitowy)
    - margin
    - generated (pełna odpowiedź modelu)
    """
    def __init__(self, eval_ds, tokenizer, k=5, seed=123):
        self.eval_ds = eval_ds
        rnd = random.Random(seed)
        pool = list(range(len(eval_ds)))
        self.sample_idx = sorted(rnd.sample(pool, min(k, len(pool))))
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()
        print("\n=== Podgląd predykcji (eval) @ step", state.global_step, "=== ")
        for idx in self.sample_idx:
            item = self.eval_ds[idx]
            instruction = item.get("raw_instruction", INSTRUCTION)
            input_text  = item.get("raw_input", "")
            gold        = item.get("raw_output", "")

            input_ids, attention_mask = _build_prompt_ids(instruction, input_text)
            pred, margin = _classify_next_token(input_ids, attention_mask)
            generated_text = _generate_full_answer(input_ids, attention_mask, max_new_tokens=5)

            print(f"[{idx}] gold={gold:5s} pred={pred:5s}  margin={margin:+.3f}")
            print(f"     generated={repr(generated_text)}  input_snip={repr(input_text[:120])}")
        print("=== koniec podglądu ===\n")

# ========================= 9) TRENING =========================
training_args = TrainingArguments(
    output_dir="C:/treningpllum/checkpoints",
    seed=42,

    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,          # efektywny batch ~8
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    optim="adamw_torch",
    max_grad_norm=1.0,
    fp16=True,
    gradient_checkpointing=True,

    # logi i ewaluacja/zapis
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=50,               # co 50 kroków
    save_strategy="steps",
    save_steps=50,               # spójnie z eval
    save_total_limit=2,

    # lekka ewaluacja
    per_device_eval_batch_size=1,
    fp16_full_eval=True,
    eval_accumulation_steps=1,

    # wybór najlepszego modelu
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,   # UWAGA: lekki eval = 100 próbek
    tokenizer=tokenizer,         # (ostrzeżenie o deprec. można zignorować)
    data_collator=data_collator,
    compute_metrics=compute_metrics_2class,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=5e-4
        ),
        EvalPrinterCallback(eval_dataset, tokenizer, k=5, seed=123),
    ],
)

# === DEBUG: szybki probe na 25 próbkach PRZED treningiem — DRUKUJEMY PEŁNE ODPOWIEDZI ===
_quick_eval_probe(model, tokenizer, eval_dataset, k=25)

print("Rozpoczęcie treningu PLLuM (True/False)...")
trainer.train()

# Ewaluacja końcowa (na najlepszym checkpointcie dzięki load_best_model_at_end=True)
results = trainer.evaluate()
print("Wyniki ewaluacji PLLuM:", results)
print("Najlepszy checkpoint:", trainer.state.best_model_checkpoint)

# ========================= 10) ZAPIS =========================
save_dir = "C:/treningpllum/"
os.makedirs(save_dir, exist_ok=True)
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Trening zakończony i model zapisany w:", save_dir)

# ========================= 11) QUICK PROBE PO TRENINGU (25 próbek) =========================
# Ponowny szybki przegląd – teraz na załadowanym najlepszym modelu z Trainer
print("\n=== QUICK EVAL PROBE PO TRENINGU (25 próbek) ===")
_quick_eval_probe(trainer.model, tokenizer, eval_dataset, k=25)

# ========================= 12) INFERENCJA (pojedynczy tekst) =========================
def classify_with_pllum(text: str) -> str:
    """
    Klasyfikacja pojedynczego tekstu: porównujemy logproby 1. sub-tokenu „True” vs „False”.
    Generujemy też pełną odpowiedź (jeśli chcesz, odkomentuj debug).
    """
    model.eval()
    instruction = INSTRUCTION
    input_ids, attention_mask = _build_prompt_ids(instruction, text)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = out.logits[:, -1, :]
        logp = torch.log_softmax(next_token_logits, dim=-1)

    # # Debug pełnej odpowiedzi:
    # generated_text = _generate_full_answer(input_ids, attention_mask, max_new_tokens=5)
    # print("Model mówi:", repr(generated_text))

    return "True" if logp[0, TRUE_ID0] > logp[0, FALSE_ID0] else "False"
