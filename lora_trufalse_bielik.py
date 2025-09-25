# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:28:02 2025

@author: darek
"""

"""
Klasyfikacja PBL (True/False) z użyciem PLLuM (CYFRAGOVPL/Llama-PLLuM-8B-instruct)
QLoRA 4-bit + dynamiczny budżet tokenów oparty na ID (bez ponownej tokenizacji).
Drukujemy przykładowe predykcje w trakcie ewaluacji (co 50 kroków).
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List

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

from huggingface_hub import login
token = os.getenv("HF_TOKEN")

# ========================= 1) ŁADOWANIE I SCALANIE DANYCH =========================

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link',
                        selected_columns_list=['Tytuł artykułu','Tekst artykułu','do PBL','hasła przedmiotowe']):
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

    # ograniczenie do ostatniego True z niepustymi hasłami
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

# Ścieżki — dostosuj
base_dir = 'D:/Nowa_praca/dane_model_jezykowy/kopia_dla_UAM/'
json_dir = os.path.join(base_dir, 'Json')

# Zbieramy pliki
json_files = {os.path.splitext(f)[0]: os.path.join(json_dir, f)
              for f in os.listdir(json_dir) if f.endswith('.json')}
excel_files = {os.path.splitext(f)[0]: os.path.join(base_dir, f)
               for f in os.listdir(base_dir) if f.endswith('.xlsx')}
common_files = set(json_files.keys()).intersection(excel_files.keys())

merged_dfs = []
for file_name in common_files:
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

# ========================= 3) DANE INSTRUKCYJNE DLA PLLuM =========================

INSTRUCTION = "Oceń, czy poniższy tekst kwalifikuje się do Polskiej Bibliografii Literackiej (odpowiedz dokładnie: True lub False)."

df_adjusted['output'] = df_adjusted['do PBL'].map(lambda x: 'True' if str(x) == 'True' else 'False')
df_adjusted['instruction'] = INSTRUCTION
df_adjusted['input'] = df_adjusted['combined_text']

df_pllum = df_adjusted[['instruction', 'input', 'output']].copy()
frac_keep = 0.5  # zachowaj ~połowę, stratyfikacja po klasie

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

model_name = "speakleash/Bielik-4.5B-v3.0-Instruct" # "CYFRAGOVPL/Llama-PLLuM-8B-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map = {"": 0} ,
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
AFTER_INPUT    = f"\n{ANSWER_PREFIX} "   # UWAGA: jedna spacja na końcu (ważne dla '▁True'/'▁False')
HARD_CAP       = 768
MODEL_LIMIT    = int(getattr(model.config, "max_position_embeddings", 131072))
BIG_NUMBER     = 2_000_000_000

def get_ctx_cap(tok, mdl, hard_cap=HARD_CAP, safety_margin=4):
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

# --- Kluczowe: klasy z LEADING SPACE (żeby 1. token odpowiedzi był '▁True'/'▁False') ---
TRUE_TOKENS  = tokenizer.encode(" True",  add_special_tokens=False)
FALSE_TOKENS = tokenizer.encode(" False", add_special_tokens=False)
TRUE_ID0, FALSE_ID0 = TRUE_TOKENS[0], FALSE_TOKENS[0]
print("DEBUG first-token decodes:",
      repr(tokenizer.decode([TRUE_ID0])),
      repr(tokenizer.decode([FALSE_ID0])))

def tokenize_row_dynamic_budgeted(example):
    instruction = example["instruction"]
    input_text  = example["input"]
    output_text = example["output"]  # "True" lub "False"

    before_input = f"### Instrukcja: {instruction}\n### Wejście: "
    after_input  = AFTER_INPUT

    # 1) Tokenizujemy stałe segmenty (bez special tokens)
    ids_before = tokenizer(before_input, add_special_tokens=False)["input_ids"]
    ids_after  = tokenizer(after_input,  add_special_tokens=False)["input_ids"]
    # UWAGA: odpowiedź z LEADING SPACE
    ids_ans    = tokenizer(" " + output_text,  add_special_tokens=False)["input_ids"]

    # 2) Liczymy budżet dla inputu
    special_count = int(tokenizer.bos_token_id is not None) + int(tokenizer.eos_token_id is not None)
    soft_cap = max(16, min(CTX_CAP, MODEL_LIMIT) - special_count)
    overhead = len(ids_before) + len(ids_after) + len(ids_ans)
    budget_for_input = max(1, soft_cap - overhead)

    # 3) Szybkie pre-cięcie po znakach
    approx_chars_per_token = 4
    max_chars = budget_for_input * approx_chars_per_token
    if len(input_text) > max_chars:
        input_text = input_text[:max_chars]

    # 4) Tokenizujemy input z twardym limitem
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

    # 6) Ostateczny bezpiecznik długości
    hard_cap = min(CTX_CAP, MODEL_LIMIT) + special_count
    if len(ids_full) > hard_cap:
        ids_full = ids_full[:hard_cap]
        if tokenizer.eos_token_id is not None:
            ids_full[-1] = tokenizer.eos_token_id

    # 7) Maskujemy prompt w labelach
    labels = ids_full.copy()
    labels[:len(ids_prompt)] = [IGNORE_INDEX] * len(ids_prompt)

    return {
        "input_ids": ids_full,
        "attention_mask": [1] * len(ids_full),
        "labels": labels,
        # surowe pola — przydadzą się do drukowania przykładów w ewaluacji
        "raw_instruction": instruction,
        "raw_input": input_text,
        "raw_output": output_text,
    }

_dataset = Dataset.from_pandas(df_pllum, preserve_index=False)
tokenized_dataset = _dataset.map(tokenize_row_dynamic_budgeted)

# ========================= 6) METRYKI (2-klasowe, 1. token odpowiedzi) =========================

def preprocess_logits_for_metrics(logits, labels):
    # logits: (B, L, V) lub tuple(...); labels: (B, L)
    if isinstance(logits, tuple):
        logits = logits[0]
    with torch.no_grad():
        if labels is None:
            sel = logits[:, -1, :]  # awaryjnie ostatnia pozycja
        else:
            pos_list = []
            for i in range(labels.shape[0]):
                pos = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=False)
                pos_list.append(pos[0].item() if len(pos) > 0 else logits.shape[1]-1)
            idx = torch.tensor(pos_list, device=logits.device)
            sel = logits[torch.arange(logits.size(0), device=logits.device), idx, :]
        # zwracamy tylko logity dla 1. tokenów ' True' i ' False'
        return sel[:, [TRUE_ID0, FALSE_ID0]]

def compute_metrics_2class(eval_pred):
    two_logits, labels = eval_pred   # two_logits: (N, 2), labels: (N, L)
    two_logits = np.asarray(two_logits)
    labels = np.asarray(labels)

    preds = two_logits.argmax(axis=-1)  # 0 -> " True", 1 -> " False"

    true_labels = []
    for i in range(labels.shape[0]):
        pos = np.where(labels[i] != IGNORE_INDEX)[0]
        if len(pos) == 0:
            continue
        true_id = int(labels[i, pos[0]])
        true_labels.append(0 if true_id == int(TRUE_ID0) else 1)

    if not true_labels:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ========================= 7) COLLATOR z dynamicznym paddingiem =========================

@dataclass
class DataCollatorForCausalWithIgnore:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
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

# ========================= 8) SPLIT + LEKKI EVAL + CALLBACK DRUKUJĄCY PRZYKŁADY =========================

split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset_full = split["test"]
# Losowy lekki eval (do ~1000 przykładów)
rng = random.Random(42)
eval_indices = list(range(len(eval_dataset_full)))
rng.shuffle(eval_indices)
eval_indices = eval_indices[:min(len(eval_indices), 1000)]
eval_dataset = eval_dataset_full.select(eval_indices)

class EvalPrinterCallback(TrainerCallback):
    """Drukuje kilka przykładowych predykcji z eval co ewaluację."""
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

            # zbuduj prompt jak w treningu
            before_input = f"### Instrukcja: {instruction}\n### Wejście: "
            after_input  = AFTER_INPUT

            ids_before = self.tokenizer(before_input, add_special_tokens=False)["input_ids"]
            ids_after  = self.tokenizer(after_input,  add_special_tokens=False)["input_ids"]

            special_count = int(self.tokenizer.bos_token_id is not None) + int(self.tokenizer.eos_token_id is not None)
            overhead = len(ids_before) + len(ids_after) + 1
            budget_for_input = max(1, min(CTX_CAP, MODEL_LIMIT) - special_count - overhead)

            ids_input = self.tokenizer(input_text, add_special_tokens=False)["input_ids"]
            if len(ids_input) > budget_for_input:
                ids_input = ids_input[:budget_for_input]

            ids_prompt = []
            if self.tokenizer.bos_token_id is not None:
                ids_prompt.append(self.tokenizer.bos_token_id)
            ids_prompt += (ids_before + ids_input + ids_after)

            input_ids = torch.tensor([ids_prompt], dtype=torch.long).to(model.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = out.logits[:, -1, :]
                logp = torch.log_softmax(next_token_logits, dim=-1)

            logp_true  = float(logp[0, TRUE_ID0].item())
            logp_false = float(logp[0, FALSE_ID0].item())
            pred = "True" if logp_true > logp_false else "False"
            margin = logp_true - logp_false

            print(f"[{idx}] gold={gold:5s} pred={pred:5s}  margin={margin:+.3f}  input_snip={repr(input_text[:120])}")
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
    eval_steps=200,              # <— co 50 kroków
    save_strategy="steps",
    save_steps=200,

    # lekka ewaluacja
    per_device_eval_batch_size=1,
    fp16_full_eval=True,
    eval_accumulation_steps=1,

    # wybór najlepszego modelu i limit checkpointów
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,

    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,   # <— tu przekazujemy LEKKI eval
    tokenizer=tokenizer,         # (ostrzeżenie o deprec. można zignorować)
    data_collator=data_collator,
    compute_metrics=compute_metrics_2class,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=5e-4
        ),
        EvalPrinterCallback(eval_dataset, tokenizer, k=5, seed=123),  # <— podgląd predykcji
    ],
)

print("Rozpoczęcie treningu PLLuM (True/False)...")
trainer.train()

# Ewaluacja końcowa (na najlepszym checkpointcie dzięki load_best_model_at_end=True)
results = trainer.evaluate()
print("Wyniki ewaluacji PLLuM:", results)
print("Najlepszy checkpoint:", trainer.state.best_model_checkpoint)

# ========================= 10) ZAPIS =========================
save_dir = "C:/treningpllum/"
os.makedirs(save_dir, exist_ok=True)
trainer.model.save_pretrained(save_dir)   # zapis najlepszego modelu
tokenizer.save_pretrained(save_dir)
print("Trening zakończony i model zapisany w:", save_dir)

# ========================= 11) INFERENCJA (logproby True/False) =========================

def classify_with_pllum(text: str) -> str:
    """
    Budujemy prompt jak w treningu i porównujemy logproby pierwszego tokenu „ True” vs „ False”.
    Z tym samym ograniczeniem kontekstu (ID-based).
    """
    model.eval()

    instruction = INSTRUCTION
    before_input = f"### Instrukcja: {instruction}\n### Wejście: "
    after_input  = AFTER_INPUT

    ids_before = tokenizer(before_input, add_special_tokens=False)["input_ids"]
    ids_after  = tokenizer(after_input,  add_special_tokens=False)["input_ids"]

    special_count = int(tokenizer.bos_token_id is not None) + int(tokenizer.eos_token_id is not None)
    overhead = len(ids_before) + len(ids_after) + 1  # miejsce na 1. token odpowiedzi
    budget_for_input = max(1, min(CTX_CAP, MODEL_LIMIT) - special_count - overhead)

    ids_input = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids_input) > budget_for_input:
        ids_input = ids_input[:budget_for_input]

    # Składamy prompt tokenami
    ids_prompt = []
    if tokenizer.bos_token_id is not None:
        ids_prompt.append(tokenizer.bos_token_id)
    ids_prompt += (ids_before + ids_input + ids_after)

    input_ids = torch.tensor([ids_prompt], dtype=torch.long).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        next_token_logits = out.logits[:, -1, :]
        logp = torch.log_softmax(next_token_logits, dim=-1)

    return "True" if logp[0, TRUE_ID0] > logp[0, FALSE_ID0] else "False"
