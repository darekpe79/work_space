# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 11:39:51 2025

@author: darek
"""

# === Unsloth + QLoRA (4-bit) dla CYFRAGOVPL/Llama-PLLuM-8B-instruct ===
# Sprzęt: RTX 4070 Ti Super 16 GB

import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType
from unsloth import FastLanguageModel

# ------------------------------
# 1) Dane treningowe (Twoje)
# ------------------------------
train_data = [
    # KSIĄŻKI
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Niedawno przeczytałem książkę 'Lalka' autorstwa Bolesława Prusa.", "output": "'Lalka'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam powieść 'Duma i uprzedzenie' Jane Austen.", "output": "'Duma i uprzedzenie'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Przeczytałem książkę 'Zbrodnia i kara' Fiodora Dostojewskiego.", "output": "'Zbrodnia i kara'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Powieść 'Władca Pierścieni' to arcydzieło fantasy.", "output": "'Władca Pierścieni'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Czytałem niedawno 'Zabić drozda' Harper Lee.", "output": "'Zabić drozda'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Hobbit, czyli tam i z powrotem' to prequel do 'Władcy Pierścieni'.", "output": "'Hobbit, czyli tam i z powrotem'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Jedna z najlepszych powieści to 'Mistrz i Małgorzata'.", "output": "'Mistrz i Małgorzata'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Harry Potter i Komnata Tajemnic' kontynuuje przygody młodego czarodzieja.", "output": "'Harry Potter i Komnata Tajemnic'"},
    
    # FILMY
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Forrest Gump' zdobył wiele nagród filmowych.", "output": "'Forrest Gump'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Wczoraj obejrzałem film 'Titanic'.", "output": "'Titanic'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam 'Incepcję', to bardzo dobry film.", "output": "'Incepcja'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądałem 'Matrix' i byłem pod wrażeniem.", "output": "'Matrix'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Mroczny Rycerz' to najlepsza część trylogii Batmana.", "output": "'Mroczny Rycerz'"},
    
    # SERIALE
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Breaking Bad' zmienił telewizję.", "output": "'Breaking Bad'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Lubię 'Gra o Tron', ale końcówka mnie zawiodła.", "output": "'Gra o Tron'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Netflix ma świetne produkcje, jak 'Stranger Things'.", "output": "'Stranger Things'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Czarne Lustro' jest bardzo intrygujący.", "output": "'Czarne Lustro'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądam 'Sherlock' i podziwiam grę aktorską.", "output": "'Sherlock'"},

    # GRY
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Wiedźmin 3: Dziki Gon' to najlepsza część serii.", "output": "'Wiedźmin 3: Dziki Gon'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gram w 'The Legend of Zelda: Breath of the Wild'.", "output": "'The Legend of Zelda: Breath of the Wild'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Najnowsza gra 'Cyberpunk 2077' miała trudny start.", "output": "'Cyberpunk 2077'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gram w 'Minecraft' od lat.", "output": "'Minecraft'"},
    
    # OBRAZY
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Mona Lisa' to arcydzieło Leonarda da Vinci.", "output": "'Mona Lisa'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam 'Gwiaździstą noc' Vincenta van Gogha.", "output": "'Gwiaździsta noc'"},
    
    # ALBUMY MUZYCZNE
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Album 'The Dark Side of the Moon' zespołu Pink Floyd jest legendarny.", "output": "'The Dark Side of the Moon'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słuchałem albumu 'Abbey Road' zespołu The Beatles.", "output": "'Abbey Road'"},
    
    # DODATKOWE PRZYKŁADY
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "'Gladiator' to film, który oglądałem wielokrotnie.", "output": "'Gladiator'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądałem 'Dexter' i było świetnie!", "output": "'Dexter'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Czy książka 'Mały Książę' jest dla dzieci?", "output": "'Mały Książę'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Nie mogę zapomnieć o 'Bohemian Rhapsody' Queen.", "output": "'Bohemian Rhapsody'"},
    
    # RÓŻNE SPOSOBY PISANIA
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Czytałem książkę pt. 'Opowieści z Narnii'.", "output": "'Opowieści z Narnii'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial pt. 'Narcos' bardzo mi się podobał.", "output": "'Narcos'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Dzieło 'Hamlet' Williama Szekspira to dramat wszech czasów.", "output": "'Hamlet'"},
{"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Sto lat samotności' Gabriela Garcíi Márqueza to klasyk literatury.", "output": "'Sto lat samotności'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Przeczytałem 'Wielki Gatsby' i byłem pod wrażeniem.", "output": "'Wielki Gatsby'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Powieść 'Anna Karenina' Lwa Tołstoja to arcydzieło.", "output": "'Anna Karenina'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Fahrenheit 451' Ray'a Bradbury'ego jest bardzo wciągająca.", "output": "'Fahrenheit 451'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Czytałem 'Rok 1984' i byłem zszokowany.", "output": "'Rok 1984'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Buszujący w zbożu' J.D. Salingera to klasyk.", "output": "'Buszujący w zbożu'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Powieść 'Dzieci z Bullerbyn' to ulubiona książka z dzieciństwa.", "output": "'Dzieci z Bullerbyn'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Kod Leonarda da Vinci' była bestsellerem.", "output": "'Kod Leonarda da Vinci'"},

    # FILMY
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Pulp Fiction' Quentina Tarantino to klasyk.", "output": "'Pulp Fiction'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądałem 'Skazani na Shawshank' i byłem wzruszony.", "output": "'Skazani na Shawshank'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Ojciec chrzestny' to arcydzieło kina.", "output": "'Ojciec chrzestny'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam 'Szósty zmysł', bo ma niesamowite zakończenie.", "output": "'Szósty zmysł'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Whiplash' pokazuje, jak daleko można się posunąć dla pasji.", "output": "'Whiplash'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądałem 'La La Land' i byłem zachwycony muzyką.", "output": "'La La Land'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Joker' z Joaquinem Phoenixem był bardzo mroczny.", "output": "'Joker'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Ostatnio obejrzałem 'Parasite' i byłem pod wrażeniem.", "output": "'Parasite'"},

    # SERIALE
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'The Office' to jedna z najlepszych komedii.", "output": "'The Office'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądam 'Friends' i uwielbiam humor.", "output": "'Friends'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'The Crown' opowiada o brytyjskiej rodzinie królewskiej.", "output": "'The Crown'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam 'The Mandalorian' ze względu na postać Baby Yoda.", "output": "'The Mandalorian'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'The Witcher' oparty jest na książkach Andrzeja Sapkowskiego.", "output": "'The Witcher'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądam 'Black Mirror' i jestem zafascynowany futurystycznymi wizjami.", "output": "'Black Mirror'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'The Big Bang Theory' to jedna z moich ulubionych komedii.", "output": "'The Big Bang Theory'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądałem 'House of Cards' i byłem pod wrażeniem gry Kevina Spacey'ego.", "output": "'House of Cards'"},

    # GRY
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Red Dead Redemption 2' ma świetną fabułę.", "output": "'Red Dead Redemption 2'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gram w 'The Last of Us' i jestem pod wrażeniem emocji.", "output": "'The Last of Us'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'God of War' zdobyła wiele nagród.", "output": "'God of War'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam 'The Elder Scrolls V: Skyrim' za otwarty świat.", "output": "'The Elder Scrolls V: Skyrim'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Portal 2' ma genialne zagadki.", "output": "'Portal 2'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gram w 'Fortnite' z przyjaciółmi.", "output": "'Fortnite'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Among Us' stała się bardzo popularna w 2020 roku.", "output": "'Among Us'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gram w 'League of Legends' od lat.", "output": "'League of Legends'"},

    # OBRAZY
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Ostatnia Wieczerza' Leonarda da Vinci to arcydzieło.", "output": "'Ostatnia Wieczerza'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam 'Krzyk' Edvarda Muncha.", "output": "'Krzyk'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Dziewczyna z perłą' jest bardzo znany.", "output": "'Dziewczyna z perłą'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Dzieło 'Guernica' Pabla Picassa jest bardzo poruszające.", "output": "'Guernica'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Słoneczniki' Vincenta van Gogha jest piękny.", "output": "'Słoneczniki'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Dzieło 'Narodziny Wenus' Sandro Botticellego to klasyk.", "output": "'Narodziny Wenus'"},

    # ALBUMY MUZYCZNE
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Album 'Thriller' Michaela Jacksona to najlepiej sprzedający się album wszech czasów.", "output": "'Thriller'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słuchałem 'Back in Black' zespołu AC/DC.", "output": "'Back in Black'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Album 'Rumours' Fleetwood Mac to klasyk.", "output": "'Rumours'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam 'The Wall' Pink Floyd.", "output": "'The Wall'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Album 'Nevermind' Nirvany zmienił muzykę rockową.", "output": "'Nevermind'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słuchałem '21' Adele i byłem zachwycony.", "output": "'21'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Album 'Lemonade' Beyoncé to arcydzieło.", "output": "'Lemonade'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Uwielbiam 'Ok Computer' Radiohead.", "output": "'Ok Computer'"},

    # RÓŻNE SPOSOBY PISANIA
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Czytałem książkę pt. 'Władca Pierścieni'.", "output": "'Władca Pierścieni'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial pt. 'Breaking Bad' jest bardzo popularny.", "output": "'Breaking Bad'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film pt. 'Incepcja' był bardzo skomplikowany.", "output": "'Incepcja'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra pt. 'Wiedźmin 3: Dziki Gon' zdobyła wiele nagród.", "output": "'Wiedźmin 3: Dziki Gon'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz pt. 'Mona Lisa' jest bardzo znany.", "output": "'Mona Lisa'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Album pt. 'The Dark Side of the Moon' to klasyk.", "output": "'The Dark Side of the Moon'"},
]

# ------------------------------
# 2) Model i tokenizer przez Unsloth
# ------------------------------
model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = "bfloat16" if use_bf16 else "float16"

MAX_LEN = 256

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    max_seq_length=MAX_LEN,
    dtype=dtype,
    device_map="auto",
)

tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------
# 3) Konfiguracja LoRA
# ------------------------------
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

model = FastLanguageModel.get_peft_model(model, lora_config)
model.print_trainable_parameters()

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# ------------------------------
# 4) Tokenizacja
# ------------------------------
IGNORE_INDEX = -100

def format_example(ex):
    return (
        f"### Instrukcja: {ex['instruction']}\n"
        f"### Wejście: {ex['input']}\n"
        f"### Odpowiedź: {ex['output']}"
    )

def format_prompt_for_mask(ex):
    return (
        f"### Instrukcja: {ex['instruction']}\n"
        f"### Wejście: {ex['input']}\n"
        f"### Odpowiedź:"
    )

def tokenize(example):
    prompt = format_example(example)
    tok = tokenizer(prompt, padding="max_length", truncation=True, max_length=MAX_LEN)
    labels = tok["input_ids"].copy()

    prompt_len = len(tokenizer(format_prompt_for_mask(example))["input_ids"])
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
    tok["labels"] = labels
    return tok

dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(tokenize, remove_columns=None)

split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# ------------------------------
# 5) Trening
# ------------------------------
training_args = TrainingArguments(
    output_dir="C:/treningpllum/checkpoint_unsloth",   # <<<<<< katalog wyników
    num_train_epochs=15,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    optim="adamw_bnb_8bit",
    max_grad_norm=2.0,
    fp16=(not use_bf16),
    bf16=use_bf16,
    logging_steps=2,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# ------------------------------
# 6) Zapis adapterów LoRA
# ------------------------------
save_dir = "C:/treningpllum/llama-pllum-title-lora-unsloth"   # <<<<<< katalog zapisania
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Trening zakończony i model zapisany w:", save_dir)

# ------------------------------
# (Opcjonalnie) Scal adaptery do pełnego modelu
# ------------------------------
# from peft import PeftModel
# base_model, _ = FastLanguageModel.from_pretrained(
#     model_name=model_name,
#     load_in_4bit=False,
#     dtype=dtype,
#     device_map="auto",
# )
# fused = PeftModel.from_pretrained(base_model, save_dir)
# fused = fused.merge_and_unload()
# fused.save_pretrained("C:/treningpllum/llama-pllum-title-merged")
# tokenizer.save_pretrained("C:/treningpllum/llama-pllum-title-merged")
