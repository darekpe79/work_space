import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

# Przykładowe dane treningowe
train_data = [
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Niedawno przeczytałem książkę 'Lalka' autorstwa Bolesława Prusa.", "output": "'Lalka'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Forrest Gump' zdobył wiele nagród filmowych.", "output": "'Forrest Gump'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Breaking Bad' ma wielu fanów na całym świecie.", "output": "'Breaking Bad'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słynny obraz 'Mona Lisa' znajduje się w muzeum Luwr.", "output": "'Mona Lisa'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Interstellar' Christophera Nolana porusza tematykę kosmosu.", "output": "'Interstellar'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka '1984' George'a Orwella opowiada o dystopijnej przyszłości.", "output": "'1984'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Wiedźmin 3: Dziki Gon' zdobyła wiele nagród.", "output": "'Wiedźmin 3: Dziki Gon'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Gwiaździsta noc' Vincenta van Gogha jest arcydziełem.", "output": "'Gwiaździsta noc'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Ostatnio obejrzałem film 'Incepcja' i zrobił na mnie wrażenie.", "output": "'Incepcja'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Piosenka 'Bohemian Rhapsody' zespołu Queen jest ikoną rocka.", "output": "'Bohemian Rhapsody'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Zbrodnia i kara' Fiodora Dostojewskiego to klasyk literatury.", "output": "'Zbrodnia i kara'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądałem ostatnio serial 'Gra o Tron'.", "output": "'Gra o Tron'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Gladiator' zdobył wiele Oscarów.", "output": "'Gladiator'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Czytałem książkę 'Hobbit, czyli tam i z powrotem' J.R.R. Tolkiena.", "output": "'Hobbit, czyli tam i z powrotem'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Titanic' opowiada tragiczną historię.", "output": "'Titanic'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Cyberpunk 2077' została wydana przez CD Projekt Red.", "output": "'Cyberpunk 2077'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Ostatnia Wieczerza' Leonarda da Vinci to arcydzieło.", "output": "'Ostatnia Wieczerza'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Mały Książę' jest jedną z najczęściej tłumaczonych.", "output": "'Mały Książę'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Matrix' stał się kultowy.", "output": "'Matrix'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Stranger Things' jest bardzo popularny.", "output": "'Stranger Things'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słuchałem albumu 'The Dark Side of the Moon' zespołu Pink Floyd.", "output": "'The Dark Side of the Moon'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Harry Potter i Kamień Filozoficzny' to początek serii.", "output": "'Harry Potter i Kamień Filozoficzny'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Shrek' to jedna z najlepszych animacji.", "output": "'Shrek'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'The Legend of Zelda: Breath of the Wild' zdobyła wiele nagród.", "output": "'The Legend of Zelda: Breath of the Wild'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Sherlock' opowiada o przygodach detektywa.", "output": "'Sherlock'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Władca Pierścieni' J.R.R. Tolkiena to klasyka fantasy.", "output": "'Władca Pierścieni'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Krzyk' Edvarda Muncha jest jednym z najbardziej znanych dzieł sztuki.", "output": "'Krzyk'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Grand Theft Auto V' sprzedała się w milionach egzemplarzy.", "output": "'Grand Theft Auto V'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Avatar' Jamesa Camerona odniósł ogromny sukces.", "output": "'Avatar'"},
]


# Model i tokenizer
model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# Konfiguracja LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenizacja
IGNORE_INDEX = -100
def tokenize(example):
    prompt = f"{example['instruction']} {example['input']}\nOdpowiedź: {example['output']}"
    tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
    labels = tokenized["input_ids"].copy()

    # Ignorowanie tokenów promptu w liczeniu loss
    prompt_len = len(tokenizer(f"{example['instruction']} {example['input']}\nOdpowiedź:")['input_ids'])
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    tokenized["labels"] = labels
    return tokenized

# Utworzenie datasetu
dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(tokenize)

# Trening
training_args = TrainingArguments(
    output_dir="llama-pllum-lora-title",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    optim="adamw_torch",  # AdamW dla stabilniejszych gradientów
    max_grad_norm=1.0,  # Clipping gradientów, by uniknąć NaN
    fp16=True,
    logging_steps=2,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Zapisanie modelu po treningu
model.save_pretrained("./llama-pllum-title-lora-trained")
tokenizer.save_pretrained("./llama-pllum-title-lora-trained")

print("Trening zakończony i model zapisany.")



import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

# Przykładowe dane treningowe (zwiększona liczba przykładów)
train_data = [
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Niedawno przeczytałem książkę 'Lalka' autorstwa Bolesława Prusa.", "output": "'Lalka'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Forrest Gump' zdobył wiele nagród filmowych.", "output": "'Forrest Gump'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Breaking Bad' ma wielu fanów na całym świecie.", "output": "'Breaking Bad'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słynny obraz 'Mona Lisa' znajduje się w muzeum Luwr.", "output": "'Mona Lisa'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Interstellar' Christophera Nolana porusza tematykę kosmosu.", "output": "'Interstellar'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka '1984' George'a Orwella opowiada o dystopijnej przyszłości.", "output": "'1984'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Wiedźmin 3: Dziki Gon' zdobyła wiele nagród.", "output": "'Wiedźmin 3: Dziki Gon'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Gwiaździsta noc' Vincenta van Gogha jest arcydziełem.", "output": "'Gwiaździsta noc'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Ostatnio obejrzałem film 'Incepcja' i zrobił na mnie wrażenie.", "output": "'Incepcja'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Piosenka 'Bohemian Rhapsody' zespołu Queen jest ikoną rocka.", "output": "'Bohemian Rhapsody'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Zbrodnia i kara' Fiodora Dostojewskiego to klasyk literatury.", "output": "'Zbrodnia i kara'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądałem ostatnio serial 'Gra o Tron'.", "output": "'Gra o Tron'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Gladiator' zdobył wiele Oscarów.", "output": "'Gladiator'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Czytałem książkę 'Hobbit, czyli tam i z powrotem' J.R.R. Tolkiena.", "output": "'Hobbit, czyli tam i z powrotem'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Titanic' opowiada tragiczną historię.", "output": "'Titanic'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Cyberpunk 2077' została wydana przez CD Projekt Red.", "output": "'Cyberpunk 2077'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Ostatnia Wieczerza' Leonarda da Vinci to arcydzieło.", "output": "'Ostatnia Wieczerza'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Mały Książę' jest jedną z najczęściej tłumaczonych.", "output": "'Mały Książę'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Matrix' stał się kultowy.", "output": "'Matrix'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Stranger Things' jest bardzo popularny.", "output": "'Stranger Things'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słuchałem albumu 'The Dark Side of the Moon' zespołu Pink Floyd.", "output": "'The Dark Side of the Moon'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Harry Potter i Kamień Filozoficzny' to początek serii.", "output": "'Harry Potter i Kamień Filozoficzny'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Shrek' to jedna z najlepszych animacji.", "output": "'Shrek'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'The Legend of Zelda: Breath of the Wild' zdobyła wiele nagród.", "output": "'The Legend of Zelda: Breath of the Wild'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Sherlock' opowiada o przygodach detektywa.", "output": "'Sherlock'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Władca Pierścieni' J.R.R. Tolkiena to klasyka fantasy.", "output": "'Władca Pierścieni'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Krzyk' Edvarda Muncha jest jednym z najbardziej znanych dzieł sztuki.", "output": "'Krzyk'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Grand Theft Auto V' sprzedała się w milionach egzemplarzy.", "output": "'Grand Theft Auto V'"},
    {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Avatar' Jamesa Camerona odniósł ogromny sukces.", "output": "'Avatar'"},
    # Dodaj więcej przykładów tutaj, aby zwiększyć różnorodność danych
]

# Model i tokenizer
model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# Konfiguracja LoRA
lora_config = LoraConfig(
    r=32,  # Zwiększony rank
    lora_alpha=64,  # Zwiększony alpha
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Więcej modułów
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenizacja
IGNORE_INDEX = -100
def tokenize(example):
    prompt = f"### Instrukcja: {example['instruction']}\n### Wejście: {example['input']}\n### Odpowiedź: {example['output']}"
    tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=256)
    labels = tokenized["input_ids"].copy()

    # Ignorowanie tokenów promptu w liczeniu loss
    prompt_len = len(tokenizer(f"### Instrukcja: {example['instruction']}\n### Wejście: {example['input']}\n### Odpowiedź:")['input_ids'])
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    tokenized["labels"] = labels
    return tokenized

# Utworzenie datasetu
dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(tokenize)

# Trening
training_args = TrainingArguments(
    output_dir="llama-pllum-lora-title",
    num_train_epochs=15,  # Zmniejszona liczba epok
    per_device_train_batch_size=2,  # Zwiększony batch size
    gradient_accumulation_steps=4,  # Zmniejszona akumulacja gradientu
    learning_rate=5e-5,  # Zmniejszony learning rate
    lr_scheduler_type="cosine",
    warmup_steps=5,
    optim="adamw_torch",
    max_grad_norm=2.0,  # Zwiększony max_grad_norm
    fp16=True,
    logging_steps=2,
    save_strategy="epoch",
    evaluation_strategy="epoch",  # Dodana walidacja
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Zapisanie modelu po treningu
model.save_pretrained("./llama-pllum-title-lora-trained")
tokenizer.save_pretrained("./llama-pllum-title-lora-trained")

print("Trening zakończony i model zapisany.")
