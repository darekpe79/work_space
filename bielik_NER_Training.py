# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:27:55 2025

@author: darek
"""

# -*- coding: utf-8 -*-
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ===================== 1. Dane =====================



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

# prompt w stylu instrukcyjnym
def format_prompt(example):
    return f"""### Instrukcja:
{example["instruction"]}

### Wejście:
{example["input"]}

### Odpowiedź:
{example["output"]}"""

dataset = Dataset.from_list(train_data)
dataset = dataset.map(lambda x: {"text": format_prompt(x)})

# ===================== 2. Model i tokenizer =====================

model_name = "speakleash/Bielik-4.5B-v3.0-Instruct"

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # dla bezpieczeństwa

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)

# przygotowanie pod LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_config)

# ===================== 3. Tokenizacja =====================

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# collator do prostego LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===================== 4. Trening =====================

args = TrainingArguments(
    output_dir="./bielik-title-finetune",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=10,
    logging_steps=5,
    save_steps=50,
    evaluation_strategy="no",  # szybki trening
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("🚀 Start treningu Bielika...")
trainer.train()

# ===================== 5. Testujemy model =====================

prompt = """### Instrukcja:
Rozpoznaj tytuł w tekście poniżej:

### Wejście:
Uwielbiam film 'Forrest Gump', oglądałem go wiele razy.

### Odpowiedź:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print("\n🧩 Odpowiedź modelu:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
