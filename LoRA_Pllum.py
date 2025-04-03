# import torch
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForCausalLM, 
#     BitsAndBytesConfig, 
#     TrainingArguments, 
#     Trainer
# )
# from peft import LoraConfig, get_peft_model, TaskType

# # Przykładowe dane treningowe
# train_data = [
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Niedawno przeczytałem książkę 'Lalka' autorstwa Bolesława Prusa.", "output": "'Lalka'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Forrest Gump' zdobył wiele nagród filmowych.", "output": "'Forrest Gump'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Breaking Bad' ma wielu fanów na całym świecie.", "output": "'Breaking Bad'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słynny obraz 'Mona Lisa' znajduje się w muzeum Luwr.", "output": "'Mona Lisa'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Interstellar' Christophera Nolana porusza tematykę kosmosu.", "output": "'Interstellar'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka '1984' George'a Orwella opowiada o dystopijnej przyszłości.", "output": "'1984'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Wiedźmin 3: Dziki Gon' zdobyła wiele nagród.", "output": "'Wiedźmin 3: Dziki Gon'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Gwiaździsta noc' Vincenta van Gogha jest arcydziełem.", "output": "'Gwiaździsta noc'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Ostatnio obejrzałem film 'Incepcja' i zrobił na mnie wrażenie.", "output": "'Incepcja'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Piosenka 'Bohemian Rhapsody' zespołu Queen jest ikoną rocka.", "output": "'Bohemian Rhapsody'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Zbrodnia i kara' Fiodora Dostojewskiego to klasyk literatury.", "output": "'Zbrodnia i kara'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Oglądałem ostatnio serial 'Gra o Tron'.", "output": "'Gra o Tron'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Gladiator' zdobył wiele Oscarów.", "output": "'Gladiator'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Czytałem książkę 'Hobbit, czyli tam i z powrotem' J.R.R. Tolkiena.", "output": "'Hobbit, czyli tam i z powrotem'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Titanic' opowiada tragiczną historię.", "output": "'Titanic'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Cyberpunk 2077' została wydana przez CD Projekt Red.", "output": "'Cyberpunk 2077'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Ostatnia Wieczerza' Leonarda da Vinci to arcydzieło.", "output": "'Ostatnia Wieczerza'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Mały Książę' jest jedną z najczęściej tłumaczonych.", "output": "'Mały Książę'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Matrix' stał się kultowy.", "output": "'Matrix'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Stranger Things' jest bardzo popularny.", "output": "'Stranger Things'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Słuchałem albumu 'The Dark Side of the Moon' zespołu Pink Floyd.", "output": "'The Dark Side of the Moon'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Harry Potter i Kamień Filozoficzny' to początek serii.", "output": "'Harry Potter i Kamień Filozoficzny'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Shrek' to jedna z najlepszych animacji.", "output": "'Shrek'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'The Legend of Zelda: Breath of the Wild' zdobyła wiele nagród.", "output": "'The Legend of Zelda: Breath of the Wild'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Serial 'Sherlock' opowiada o przygodach detektywa.", "output": "'Sherlock'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Książka 'Władca Pierścieni' J.R.R. Tolkiena to klasyka fantasy.", "output": "'Władca Pierścieni'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Obraz 'Krzyk' Edvarda Muncha jest jednym z najbardziej znanych dzieł sztuki.", "output": "'Krzyk'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Gra 'Grand Theft Auto V' sprzedała się w milionach egzemplarzy.", "output": "'Grand Theft Auto V'"},
#     {"instruction": "Rozpoznaj tytuł w tekście poniżej:", "input": "Film 'Avatar' Jamesa Camerona odniósł ogromny sukces.", "output": "'Avatar'"},
# ]


# # Model i tokenizer
# model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=quantization_config,
#     device_map="auto",
#     trust_remote_code=True
# )

# # Konfiguracja LoRA
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=64,
#     target_modules=["q_proj", "v_proj"],
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False
# )

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# # Tokenizacja
# IGNORE_INDEX = -100
# def tokenize(example):
#     prompt = f"{example['instruction']} {example['input']}\nOdpowiedź: {example['output']}"
#     tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=512)
#     labels = tokenized["input_ids"].copy()

#     # Ignorowanie tokenów promptu w liczeniu loss
#     prompt_len = len(tokenizer(f"{example['instruction']} {example['input']}\nOdpowiedź:")['input_ids'])
#     labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

#     tokenized["labels"] = labels
#     return tokenized

# # Utworzenie datasetu
# dataset = Dataset.from_list(train_data)
# tokenized_dataset = dataset.map(tokenize)

# # Trening
# training_args = TrainingArguments(
#     output_dir="llama-pllum-lora-title",
#     num_train_epochs=10,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,
#     learning_rate=1e-4,
#     lr_scheduler_type="cosine",
#     warmup_steps=5,
#     optim="adamw_torch",  # AdamW dla stabilniejszych gradientów
#     max_grad_norm=1.0,  # Clipping gradientów, by uniknąć NaN
#     fp16=True,
#     logging_steps=2,
#     save_strategy="epoch"
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     tokenizer=tokenizer,
# )

# trainer.train()

# # Zapisanie modelu po treningu
# model.save_pretrained("./llama-pllum-title-lora-trained")
# tokenizer.save_pretrained("./llama-pllum-title-lora-trained")

# print("Trening zakończony i model zapisany.")



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



# Podział danych na zestaw treningowy i walidacyjny (np. 80% treningowy, 20% walidacyjny)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

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
    evaluation_strategy="epoch",  # Włączona walidacja
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Dodany zestaw walidacyjny
    tokenizer=tokenizer,
)

trainer.train()

# Zapisanie modelu po treningu
model.save_pretrained("./llama-pllum-title-lora-trained")
tokenizer.save_pretrained("./llama-pllum-title-lora-trained")

print("Trening zakończony i model zapisany.")
