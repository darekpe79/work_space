# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 08:14:20 2025

@author: darek
"""

"""
Fine-tuning PLLuM-8B-instruct (8-bit + LoRA) jako klasyfikatora
„important for research” na pojedynczej karcie GPU 16 GB.

• kontekst 1 024 tokenów  (mieści się na 16 GB w 8-bit + LoRA)
• obcinamy **z lewej** – koniec promptu („Odpowiedź: Tak/Nie”) nigdy nie ginie
• dynamiczny padding (“longest”) ⇒ brak zbędnych <pad> przy krótkich przykładach
"""

import ast, torch, pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------- 1. Dane --------------------------------------------------------
df = pd.read_excel(r"C:/Users/darek/Downloads/gortych_perlentaucher (1).xlsx")

def parse_cell(x):
    if pd.isna(x): return []
    if isinstance(x,(list,set)): return list(x)
    if isinstance(x,str):
        try:
            v = ast.literal_eval(x);  return list(v) if isinstance(v,(list,set)) else [v]
        except: return []
    return []

for col in ["keywords","rezensionsnotiz","stichworter"]:
    df[col] = df[col].apply(parse_cell)

df["keywords_text"] = df["keywords"].apply(" ".join)
df["combined_text"] = df.apply(
    lambda r: " ".join(filter(None, [
        " ".join(r["rezensionsnotiz"]),
        str(r.get("klappentext","")),
        r["keywords_text"],
        " ".join(r["stichworter"])
    ])), axis=1
)

df = df[df["important for research"].notna()]
df["important for research"] = df["important for research"].astype(int).astype(bool)

train_data = [{
    "instruction": "Czy ten tekst jest istotny dla badań naukowych?",
    "input"      : row["combined_text"],
    "output"     : "Tak" if row["important for research"] else "Nie"
} for _, row in df.iterrows()]

# ---------- 2. Model & tokenizer ------------------------------------------
model_name = "CYFRAGOVPL/Llama-PLLuM-8B-instruct"
quant_cfg  = BitsAndBytesConfig(load_in_8bit=True,
                                llm_int8_threshold=6.0,
                                llm_int8_has_fp16_weight=False)

tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token        = tokenizer.eos_token
tokenizer.truncation_side  = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto",
    quantization_config=quant_cfg, trust_remote_code=True
)

lora_cfg = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    task_type=TaskType.CAUSAL_LM, inference_mode=False
)
model = get_peft_model(model, lora_cfg)

# ---------- 3. Tokenizacja -------------------------------------------------
MAX_LEN = 768
IGNORE  = -100

def build_prompt(inst, inp, out=None):
    base = f"### Instrukcja: {inst}\n### Wejście: {inp}\n### Odpowiedź:"
    return base if out is None else f"{base} {out}"

def tokenize(ex):
    full_prompt = build_prompt(ex["instruction"], ex["input"], ex["output"])
    prompt_only = build_prompt(ex["instruction"], ex["input"])

    # ❶ długość promptu BEZ paddingu
    prompt_ids = tokenizer(prompt_only,
                           truncation=True,
                           max_length=MAX_LEN,
                           add_special_tokens=False,
                           padding=False)["input_ids"]

    # ❷ pełny prompt Z paddingiem
    enc_full = tokenizer(full_prompt,
                         truncation=True,
                         max_length=MAX_LEN,
                         padding="max_length")

    labels = enc_full["input_ids"].copy()
    labels[:len(prompt_ids)] = [IGNORE]*len(prompt_ids)
    enc_full["labels"] = labels
    return enc_full

ds     = Dataset.from_list(train_data).map(tokenize, batched=False)
split  = ds.train_test_split(0.2, seed=42)
train_ds, eval_ds = split["train"], split["test"]

# ---------- 4. Metryki -----------------------------------------------------
id2label = {"Tak":"Tak", "Nie":"Nie"}  # ułatwi odczyt

def postprocess(pred_ids):
    txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    return ["Tak" if "Tak" in t else "Nie" for t in txt]

def compute_metrics(eval_pred):
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    # weź pierwszy przewidziany token po stronie odpowiedzi
    preds = [p[p != IGNORE][0:1] for p in preds]           #  shape (batch,1)
    labels= [l[l != IGNORE][0:1] for l in labels]

    preds_txt  = postprocess(preds)
    labels_txt = postprocess(labels)

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_txt, preds_txt, average="binary", pos_label="Tak", zero_division=0
    )
    acc = accuracy_score(labels_txt, preds_txt)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ---------- 5. TrainingArguments ------------------------------------------
args = TrainingArguments(
    output_dir               = "plluma-research-classifier",
    num_train_epochs         = 3,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate            = 5e-5,
    lr_scheduler_type        = "cosine",
    warmup_steps             = 20,
    optim                    = "adamw_torch",
    max_grad_norm            = 1.0,
    fp16                     = True,
    logging_steps            = 10,
    save_strategy            = "epoch",
    eval_strategy            = "epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ---------- 6. Trening -----------------------------------------------------
trainer.train()

# ---------- 7. Zapis -------------------------------------------------------
out_dir = "./llama-pllum-important-research-lora"
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print("\nModel wytrenowany i zapisany →", out_dir)


#%%
# import pandas as pd
# import ast
# from tqdm import tqdm
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report

# # ---- KROK 1: Parsowanie i wczytanie danych do treningu ----

# # Bezpieczna funkcja do konwersji x -> lista
# def parse_cell(x):
#     """ Zamienia string w listę lub set za pomocą ast.literal_eval,
#         obsługuje też puste/NaN. Zwraca listę (zamiast set). """
#     if pd.isna(x):  # np. NaN
#         return []
#     if isinstance(x, (list, set)):
#         return list(x)
#     if isinstance(x, str):
#         try:
#             val = ast.literal_eval(x)
#             if isinstance(val, set):
#                 val = list(val)
#             if isinstance(val, list):
#                 return val
#             else:
#                 # jeśli np. to jest pojedynczy string/liczba
#                 return [val]
#         except:
#             # jeśli string nie jest poprawnym wyrażeniem Python
#             return []
#     return []

# def to_bool(val):
#     """ Konwersja różnych formatów (0/1, True/False, stringi) do bool. """
#     if isinstance(val, bool):
#         return val
#     if isinstance(val, (int, float)):
#         return bool(int(val))
#     if isinstance(val, str):
#         lower_val = val.strip().lower()
#         if lower_val in ['true', '1']:
#             return True
#         if lower_val in ['false', '0']:
#             return False
#     return False

# # Wczytanie datasetu do treningu
# df = pd.read_excel("C:/Users/darek/Downloads/gortych_perlentaucher.xlsx")
# # Alternatywnie:
# # df = pd.read_excel("data/Gortych/gortych_perlentaucher.xlsx", sheet_name='Sheet1')

# # Parsowanie kolumn
# df['keywords'] = df['keywords'].apply(parse_cell)
# df['rezensionsnotiz'] = df['rezensionsnotiz'].apply(parse_cell)
# df['stichworter'] = df['stichworter'].apply(parse_cell)

# # Tworzymy keywords_text (string)
# df['keywords_text'] = df['keywords'].apply(lambda x: " ".join(x))

# # Budujemy combined_text
# df['combined_text'] = df.apply(
#     lambda row: ' '.join(filter(None, [
#         ' '.join(row['rezensionsnotiz']) if row['rezensionsnotiz'] else '',
#         row.get('klappentext', '') if isinstance(row.get('klappentext', ''), str) else '',
#         row['keywords_text'] if row['keywords_text'] else '',
#         ' '.join(row['stichworter']) if row['stichworter'] else ''
#     ])),
#     axis=1
# )

# # Czyścimy kolumnę z targetem
# df = df.dropna(subset=['important for research'])
# df['important for research'] = df['important for research'].apply(to_bool)
# print("Value counts for 'important for research':")
# print(df["important for research"].value_counts(dropna=False))

# # ---- KROK 2: Przygotowanie X/y, pipeline, trening ----

# X = df[['year', 'combined_text']]
# y = df['important for research']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, 
#     y, 
#     test_size=0.2, 
#     random_state=42
# )

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('text', TfidfVectorizer(stop_words='english'), 'combined_text'),
#         ('year', 'passthrough', ['year'])
#     ]
# )

# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])

# pipeline.fit(X_train, y_train)

# # Ewaluacja
# y_pred = pipeline.predict(X_test)
# print("\n===== EVALUATION ON TEST SPLIT =====")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Opcjonalny GridSearch
# param_grid = {
#     'classifier__n_estimators': [50, 100, 150],
#     'classifier__max_depth': [None, 10, 20]
# }

# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-val accuracy:", grid_search.best_score_)

# # ---- KROK 3: Funkcja predykcji na nowym wierszu ----

# def predict_book_usefulness(row):
#     """Tworzy combined_text na podstawie row i zwraca True/False przewidziane przez pipeline."""
#     # W Twoim kodzie: 
#     # reviews ~ rezensionsnotiz (lista/string)
#     # keywords_dg ~ keywords (set/string)
#     # stichworter ~ stichworter (lista)
#     # klappentext ~ description

#     reviews = " ".join(row['rezensionsnotiz']) if isinstance(row['rezensionsnotiz'], list) else str(row['rezensionsnotiz'])
#     keywords_dg = " ".join(row['keywords']) if isinstance(row['keywords'], list) else str(row['keywords'])
#     keywords_text = " ".join(row['stichworter']) if isinstance(row['stichworter'], list) else str(row['stichworter'])
#     description = row.get('klappentext', '')

#     # Połącz wszystko w jedną kolumnę
#     combined_text = f"{description or ''} {reviews or ''} {keywords_dg or ''} {keywords_text or ''}"

#     # Rok
#     year = row['year'] if 'year' in row else 9999  # default?

#     data = pd.DataFrame({'year': [year], 'combined_text': [combined_text]})
#     prediction = pipeline.predict(data)
#     return bool(prediction[0])

# # ---- KROK 4: Klasyfikacja drugiego pliku + porównanie z ground truth ----

# df_to_classify = pd.read_excel("C:/Users/darek/Downloads/gortych_perlentaucher.xlsx", sheet_name='Sheet1')

# # Parsujemy kolumny tak jak wyżej
# df_to_classify['keywords'] = df_to_classify['keywords'].apply(parse_cell)
# df_to_classify['rezensionsnotiz'] = df_to_classify['rezensionsnotiz'].apply(parse_cell)
# df_to_classify['stichworter'] = df_to_classify['stichworter'].apply(parse_cell)

# results = []
# for i, row in tqdm(df_to_classify.iterrows(), total=df_to_classify.shape[0]):
#     # Konwersja wiersza Series do dictionary, by predict_book_usefulness się nie wykrzaczył
#     row_dict = row.to_dict()
#     results.append(predict_book_usefulness(row_dict))

# # Dodajemy kolumnę z predykcjami
# df_to_classify['predicted_important_for_research'] = results

# # Jeśli mamy w tym pliku kolumnę "important for research", możemy porównać
# if 'important for research' in df_to_classify.columns:
#     # Usuwamy NaN
#     df_to_classify = df_to_classify[df_to_classify['important for research'].notna()]
#     # Konwersja do bool
#     df_to_classify['important_for_research_bool'] = df_to_classify['important for research'].apply(to_bool)
#     df_to_classify['predicted_important_for_research'] = df_to_classify['predicted_important_for_research'].apply(to_bool)

#     # Tworzymy mini-df do raportu
#     y_true = df_to_classify['important_for_research_bool']
#     y_pred = df_to_classify['predicted_important_for_research']

#     print("\n===== EVALUATION ON GORTYCH_PERLENTAUCHER =====")
#     print("Accuracy:", accuracy_score(y_true, y_pred))
#     print("Classification Report:\n", classification_report(y_true, y_pred))
# else:
#     print("\nBrak kolumny 'important for research' w pliku do klasyfikacji. Nie można porównać z ground truth.")

# # Możesz też zapisać do pliku, jeśli chcesz
# # df_to_classify.to_excel("data/Gortych/gortych_perlentaucher_predicted.xlsx", index=False)
# print("\nZrobione!")



# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# import ast
# from tqdm import tqdm

# #%%
# # Wczytanie danych z pliku Excel (dostosuj nazwę pliku i arkusz w razie potrzeby)
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# import ast
# from tqdm import tqdm
# # Wczytanie danych z pliku Excel (dostosuj nazwę pliku i arkusz w razie potrzeby)
# df = pd.read_excel("C:/Users/darek/Downloads/gortych_perlentaucher.xlsx")
# # df = pd.read_excel("data/Gortych/gortych_perlentaucher.xlsx", sheet_name='Sheet1')
# df = df.dropna(subset=['important for research'])
# # Załóżmy, że kolumny nazywają się: 'year', 'keywords', 'description', 'reviews', 'important_for_research'
# # Przygotowanie kolumny tekstowej – łączymy opis, recenzje i słowa kluczowe
# df['keywords'] = df['keywords'].apply(lambda x: ast.literal_eval(x))
# df['keywords_text'] = df['keywords'].apply(lambda x: " ".join(x) if isinstance(x, set) else str(x))
# df['rezensionsnotiz'] = df['rezensionsnotiz'].apply(lambda x: ast.literal_eval(x))
# df['stichworter'] = df['stichworter'].apply(lambda x: ast.literal_eval(x))
# df['combined_text'] = df.apply(
#     lambda row: ' '.join(filter(None, [
#         ' '.join(row['rezensionsnotiz']) if row['rezensionsnotiz'] else '',
#         row['klappentext'] if row['klappentext'] else '',
#         row['keywords_text'] if row['keywords_text'] else '',
#         ' '.join(row['stichworter']) if row['stichworter'] else ''
#     ])),
#     axis=1
# )

# # Przygotowanie zbioru cech i etykiety
# X = df[['year', 'combined_text']]
# y = df['important for research']

# # Podział na zbiór treningowy i testowy
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Definicja przetwarzania kolumn
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('text', TfidfVectorizer(stop_words='english'), 'combined_text'),
#         ('year', 'passthrough', ['year'])
#     ]
# )

# # Budowanie pipeline’u: preprocessing -> klasyfikator
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])

# # Trenowanie klasyfikatora
# pipeline.fit(X_train, y_train)

# # Ewaluacja na zbiorze testowym
# y_pred = pipeline.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# # Opcjonalnie: GridSearchCV do optymalizacji hiperparametrów
# param_grid = {
#     'classifier__n_estimators': [50, 100, 150],
#     'classifier__max_depth': [None, 10, 20]
# }

# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-val accuracy:", grid_search.best_score_)

# # Użycie wytrenowanego modelu na nowych danych
# def predict_book_usefulness(year, description, reviews, keywords_dg, keywords):
#     reviews = " ".join(reviews) if isinstance(reviews, list) else str(reviews)
#     keywords_dg = " ".join(keywords_dg) if isinstance(keywords_dg, set) else str(keywords_dg)
#     keywords_text = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
#     combined_text = f"{description} {reviews} {keywords_dg} {keywords_text}"
#     data = pd.DataFrame({'year': [year], 'combined_text': [combined_text]})
#     prediction = pipeline.predict(data)
#     return prediction[0]

# # Przykład użycia funkcji
# example_prediction = predict_book_usefulness(
#     year=2005,
#     description="Przykładowy opis książki zawierający odniesienia do przemocy prawicowej i neonazizmu.",
#     reviews="Recenzja podkreśla intersekcjonalne ujęcie przemocy.",
#     keywords=["przemoc prawicowa", "intersekcjonalna"]
# )

# example_prediction = predict_book_usefulness(
#     year=2023,
#     description='Deutschland, Anfang der siebziger Jahre: ein Land voller Angst vor allem Fremden. Der einzige Italiener an der Schule wirkt wie ein außerirdisches Wesen. In den Achtzigern sind es die Türken, die zum ersten Mal die Tische vor die Wirtschaft stellen. Während die Wetterauer den ersten Döner im Landkreis als Widerstandsnahrung feiern, erobert der lange verschwundene Hitler den öffentlichen Raum in Funk und Fernsehen. In den Neunzigern träumt der Erzähler seinen großen Traum vom Wetterauer Land, verschwindet allerdings erst mal mit seiner Cousine unter einer Bettdecke am Ostrand der neuen Republik. Die Heimkunft gelingt innerfamiliär, das Haus der Großmutter wird als musealer Ort rekonstruiert, während im Ort wenigstens der Grundriss der 1938 niedergebrannten Synagoge wiederhergestellt wird. Aber noch im neuen Jahrtausend, als die ganze Republik ständig den Begriff "Heimat" diskutiert, will niemand vom früheren Leben in der konkreten Heimat wissen, als es die noch gab, die es seit ihrer Deportation nicht mehr gab. Mit Gespür für alles Abgründige in der gelebten Normalität erzählt Andreas Maier von Deutschland zwischen Weltkrieg, Mauerfall und Jahrtausendwende; davon, wie es sich die Menschen gemütlich machen in vierzig Jahren Geschichte.',
#     reviews=['Ebenso "intim wie ausschweifend" ist der neunte Band von Andreas Maiers Erzählzyklus "Ortsumgehung" Rezensent Paul Jandl zufolge. Der Autor setzt sich mit seiner Kindheit und Jugend in der Wetterau bei Frankfurt auseinander und zeichnet gleichzeitig ein Bild Deutschlands von den siebziger bis zu den neunziger Jahren, lesen wir. Maier wirft dabei anekdotische Schlaglichter, so Jandl auf eine "linksutopische Jugend mit Batiktüchern", die Aufklärung über den Holocaust in der Schule, den Mauerfall. Der Roman wird dabei durchzogen von der Frage, was Heimat sein kann und wem man die Definition von Heimat gerade nicht überlassen sollte, schreibt der Kritiker, nämlich denen, die mit ihr nur "herumtümeln" wollen. Vor allem die Erinnerung selbst wird in diesem Roman zur Heimat, schließt Jandl.', 'Was zunächst wie eine gekonnt erzählte Anekdote erscheinen mag, trifft in Wahrheit ins "Grundsätzliche", betont Rezensent Dirk Knipphals. Andreas Meier schreibt deutsche Mentalitätsgeschichte, und zwar immer entlang konkreter, individueller Erfahrungen. Die Dynamik seines Textes gleicht allerdings weniger einem bedächtigen Schürfen, sondern einem sprunghaften Zupacken. Die Szenen aus vergangenen Alltagen, die er so erhascht, zeigen eindrücklich, wie weit entfernt von uns diese Vergangenheiten einerseits sind und wie präsent sie doch andererseits sind als Grundierung gegenwärtiger Lebenswelten. Anders gesagt: Es geht darum, sich Heimat zu erschreiben - was sowohl Aneignungs- als auch Distanzierungsprozesse mit einschließt. Schließlich ist Meier in einem Spalt aufgewachsen, zwischen einem "affirmativ noch mit Blut und Boden gründelnden" Heimatbegriff einerseits und der kritischen Ablehnung in linken Diskursen andererseits. Diese Lücke will er nicht etwa schließen, betont Knipphals, sondern eher "literarisch ausmessen". Doch all das, was sich über diesen Roman sagen lässt, bzw. was sich über das sagen lässt, was dieser Roman sagen kann, klingt viel zu abstrakt, bemerkt der Rezensent, besser liest man es bei Meier - stets konkret.', 'Rezensent Marcus Hladec liest mit diesem Buch von Andreas Maier, das den neunten Band seines erzählerischen Zyklus "Ortsumgehung" darstellt, einen "autobiografischen Essay-Roman", in dem der Autor Erinnerungen an seine eigene Heimat niederschreibt, und dabei die Bedeutung des Begriffes in all seinen Dimensionen zu fassen versucht. Maiers Buch beginnt in den siebziger Jahren, so der Rezensent, Heimatfilme laufen im Fernsehen, die Deutschen haben Angst vor allem Fremden, in der Schule werden Ausländer ausgegrenzt. Von hier bis in die Nuller-Jahre greift der Autor unterschiedliche Aspekte des Heimat-Themas auf, einen Schüleraustausch, die NS-Aufklärung in der Schule, die Mauer und ihren Fall, bis hin zum ehelich-häuslichen Leben als Erwachsener. Mit Wortexperimenten und philosophischen Reflexionen umkreist der Autor dabei den Begriff und das Konzept "Heimat", schreibt Hladec, und zeigt dabei dem Rezensenten zu Folge vor allem, wie schillernd dessen Bedeutung ist.', 'Der neunte autofiktionale Streich von Andreas Maier liegt vor und Rezensentin Martina Wagner-Egelhaaf ist sehr angetan von den Kindheitserinnerungen des Autors, der sich die Wirtschaftswunderjahre vorgeknöpft hat. Was und wie Maier über die Zeit knapp 15 Jahre nach der Shoah aus der Ich-Perspektive erzählt, ist in einem pädagogisch klugen Duktus geschrieben und wirke dabei zugleich melancholisch und rigide, findet die Rezensentin. Heimat sei für den Autor ein Synonym für das Schweigen. Dessen Dämonen hätten Maier zu einem Heimatlosen gemacht. Wagner-Egelhaaf kann gut nachvollziehen, wieso der Autor sein Buch Edgar Reitz gewidmet hat.'],
#     keywords_dg = {'neunziger jahre', 'mauerfall', 'neunziger ', 'mauer', 'wende'},
#     keywords=['Maier, Andreas', 'Heimat', 'Bundesrepublik Deutschland', 'Bonner Republik', 'Wende', 'Westdeutschland', 'Autofiktion', 'Wirtschaftswunder']
# )


# print("Przydatna do badań:", example_prediction)

# #%%
# # Wczytanie danych z pliku Excel (dostosuj nazwę pliku i arkusz w razie potrzeby)
# df = pd.read_excel("data/Gortych/gewalt_research_examples.xlsx")
# # df = pd.read_excel("data/Gortych/gortych_perlentaucher.xlsx", sheet_name='Sheet1')

# # Załóżmy, że kolumny nazywają się: 'year', 'keywords', 'description', 'reviews', 'important_for_research'
# # Przygotowanie kolumny tekstowej – łączymy opis, recenzje i słowa kluczowe
# df['keywords'] = df['keywords'].apply(lambda x: ast.literal_eval(x))
# df['keywords_text'] = df['keywords'].apply(lambda x: " ".join(x) if isinstance(x, set) else str(x))
# df['rezensionsnotiz'] = df['rezensionsnotiz'].apply(lambda x: ast.literal_eval(x))
# df['stichworter'] = df['stichworter'].apply(lambda x: ast.literal_eval(x))
# df['combined_text'] = df.apply(
#     lambda row: ' '.join(filter(None, [
#         ' '.join(row['rezensionsnotiz']) if row['rezensionsnotiz'] else '',
#         row['klappentext'] if row['klappentext'] else '',
#         row['keywords_text'] if row['keywords_text'] else '',
#         ' '.join(row['stichworter']) if row['stichworter'] else ''
#     ])),
#     axis=1
# )

# # Przygotowanie zbioru cech i etykiety
# X = df[['year', 'combined_text']]
# y = df['important for research']

# # Podział na zbiór treningowy i testowy
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Definicja przetwarzania kolumn
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('text', TfidfVectorizer(stop_words='english'), 'combined_text'),
#         ('year', 'passthrough', ['year'])
#     ]
# )

# # Budowanie pipeline’u: preprocessing -> klasyfikator
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])

# # Trenowanie klasyfikatora
# pipeline.fit(X_train, y_train)

# # Ewaluacja na zbiorze testowym
# y_pred = pipeline.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# # Opcjonalnie: GridSearchCV do optymalizacji hiperparametrów
# param_grid = {
#     'classifier__n_estimators': [50, 100, 150],
#     'classifier__max_depth': [None, 10, 20]
# }

# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-val accuracy:", grid_search.best_score_)

# # Użycie wytrenowanego modelu na nowych danych
# def predict_book_usefulness(year, description, reviews, keywords_dg, keywords):
#     reviews = " ".join(reviews) if isinstance(reviews, list) else str(reviews)
#     keywords_dg = " ".join(keywords_dg) if isinstance(keywords_dg, set) else str(keywords_dg)
#     keywords_text = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
#     combined_text = f"{description} {reviews} {keywords_dg} {keywords_text}"
#     data = pd.DataFrame({'year': [year], 'combined_text': [combined_text]})
#     prediction = pipeline.predict(data)
#     return prediction[0]

# # Przykład użycia funkcji
# example_prediction = predict_book_usefulness(
#     year=2005,
#     description="Przykładowy opis książki zawierający odniesienia do przemocy prawicowej i neonazizmu.",
#     reviews="Recenzja podkreśla intersekcjonalne ujęcie przemocy.",
#     keywords=["przemoc prawicowa", "intersekcjonalna"]
# )

# example_prediction = predict_book_usefulness(
#     year=2023,
#     description='Deutschland, Anfang der siebziger Jahre: ein Land voller Angst vor allem Fremden. Der einzige Italiener an der Schule wirkt wie ein außerirdisches Wesen. In den Achtzigern sind es die Türken, die zum ersten Mal die Tische vor die Wirtschaft stellen. Während die Wetterauer den ersten Döner im Landkreis als Widerstandsnahrung feiern, erobert der lange verschwundene Hitler den öffentlichen Raum in Funk und Fernsehen. In den Neunzigern träumt der Erzähler seinen großen Traum vom Wetterauer Land, verschwindet allerdings erst mal mit seiner Cousine unter einer Bettdecke am Ostrand der neuen Republik. Die Heimkunft gelingt innerfamiliär, das Haus der Großmutter wird als musealer Ort rekonstruiert, während im Ort wenigstens der Grundriss der 1938 niedergebrannten Synagoge wiederhergestellt wird. Aber noch im neuen Jahrtausend, als die ganze Republik ständig den Begriff "Heimat" diskutiert, will niemand vom früheren Leben in der konkreten Heimat wissen, als es die noch gab, die es seit ihrer Deportation nicht mehr gab. Mit Gespür für alles Abgründige in der gelebten Normalität erzählt Andreas Maier von Deutschland zwischen Weltkrieg, Mauerfall und Jahrtausendwende; davon, wie es sich die Menschen gemütlich machen in vierzig Jahren Geschichte.',
#     reviews=['Ebenso "intim wie ausschweifend" ist der neunte Band von Andreas Maiers Erzählzyklus "Ortsumgehung" Rezensent Paul Jandl zufolge. Der Autor setzt sich mit seiner Kindheit und Jugend in der Wetterau bei Frankfurt auseinander und zeichnet gleichzeitig ein Bild Deutschlands von den siebziger bis zu den neunziger Jahren, lesen wir. Maier wirft dabei anekdotische Schlaglichter, so Jandl auf eine "linksutopische Jugend mit Batiktüchern", die Aufklärung über den Holocaust in der Schule, den Mauerfall. Der Roman wird dabei durchzogen von der Frage, was Heimat sein kann und wem man die Definition von Heimat gerade nicht überlassen sollte, schreibt der Kritiker, nämlich denen, die mit ihr nur "herumtümeln" wollen. Vor allem die Erinnerung selbst wird in diesem Roman zur Heimat, schließt Jandl.', 'Was zunächst wie eine gekonnt erzählte Anekdote erscheinen mag, trifft in Wahrheit ins "Grundsätzliche", betont Rezensent Dirk Knipphals. Andreas Meier schreibt deutsche Mentalitätsgeschichte, und zwar immer entlang konkreter, individueller Erfahrungen. Die Dynamik seines Textes gleicht allerdings weniger einem bedächtigen Schürfen, sondern einem sprunghaften Zupacken. Die Szenen aus vergangenen Alltagen, die er so erhascht, zeigen eindrücklich, wie weit entfernt von uns diese Vergangenheiten einerseits sind und wie präsent sie doch andererseits sind als Grundierung gegenwärtiger Lebenswelten. Anders gesagt: Es geht darum, sich Heimat zu erschreiben - was sowohl Aneignungs- als auch Distanzierungsprozesse mit einschließt. Schließlich ist Meier in einem Spalt aufgewachsen, zwischen einem "affirmativ noch mit Blut und Boden gründelnden" Heimatbegriff einerseits und der kritischen Ablehnung in linken Diskursen andererseits. Diese Lücke will er nicht etwa schließen, betont Knipphals, sondern eher "literarisch ausmessen". Doch all das, was sich über diesen Roman sagen lässt, bzw. was sich über das sagen lässt, was dieser Roman sagen kann, klingt viel zu abstrakt, bemerkt der Rezensent, besser liest man es bei Meier - stets konkret.', 'Rezensent Marcus Hladec liest mit diesem Buch von Andreas Maier, das den neunten Band seines erzählerischen Zyklus "Ortsumgehung" darstellt, einen "autobiografischen Essay-Roman", in dem der Autor Erinnerungen an seine eigene Heimat niederschreibt, und dabei die Bedeutung des Begriffes in all seinen Dimensionen zu fassen versucht. Maiers Buch beginnt in den siebziger Jahren, so der Rezensent, Heimatfilme laufen im Fernsehen, die Deutschen haben Angst vor allem Fremden, in der Schule werden Ausländer ausgegrenzt. Von hier bis in die Nuller-Jahre greift der Autor unterschiedliche Aspekte des Heimat-Themas auf, einen Schüleraustausch, die NS-Aufklärung in der Schule, die Mauer und ihren Fall, bis hin zum ehelich-häuslichen Leben als Erwachsener. Mit Wortexperimenten und philosophischen Reflexionen umkreist der Autor dabei den Begriff und das Konzept "Heimat", schreibt Hladec, und zeigt dabei dem Rezensenten zu Folge vor allem, wie schillernd dessen Bedeutung ist.', 'Der neunte autofiktionale Streich von Andreas Maier liegt vor und Rezensentin Martina Wagner-Egelhaaf ist sehr angetan von den Kindheitserinnerungen des Autors, der sich die Wirtschaftswunderjahre vorgeknöpft hat. Was und wie Maier über die Zeit knapp 15 Jahre nach der Shoah aus der Ich-Perspektive erzählt, ist in einem pädagogisch klugen Duktus geschrieben und wirke dabei zugleich melancholisch und rigide, findet die Rezensentin. Heimat sei für den Autor ein Synonym für das Schweigen. Dessen Dämonen hätten Maier zu einem Heimatlosen gemacht. Wagner-Egelhaaf kann gut nachvollziehen, wieso der Autor sein Buch Edgar Reitz gewidmet hat.'],
#     keywords_dg = {'neunziger jahre', 'mauerfall', 'neunziger ', 'mauer', 'wende'},
#     keywords=['Maier, Andreas', 'Heimat', 'Bundesrepublik Deutschland', 'Bonner Republik', 'Wende', 'Westdeutschland', 'Autofiktion', 'Wirtschaftswunder']
# )


# print("Przydatna do badań:", example_prediction)


# #%%

# def predict_book_usefulness(row):
#     reviews = " ".join(row['rezensionsnotiz']) if isinstance(row['rezensionsnotiz'], list) else str(row['rezensionsnotiz'])
#     keywords_dg = " ".join(row['keywords']) if isinstance(row['keywords'], set) else str(row['keywords'])
#     keywords_text = " ".join(row['stichworter']) if isinstance(row['stichworter'], list) else str(row['stichworter'])
#     description = row['klappentext']
#     combined_text = f"{description} {reviews} {keywords_dg} {keywords_text}"
#     data = pd.DataFrame({'year': [row['year']], 'combined_text': [combined_text]})
#     prediction = pipeline.predict(data)
#     return bool(prediction[0])


# df_to_classify = pd.read_excel("data/Gortych/gortych_perlentaucher.xlsx", sheet_name='Sheet1')

# df_to_classify['keywords'] = df_to_classify['keywords'].apply(lambda x: ast.literal_eval(x))
# df_to_classify['keywords_text'] = df_to_classify['keywords'].apply(lambda x: " ".join(x) if isinstance(x, set) else str(x))
# df_to_classify['rezensionsnotiz'] = df_to_classify['rezensionsnotiz'].apply(lambda x: ast.literal_eval(x))
# df_to_classify['stichworter'] = df_to_classify['stichworter'].apply(lambda x: ast.literal_eval(x))

# results = []
# for i, row in tqdm(df_to_classify.iterrows(), total=df_to_classify.shape[0]):
#     results.append(predict_book_usefulness(row))
    
# df_test = pd.DataFrame(results)
