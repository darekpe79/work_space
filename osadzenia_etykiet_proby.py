# -*- coding: utf-8 -*-
"""
    Modelowanie Semantyczne Etykiet:
        W podejściu semantycznym, jeśli model zostanie wytrenowany na osadzeniach słów (word embeddings), które reprezentują znaczenie słów i fraz, może być w stanie lepiej radzić sobie z nieznajomymi etykietami. Dzieje się tak, ponieważ osadzenia słów uchwycą kontekstowe znaczenie słów, a nie tylko ich tożsamość.
        Oznacza to, że jeśli w przyszłości pojawi się nowa etykieta o znaczeniu semantycznym zbliżonym do etykiet widzianych podczas treningu, model może być w stanie lepiej przewidzieć tę etykietę na podstawie podobieństwa semantycznego.

    Zero-Shot Learning:
        Innym podejściem może być zastosowanie technik "zero-shot learning", gdzie model jest trenowany w taki sposób, aby mógł generalizować i rozumieć kategorie, na których się nie trenował. Modele językowe jak BERT czy GPT-3 często są wykorzystywane w zadaniach "zero-shot", dzięki swojej zdolności do zrozumienia języka i kontekstu.
        W takim przypadku, model może być w stanie przypisać odpowiednią etykietę do tekstu, nawet jeśli ta etykieta nie była obecna w danych treningowych.

    Regularne Aktualizacje Modelu:
        Innym praktycznym podejściem jest regularne aktualizowanie modelu, włączając do treningu nowe etykiety, jak tylko staną się dostępne. To zapewnia, że model ma szansę nauczyć się nowych kategorii i poprawia jego zdolność do przewidywania na bieżąco.
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import logging
import pandas as pd
import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "hasła przedmiotowe"]):
    # Wczytanie danych z pliku JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)

    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']]

    # Konwersja wartości w kolumnie 'Tekst artykułu' na stringi
    df_json['Tekst artykułu'] = df_json['Tekst artykułu'].astype(str)

    # Wczytanie danych z pliku Excel
    df_excel = pd.read_excel(excel_file_path)

    # Połączenie DataFrame'ów
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")

    # Konwersja wartości w kolumnach 'Tytuł artykułu' i 'Tekst artykułu' na stringi w połączonym DataFrame
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)

    # Ograniczenie do wybranych kolumn i usunięcie wierszy z pustymi wartościami w 'hasła przedmiotowe'
    if 'hasła przedmiotowe' in merged_df.columns:
        selected_columns = merged_df[selected_columns_list]
        selected_columns = selected_columns.dropna(subset=['hasła przedmiotowe'])
        return selected_columns
    else:
        return pd.DataFrame(columns=selected_columns_list)


# Ścieżki do plików
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'
json_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afront_2022-09-08.json'
excel_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afront_2022-09-08.xlsx'
json_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/artpapier_2022-10-05.json'
excel_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/artpapier_2022-10-05.xlsx'
json_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/audycjekulturalne_2022-10-11.json'
excel_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/audycjekulturalne_2022-10-11.xlsx'
json_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/bylam_widzialam_2023-02-21.json'
excel_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/bylam_widzialam_2023-02-21.xlsx'
json_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/czas_kultury_2023-03-24.json'
excel_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/czas_kultury_2023-03-24.xlsx'
json_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_dziennik_2023-10-23.json'
excel_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_dziennik_2023-10-23.xlsx'
json_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/intimathule_2022-09-09.json'
excel_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/intimathule_2022-09-09.xlsx'
json_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/jerzy_sosnowski_2022-09-09.json'
excel_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jerzy_sosnowski_2022-09-09.xlsx'
json_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/komnen_kastamonu_2022-09-12.json'
excel_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/komnen_kastamonu_2022-09-12.xlsx'
json_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/krzysztof_jaworski_2022-12-08.json'
excel_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/krzysztof_jaworski_2022-12-08.xlsx'
json_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pgajda_2022-09-13.json'
excel_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pgajda_2022-09-13.xlsx'
json_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/poeci_po_godzinach_2022-09-14.json'
excel_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/poeci_po_godzinach_2022-09-14.xlsx'
# ... więcej plików w razie potrzeby

# Użycie funkcji
df1 = load_and_merge_data(json_file_path, excel_file_path)
df2 = load_and_merge_data(json_file_path2, excel_file_path2)
df3 = load_and_merge_data(json_file_path3, excel_file_path3)
df4 = load_and_merge_data(json_file_path4, excel_file_path4)
df5 = load_and_merge_data(json_file_path5, excel_file_path5)
df6 = load_and_merge_data(json_file_path6, excel_file_path6)
df7 = load_and_merge_data(json_file_path7, excel_file_path7)
df8 = load_and_merge_data(json_file_path8, excel_file_path8)
df9 = load_and_merge_data(json_file_path9, excel_file_path9)
df10 = load_and_merge_data(json_file_path10, excel_file_path10)
df11 = load_and_merge_data(json_file_path11, excel_file_path11)
df12 = load_and_merge_data(json_file_path12, excel_file_path12)
df13 = load_and_merge_data(json_file_path13, excel_file_path13)
df14 = load_and_merge_data(json_file_path14, excel_file_path14)
df15 = load_and_merge_data(json_file_path15, excel_file_path15)

# ... wczytanie kolejnych par plików

# Połączenie wszystkich DataFrame'ów
df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], ignore_index=True)

logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)
df_excel = pd.read_excel('C:/Users/dariu/Downloads/Mapowanie działów.xlsx')
df_excel['połączony dział'] = df_excel['nr działu'].astype(str) + " " + df_excel['nazwa działu']

mapowanie = pd.Series(df_excel['string uproszczony'].values, index=df_excel['połączony dział']).to_dict()

# Użycie mapowania do stworzenia nowej kolumny w df
df['rozwiniete_haslo'] = df['hasła przedmiotowe'].map(mapowanie)

df = df.dropna(subset=['rozwiniete_haslo'])
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

# Kodowanie etykiet
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("allegro/herbert-base-cased", output_hidden_states=True)

#unique_labels = df_excel['string uproszczony'].unique()
# label_embeddings = {}

# for label in unique_labels:
#     inputs = tokenizer(label, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     outputs = model(**inputs)
#     # Użyj przedostatniej warstwy ukrytej
#     hidden_states = outputs.hidden_states
#     label_embedding = hidden_states[-2].mean(dim=1).detach().numpy()  # przedostatnia warstwa
#     label_embeddings[label] = label_embedding
# import pickle

# Zakładając, że label_embeddings jest już obliczony
# with open('label_embeddings.pickle', 'wb') as handle:
#     pickle.dump(label_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)    
# import pickle

# with open('label_embeddings.pickle', 'rb') as handle:
#     label_embeddings = pickle.load(handle)

# # Sprawdzenie rozmiaru osadzenia etykiety
# labels_tensors_list = [torch.tensor(embedding).view(-1) for embedding in label_embeddings.values()]
#label_to_index = {label: idx for idx, label in enumerate(label_embeddings.keys())}
unique_labels = df['rozwiniete_haslo'].unique()
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

def tokenize_and_encode(examples):
    # Ustawienie maksymalnej długości sekwencji
    max_length = 512

    # Tokenizacja i kodowanie z paddingiem i obcięciem
    tokenized_inputs = tokenizer(
        examples['combined_text'], 
        padding='max_length', 
        truncation=True, 
        max_length=max_length
    )

    # Przypisanie indeksów etykiet
    labels = [label_to_index.get(label, -1) for label in examples['rozwiniete_haslo']]

    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': torch.tensor(labels)
    }

# Stosowanie funkcji tokenize_and_encode
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_encode, batched=True)

#%%
from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

# Definicja metryk
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted')
    }
sample_batch = next(iter(train_dataset))
input_ids = sample_batch['input_ids']
attention_mask = sample_batch['attention_mask']
labels = sample_batch['labels']

# Sprawdzenie rozmiaru tensorów
assert input_ids.shape[1] == 512, "Nieprawidłowy rozmiar tensora input_ids"
assert attention_mask.shape[1] == 512, "Nieprawidłowy rozmiar tensora attention_mask"
assert labels.shape[1] == len(label_embeddings), "Nieprawidłowy rozmiar tensora labels"

# Konfiguracja argumentów treningowych
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)


# Tworzenie obiektu Trenera
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)


# Trening modelu
trainer.train()

# Ewaluacja modelu
trainer.evaluate()



#%%









sample_data = {"combined_text": ["Przykładowy tekst 1", "Przykładowy tekst 2"]}
tokenize_and_encode(sample_data)
# Wydrukowanie przykładowych danych z przetworzonego zbioru
for i in range(5):  # Wybierz, ile przykładów chcesz wyświetlić
    print(f"Przykład {i}:")
    print("input_ids shape:", torch.tensor(dataset[i]['input_ids']).shape)
    print("attention_mask shape:", torch.tensor(dataset[i]['attention_mask']).shape)
    print("labels shape:", dataset[i]['labels'].shape)
    print("\n")

# Podział na zbiór treningowy i walidacyjny
train_test_dataset = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    'train': train_test_dataset['train'],
    'test': train_test_dataset['test']
})
import torch
from torch import nn
from transformers import AutoModel

class CustomModel(nn.Module):
    def __init__(self, model_name):
        super(CustomModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, input_ids, attention_mask, label_embeddings):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs[0][:, 0, :]  # Weź osadzenia dla tokenów [CLS] całego batcha
        return self.cos(text_embeddings, label_embeddings)


def custom_loss_fn(outputs, labels):
    # Tu zaimplementuj swoją logikę obliczania straty, na przykład:
    loss_fn = nn.MSELoss()
    return loss_fn(outputs, labels)

from torch.utils.data import DataLoader

# Tworzenie niestandardowego modelu
custom_model = CustomModel("allegro/herbert-base-cased")
custom_model.to('cpu') # device może być 'cuda' lub 'cpu'
# Generowanie przykładowego osadzenia tekstu dla weryfikacji
inputs = tokenizer("przykładowy tekst", return_tensors="pt")
text_embeddings = custom_model.text_model(**inputs)[0][:, 0, :]

# Sprawdzenie rozmiaru osadzenia tekstu
print(f"Rozmiar osadzenia tekstu: {text_embeddings.shape}")
# Przygotowanie DataLoadera
train_loader = DataLoader(dataset_dict['train'], batch_size=4, shuffle=True)
test_loader = DataLoader(dataset_dict['test'], batch_size=4, shuffle=False)

optimizer = torch.optim.AdamW(custom_model.parameters(), lr=5e-5)

# W pętli treningowej
for epoch in range(4):  # liczba epok
    custom_model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Poprawa kształtu tensorów
        input_ids = torch.stack(batch['input_ids']).t().to('cpu')
        attention_mask = torch.stack(batch['attention_mask']).t().to('cpu')
        label_embeddings = torch.stack([torch.tensor(label).view(-1) for label in batch['labels']]).to('cpu')
        
        print(f"input_ids shape: {input_ids.shape}")  # Powinno być [batch_size, sequence_length]
        print(f"attention_mask shape: {attention_mask.shape}")  # Powinno być [batch_size, sequence_length]
        print(f"label_embeddings shape: {label_embeddings.shape}")  # Powinno być [batch_size, 768]

        outputs = custom_model(input_ids, attention_mask, label_embeddings)
        loss = custom_loss_fn(outputs, label_embeddings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Total Loss: {total_loss}")

model_path = "model_hasla_4epoch"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
# Rozdzielenie haseł, mapowanie i ponowne połączenie

'''    Definicja Niestandardowego Modelu:
        Twój model musi być w stanie przyjąć dwa wejścia: osadzenie tekstu (zwrócone przez model językowy) oraz osadzenie etykiety.
        Musisz zdefiniować, jak model będzie porównywał te dwa wektory, aby dokonać klasyfikacji. Można to zrobić, na przykład, poprzez obliczenie podobieństwa kosinusowego między osadzeniem tekstu a osadzeniem etykiety.

    Modyfikacja Funkcji Strat (Loss Function):
        Zamiast używać standardowej funkcji straty, takiej jak cross-entropy, która jest używana w klasyfikacji kategorialnej, będziesz musiał zdefiniować własną funkcję straty, która może obsłużyć porównanie wektorów.

    Przygotowanie Danych Treningowych:
        Dane wejściowe do modelu powinny teraz zawierać zarówno osadzenia tekstu, jak i osadzenia etykiet.

    Trening Modelu:
        Proces treningu musi być dostosowany do nowej struktury danych i funkcji straty.'''

from transformers import AutoModelForSequenceClassification

# Załaduj model
model_name = "allegro/herbert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sprawdź rozmiar ostatniej warstwy ukrytej
embedding_size = model.classifier.in_features
print(f"Rozmiar osadzeń (embedding size): {embedding_size}")


#%% CALKOWICIE NOWA PROBA
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import logging
import pandas as pd
import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "hasła przedmiotowe"]):
    # Wczytanie danych z pliku JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)

    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']]

    # Konwersja wartości w kolumnie 'Tekst artykułu' na stringi
    df_json['Tekst artykułu'] = df_json['Tekst artykułu'].astype(str)

    # Wczytanie danych z pliku Excel
    df_excel = pd.read_excel(excel_file_path)

    # Połączenie DataFrame'ów
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")

    # Konwersja wartości w kolumnach 'Tytuł artykułu' i 'Tekst artykułu' na stringi w połączonym DataFrame
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)

    # Ograniczenie do wybranych kolumn i usunięcie wierszy z pustymi wartościami w 'hasła przedmiotowe'
    if 'hasła przedmiotowe' in merged_df.columns:
        selected_columns = merged_df[selected_columns_list]
        selected_columns = selected_columns.dropna(subset=['hasła przedmiotowe'])
        return selected_columns
    else:
        return pd.DataFrame(columns=selected_columns_list)


# Ścieżki do plików
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'
json_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afront_2022-09-08.json'
excel_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afront_2022-09-08.xlsx'
json_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/artpapier_2022-10-05.json'
excel_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/artpapier_2022-10-05.xlsx'
json_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/audycjekulturalne_2022-10-11.json'
excel_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/audycjekulturalne_2022-10-11.xlsx'
json_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/bylam_widzialam_2023-02-21.json'
excel_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/bylam_widzialam_2023-02-21.xlsx'
json_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/czas_kultury_2023-03-24.json'
excel_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/czas_kultury_2023-03-24.xlsx'
json_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_dziennik_2023-10-23.json'
excel_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_dziennik_2023-10-23.xlsx'
json_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/intimathule_2022-09-09.json'
excel_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/intimathule_2022-09-09.xlsx'
json_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/jerzy_sosnowski_2022-09-09.json'
excel_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jerzy_sosnowski_2022-09-09.xlsx'
json_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/komnen_kastamonu_2022-09-12.json'
excel_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/komnen_kastamonu_2022-09-12.xlsx'
json_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/krzysztof_jaworski_2022-12-08.json'
excel_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/krzysztof_jaworski_2022-12-08.xlsx'
json_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pgajda_2022-09-13.json'
excel_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pgajda_2022-09-13.xlsx'
json_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/poeci_po_godzinach_2022-09-14.json'
excel_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/poeci_po_godzinach_2022-09-14.xlsx'
# ... więcej plików w razie potrzeby

# Użycie funkcji
df1 = load_and_merge_data(json_file_path, excel_file_path)
df2 = load_and_merge_data(json_file_path2, excel_file_path2)
df3 = load_and_merge_data(json_file_path3, excel_file_path3)
df4 = load_and_merge_data(json_file_path4, excel_file_path4)
df5 = load_and_merge_data(json_file_path5, excel_file_path5)
df6 = load_and_merge_data(json_file_path6, excel_file_path6)
df7 = load_and_merge_data(json_file_path7, excel_file_path7)
df8 = load_and_merge_data(json_file_path8, excel_file_path8)
df9 = load_and_merge_data(json_file_path9, excel_file_path9)
df10 = load_and_merge_data(json_file_path10, excel_file_path10)
df11 = load_and_merge_data(json_file_path11, excel_file_path11)
df12 = load_and_merge_data(json_file_path12, excel_file_path12)
df13 = load_and_merge_data(json_file_path13, excel_file_path13)
df14 = load_and_merge_data(json_file_path14, excel_file_path14)
df15 = load_and_merge_data(json_file_path15, excel_file_path15)

# ... wczytanie kolejnych par plików

# Połączenie wszystkich DataFrame'ów
df = pd.concat([df1, df2, df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], ignore_index=True)

logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)
df_excel = pd.read_excel('C:/Users/dariu/Downloads/Mapowanie działów.xlsx')
df_excel['połączony dział'] = df_excel['nr działu'].astype(str) + " " + df_excel['nazwa działu']

mapowanie = pd.Series(df_excel['string uproszczony'].values, index=df_excel['połączony dział']).to_dict()

# Użycie mapowania do stworzenia nowej kolumny w df
df['rozwiniete_haslo'] = df['hasła przedmiotowe'].map(mapowanie)

df = df.dropna(subset=['rozwiniete_haslo'])
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

from transformers import AutoTokenizer

# Inicjalizacja toknizera
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

def tokenize_and_encode(examples):
    # Tokenizacja i kodowanie tekstu
    tokenized_inputs = tokenizer(
        examples['combined_text'], 
        padding='max_length', 
        truncation=True, 
        max_length=512
    )

    # Przypisanie etykiet tekstowych
    labels = examples['rozwiniete_haslo']

    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels
    }




from datasets import Dataset

# Konwersja DataFrame do formatu Dataset
dataset = Dataset.from_pandas(df)

# Zastosowanie funkcji tokenize_and_encode do dataset
dataset = dataset.map(tokenize_and_encode, batched=True)
from transformers import TrainingArguments, Trainer

# Definiowanie argumentów treningowych
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
)
num_labels = df_excel['string uproszczony'].nunique()
# Inicjalizacja modelu
model= AutoModelForSequenceClassification.from_pretrained("allegro/herbert-base-cased", num_labels=num_labels)

# Inicjalizacja Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=dataset,        
)

# Trenowanie modelu
trainer.train()


# Kodowanie etykiet
# tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
# model = AutoModelForSequenceClassification.from_pretrained("allegro/herbert-base-cased", output_hidden_states=True)


import torch
from transformers import AutoTokenizer, AutoModel, BertModel

# Inicjalizacja modelu i toknizera
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModel.from_pretrained("allegro/herbert-base-cased")

# def get_label_embedding(label):
#     inputs = tokenizer(label, return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     # Użyj ostatniej warstwy ukrytej
#     hidden_states = outputs.last_hidden_state
#     # Uśrednij po tokenach
#     return hidden_states.mean(dim=1)

# # Słownik dla osadzeń etykiet
# label_embeddings = {label: get_label_embedding(label).detach().numpy() for label in df_excel['string uproszczony'].unique()}
# # Zakładając, że label_embeddings jest już obliczony
# import pickle
# with open('label_embeddings.pickle', 'wb') as handle:
#     pickle.dump(label_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)    
import pickle

with open('label_embeddings.pickle', 'rb') as handle:
    label_embeddings = pickle.load(handle)
    
first_label_embedding = list(label_embeddings.values())[0]
print("Wymiary osadzenia dla pierwszej etykiety:", first_label_embedding.shape)
print("Typ danych osadzenia:", first_label_embedding.dtype)
class CustomModel(nn.Module):
    def __init__(self, label_embedding_size, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("allegro/herbert-base-cased")
        self.classifier = nn.Linear(self.bert.config.hidden_size + label_embedding_size, num_labels)

    def forward(self, input_ids, attention_mask, label_embeddings):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.pooler_output
        combined = torch.cat((sequence_output, label_embeddings), dim=1)
        logits = self.classifier(combined)
        return logits

# Utworzenie instancji modelu
liczba_klas = len(label_embeddings)  # Może być dostosowana do Twoich potrzeb
model = CustomModel(label_embedding_size=768, num_labels=liczba_klas)

# Funkcja do tokenizacji i kodowania danych
def tokenize_and_encode(examples):
    # Tokenizacja i kodowanie tekstu
    tokenized_inputs = tokenizer(
        examples['combined_text'], 
        padding='max_length', 
        truncation=True, 
        max_length=512
    )

    # Przypisanie osadzeń etykiet
    labels_embeddings = torch.stack([torch.tensor(label_embeddings[label]) for label in examples['rozwiniete_haslo']])

    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels_embeddings
    }

# Przygotowanie danych do treningu
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_encode, batched=True)




#%%


import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import logging
import pandas as pd
import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

def load_and_merge_data(json_file_path, excel_file_path, common_column='Link', selected_columns_list=['Tytuł artykułu', 'Tekst artykułu', "hasła przedmiotowe"]):
    # Wczytanie danych z pliku JSON
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    df_json = pd.DataFrame(json_data)

    # Ograniczenie DataFrame JSON do kolumn 'Link' i 'Tekst artykułu'
    df_json = df_json[['Link', 'Tekst artykułu']]

    # Konwersja wartości w kolumnie 'Tekst artykułu' na stringi
    df_json['Tekst artykułu'] = df_json['Tekst artykułu'].astype(str)

    # Wczytanie danych z pliku Excel
    df_excel = pd.read_excel(excel_file_path)

    # Połączenie DataFrame'ów
    merged_df = pd.merge(df_json, df_excel, on=common_column, how="inner")

    # Konwersja wartości w kolumnach 'Tytuł artykułu' i 'Tekst artykułu' na stringi w połączonym DataFrame
    merged_df['Tytuł artykułu'] = merged_df['Tytuł artykułu'].astype(str)
    merged_df['Tekst artykułu'] = merged_df['Tekst artykułu'].astype(str)

    # Ograniczenie do wybranych kolumn i usunięcie wierszy z pustymi wartościami w 'hasła przedmiotowe'
    if 'hasła przedmiotowe' in merged_df.columns:
        selected_columns = merged_df[selected_columns_list]
        selected_columns = selected_columns.dropna(subset=['hasła przedmiotowe'])
        return selected_columns
    else:
        return pd.DataFrame(columns=selected_columns_list)


# Ścieżki do plików
json_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/booklips_posts_2022-11-22.json'
excel_file_path = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/booklips_2022-11-22.xlsx'
json_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afisz_teatralny_2022-09-08.json'
excel_file_path2 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afisz_teatralny_2022-09-08.xlsx'
json_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pisarze_2023-01-27.json'
excel_file_path3 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pisarze_2023-01-27.xlsx'
json_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/afront_2022-09-08.json'
excel_file_path4 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/afront_2022-09-08.xlsx'
json_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/artpapier_2022-10-05.json'
excel_file_path5 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/artpapier_2022-10-05.xlsx'
json_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/audycjekulturalne_2022-10-11.json'
excel_file_path6 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/audycjekulturalne_2022-10-11.xlsx'
json_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/bylam_widzialam_2023-02-21.json'
excel_file_path7 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/bylam_widzialam_2023-02-21.xlsx'
json_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/czas_kultury_2023-03-24.json'
excel_file_path8 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/czas_kultury_2023-03-24.xlsx'
json_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/film_dziennik_2023-10-23.json'
excel_file_path9 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/film_dziennik_2023-10-23.xlsx'
json_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/intimathule_2022-09-09.json'
excel_file_path10 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/intimathule_2022-09-09.xlsx'
json_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/jerzy_sosnowski_2022-09-09.json'
excel_file_path11 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jerzy_sosnowski_2022-09-09.xlsx'
json_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/komnen_kastamonu_2022-09-12.json'
excel_file_path12 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/komnen_kastamonu_2022-09-12.xlsx'
json_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/krzysztof_jaworski_2022-12-08.json'
excel_file_path13 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/krzysztof_jaworski_2022-12-08.xlsx'
json_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/pgajda_2022-09-13.json'
excel_file_path14 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/pgajda_2022-09-13.xlsx'
json_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/jsony/poeci_po_godzinach_2022-09-14.json'
excel_file_path15 = 'D:/Nowa_praca/dane_model_jezykowy/drive-download-20231211T112144Z-001/poeci_po_godzinach_2022-09-14.xlsx'
# ... więcej plików w razie potrzeby

# # Użycie funkcji
df1 = load_and_merge_data(json_file_path, excel_file_path)
df2 = load_and_merge_data(json_file_path2, excel_file_path2)
df3 = load_and_merge_data(json_file_path3, excel_file_path3)
df4 = load_and_merge_data(json_file_path4, excel_file_path4)
df5 = load_and_merge_data(json_file_path5, excel_file_path5)
df6 = load_and_merge_data(json_file_path6, excel_file_path6)
df7 = load_and_merge_data(json_file_path7, excel_file_path7)
df8 = load_and_merge_data(json_file_path8, excel_file_path8)
df9 = load_and_merge_data(json_file_path9, excel_file_path9)
df10 = load_and_merge_data(json_file_path10, excel_file_path10)
df11 = load_and_merge_data(json_file_path11, excel_file_path11)
df12 = load_and_merge_data(json_file_path12, excel_file_path12)
df13 = load_and_merge_data(json_file_path13, excel_file_path13)
df14 = load_and_merge_data(json_file_path14, excel_file_path14)
df15 = load_and_merge_data(json_file_path15, excel_file_path15)
#Inicjalizacja toknizera

# Przygotowanie danych
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15], ignore_index=True)

# Logowanie dla transformers i datasets
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)
datasets_logger = logging.getLogger("datasets")
datasets_logger.setLevel(logging.INFO)

# Wczytanie mapowania etykiet
df_excel = pd.read_excel('C:/Users/dariu/Downloads/Mapowanie działów.xlsx')
df_excel['połączony dział'] = df_excel['nr działu'].astype(str) + " " + df_excel['nazwa działu']
mapowanie = pd.Series(df_excel['string uproszczony'].values, index=df_excel['połączony dział']).to_dict()

# Mapowanie etykiet i przygotowanie tekstu
df['rozwiniete_haslo'] = df['hasła przedmiotowe'].map(mapowanie)
df = df.dropna(subset=['rozwiniete_haslo'])
df['combined_text'] = df['Tytuł artykułu'] + " " + df['Tekst artykułu']

# Tokenizacja i kodowanie
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

def tokenize_and_encode(examples):
    tokenized_inputs = tokenizer(
        examples['combined_text'], 
        padding='max_length', 
        truncation=True, 
        max_length=512
    )
    labels = examples['rozwiniete_haslo']
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels
    }

# Konwersja DataFrame do formatu Dataset i zastosowanie tokenizacji
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_encode, batched=True)

# Podział na zestawy treningowe i testowe
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

# Definicja modelu
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-base-cased",
    num_labels=len(df['rozwiniete_haslo'].unique())
)

# Argumenty treningowe
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Inicjalizacja i trening modelu
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Ewaluacja modelu
trainer.evaluate()