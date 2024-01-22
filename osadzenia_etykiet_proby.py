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
model = AutoModelForSequenceClassification.from_pretrained(
    "allegro/herbert-base-cased"
)

# Generowanie osadzeń etykiet
unique_labels = df_excel['string uproszczony'].unique()
label_embeddings = {}
for label in unique_labels:
    inputs = tokenizer(label, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    label_embeddings[label] = outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Zastąpienie etykiet w df osadzeniami
df['label_embeddings'] = df['rozwiniete_haslo'].apply(lambda x: label_embeddings.get(x))

# Funkcja do tokenizacji i kodowania danych
from torch import tensor

def tokenize_and_encode(examples):
    # Tokenizacja tekstu
    tokenized_inputs = tokenizer(examples['combined_text'], padding='max_length', truncation=True, max_length=512)

    # Konwersja osadzeń etykiet na tensory PyTorch
    labels = [label_embeddings[label] for label in examples['rozwiniete_haslo']]
    labels = tensor(labels)

    # Zwrócenie słownika z tokenizowanym tekstem i osadzeniami etykiet
    return {'input_ids': tokenized_inputs['input_ids'], 'attention_mask': tokenized_inputs['attention_mask'], 'labels': labels}

# Mapowanie funkcji tokenizującej do datasetu
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_and_encode, batched=True)

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
        text_embeddings = self.text_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        return self.cos(text_embeddings, label_embeddings)

def custom_loss_fn(outputs, labels):
    # Tu zaimplementuj swoją logikę obliczania straty, na przykład:
    loss_fn = nn.MSELoss()
    return loss_fn(outputs, labels)

from torch.utils.data import DataLoader

# Tworzenie niestandardowego modelu
custom_model = CustomModel("allegro/herbert-base-cased")
custom_model.to('cpu') # device może być 'cuda' lub 'cpu'

# Przygotowanie DataLoadera
train_loader = DataLoader(dataset_dict['train'], batch_size=4, shuffle=True)
test_loader = DataLoader(dataset_dict['test'], batch_size=4, shuffle=False)

optimizer = torch.optim.AdamW(custom_model.parameters(), lr=5e-5)

for epoch in range(4):  # liczba epok
    custom_model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cpu')
        attention_mask = batch['attention_mask'].to('cpu')
        label_embeddings = batch['label_embeddings'].to('cpu')
        outputs = custom_model(input_ids, attention_mask, label_embeddings)
        loss = custom_loss_fn(outputs, label_embeddings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Total Loss: {total_loss}")
# Wyniki

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

