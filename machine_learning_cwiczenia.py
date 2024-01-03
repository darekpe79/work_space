# -*- coding: utf-8 -*-
"""
W projektach klasyfikacji sentymentu warto zacząć od prostszych modeli, a następnie stopniowo przechodzić do bardziej złożonych, jeśli jest to konieczne. Oto krótki przegląd kilku modeli, które mogą być użyte:
1. Naive Bayes

    Opis: To prosty model probabilistyczny, opierający się na założeniu niezależności cech (stąd nazwa "naive" - naiwny). Mimo swojej prostoty, często działa zaskakująco dobrze w zadaniach klasyfikacji tekstów.
    Zastosowanie: Idealny do pierwszych eksperymentów, zwłaszcza w klasyfikacji binarnej, takiej jak klasyfikacja sentymentu.
    Wady: Może nie radzić sobie dobrze z zależnościami między cechami i może być niewystarczający dla bardziej złożonych zadań.

2. Logistyczna Regresja

    Opis: Mimo nazwy, jest to model klasyfikacyjny, który używa funkcji logistycznej do modelowania prawdopodobieństwa przynależności do klasy.
    Zastosowanie: Bardzo popularny w problemach klasyfikacji binarnej. Jest łatwy w interpretacji i często stanowi dobry punkt odniesienia.
    Wady: Może mieć trudności z dobrze oddzielonymi lub wysoko wymiarowymi danymi.

3. Drzewa Decyzyjne

    Opis: Model, który dzieli dane na podzbiory na podstawie wartości cech, tworząc strukturę drzewiastą decyzji.
    Zastosowanie: Użyteczne, gdy chcesz łatwą wizualizację i interpretację modelu.
    Wady: Podatne na przetrenowanie (overfitting), zwłaszcza z złożonymi danymi.

4. Random Forest (Las Losowy)

    Opis: Metoda zespołowa, która tworzy wiele drzew decyzyjnych i wykorzystuje średnią ich prognoz.
    Zastosowanie: Bardziej zaawansowana niż pojedyncze drzewo decyzyjne, często zapewnia lepszą wydajność i stabilność.
    Wady: Większa złożoność modelu i trudniejsza interpretacja niż pojedyncze drzewo decyzyjne.

5. Support Vector Machines (SVM)

    Opis: Model, który próbuje znaleźć hiperpłaszczyznę w przestrzeni cech, która najlepiej oddziela klasy.
    Zastosowanie: Skuteczny w wysoko wymiarowych przestrzeniach, takich jak przestrzeń tekstowa.
    Wady: Może być trudniejszy w optymalizacji i interpretacji. Wymaga również wyboru funkcji jądra.

6. Sieci Neuronowe / Głębokie Uczenie

    Opis: Bardziej zaawansowane modele, które potrafią modelować skomplikowane zależności i interakcje między cechami.
    Zastosowanie: Użyteczne w bardzo złożonych zadaniach klasyfikacyjnych, mogą dawać znakomite wyniki.
    Wady: Znacznie bardziej złożone w konfiguracji, wymagają więcej danych i czasu do trenowania.
"""

import os
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

def wczytaj_dane(sciezka):
    recenzje = []
    sentymenty = []

    for etykieta in ["pos", "neg"]:
        sciezka_katalogu = os.path.join(sciezka, etykieta)
        for plik in os.listdir(sciezka_katalogu):
            if plik.endswith('.txt'):
                sciezka_pliku = os.path.join(sciezka_katalogu, plik)
                with open(sciezka_pliku, 'r', encoding='utf-8') as plik:
                    tekst = plik.read()
                    recenzje.append(tekst)
                    sentymenty.append(1 if etykieta == "pos" else 0)

    return pd.DataFrame({
        'recenzja': recenzje,
        'sentyment': sentymenty
    })
sciezka_do_danych = r'C:/Users/dariu/aclImdb/train'  # Zaktualizuj, jeśli potrzeba

# Wczytanie danych treningowych
df_train = wczytaj_dane(sciezka_do_danych)

# Wyświetl pierwsze wiersze ramki danych
print(df_train.head())

# Pobranie listy słów nieznaczących (stop words)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Usunięcie znaków specjalnych
    text = re.sub(r'\W', ' ', text)
    # Usunięcie wszystkich pojedynczych znaków
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Zamiana wielokrotnych spacji na pojedynczą spację
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Konwersja na małe litery
    text = text.lower()
    # Usunięcie stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text
# Ścieżka do zbioru danych treningowych

df_train['recenzja'] = df_train['recenzja'].apply(clean_text)

#%% Wektoryzacja TfiDf

from sklearn.feature_extraction.text import TfidfVectorizer

# Inicjalizacja wektoryzatora TF-IDF
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)

# Wektoryzacja tekstu recenzji
X = tfidfconverter.fit_transform(df_train['recenzja']).toarray()

# Y to nasze etykiety sentymentu
Y = df_train['sentyment']
#%% Wektryzacja word2vec
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np

# Wczytanie wytrenowanego modelu Word2Vec (na przykład Google News vectors)
word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/dariu/aclImdb/GoogleNews-vectors-negative300(1).bin/GoogleNews-vectors-negative300.bin', binary=True)

def document_vector(word2vec_model, doc):
    # Usunięcie słów, których nie ma w słowniku
    words = [word for word in word_tokenize(doc) if word in word2vec_model.key_to_index]
    if len(words) == 0:
       return np.zeros(word2vec_model.vector_size)
    # Uśrednianie wektorów dla słów w dokumencie
    return np.mean(word2vec_model[words], axis=0)

# Wektoryzacja tekstu recenzji
X = np.array([document_vector(word2vec_model, doc) for doc in df_train['recenzja']])
Y = df_train['sentyment'].values

#%%model Naive Bayes, - działa z tfidf
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Trenowanie modelu
model = MultinomialNB()
model.fit(X, Y)

# Wczytanie i przetwarzanie danych testowych
# Uwaga: Użyj odpowiednich funkcji wczytywania i czyszczenia danych, podobnie jak dla danych treningowych
df_test = wczytaj_dane('C:/Users/dariu/aclImdb/test/')
# Czyszczenie tekstu
df_test['recenzja'] = df_test['recenzja'].apply(clean_text)

# Wektoryzacja tekstu
X_test = tfidfconverter.transform(df_test['recenzja']).toarray()
Y_test = df_test['sentyment']
# Przewidywanie sentymentu na danych testowych
predictions = model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

# Ewaluacja modelu
print(classification_report(Y_test, predictions))
print("Dokładność:", accuracy_score(Y_test, predictions))
#%% logistyczna regresja dziala z word2vec

from sklearn.linear_model import LogisticRegression

# Utworzenie nowej instancji modelu regresji logistycznej
model = LogisticRegression()

# Trenowanie modelu na danych treningowych (X, Y) z wektorami Word2Vec
model.fit(X, Y)  # Upewnij się, że X ma 300 cech (dla Word2Vec)

# Używanie modelu do przewidywania i oceny na danych testowych
# Upewnij się, że dane testowe X_test również są wektoryzowane za pomocą Word2Vec i mają 300 cech
#word2vec_model = KeyedVectors.load_word2vec_format('C:/Users/dariu/aclImdb/GoogleNews-vectors-negative300(1).bin/GoogleNews-vectors-negative300.bin', binary=True)




# Przygotowanie danych testowych
df_test = wczytaj_dane('C:/Users/dariu/aclImdb/test/')
df_test['recenzja'] = df_test['recenzja'].apply(clean_text)
X_test = np.array([document_vector(word2vec_model, doc) for doc in df_test['recenzja']])
Y_test = df_test['sentyment'].values

# Przewidywanie i ocena na danych testowych
predictions = model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

# Ewaluacja modelu
print(classification_report(Y_test, predictions))
print("Dokładność:", accuracy_score(Y_test, predictions))

from joblib import dump

# Zapisz model do pliku
dump(model, 'model_regresji_logistycznej.joblib')
from joblib import load

# Wczytaj model z pliku
model = load('model_regresji_logistycznej.joblib')

# Użyj modelu do przewidywań
predictions = model.predict(X_test)
#%% drzewa decyzyjne działa z word2vec
from sklearn.tree import DecisionTreeClassifier

# Trenowanie drzewa decyzyjnego
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X, Y)  # X i Y z wcześniejszej wektoryzacji Word2Vec
#ocena modelu
df_test = wczytaj_dane('C:/Users/dariu/aclImdb/test/')
df_test['recenzja'] = df_test['recenzja'].apply(clean_text)
X_test = np.array([document_vector(word2vec_model, doc) for doc in df_test['recenzja']])
Y_test = df_test['sentyment'].values

# Przewidywanie i ocena na danych testowych
predictions = decision_tree_model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

# Ewaluacja modelu
print(classification_report(Y_test, predictions))
print("Dokładność:", accuracy_score(Y_test, predictions))

#%% Random forrest

from sklearn.ensemble import RandomForestClassifier

# Trenowanie modelu Random Forest
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X, Y) 

#ocena modelu
df_test = wczytaj_dane('C:/Users/dariu/aclImdb/test/')
df_test['recenzja'] = df_test['recenzja'].apply(clean_text)
X_test = np.array([document_vector(word2vec_model, doc) for doc in df_test['recenzja']])
Y_test = df_test['sentyment'].values
# Przewidywanie na danych testowych
predictions = random_forest_model.predict(X_test)

# Ewaluacja modelu
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(Y_test, predictions))
print("Dokładność:", accuracy_score(Y_test, predictions))

#%% support vector machines
from sklearn.svm import SVC

# Trenowanie modelu SVM
svm_model = SVC()
svm_model.fit(X, Y)  # X i Y z wektoryzacji Word2Vec

df_test = wczytaj_dane('C:/Users/dariu/aclImdb/test/')
df_test['recenzja'] = df_test['recenzja'].apply(clean_text)
X_test = np.array([document_vector(word2vec_model, doc) for doc in df_test['recenzja']])
Y_test = df_test['sentyment'].values
# Przewidywanie na danych testowych
predictions = svm_model.predict(X_test)

# Ewaluacja modelu
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(Y_test, predictions))
print("Dokładność:", accuracy_score(Y_test, predictions))


# Stwórz DataFrame
wyniki_df = pd.DataFrame({
    'Recenzja': df_test['recenzja'],
    'Przewidziany Sentyment': predictions
})

# Zapisz do Excela
wyniki_df.to_excel('wyniki_sentymentu.xlsx', index=False)

from joblib import dump

# Zapisz model do pliku
dump(svm_model, 'svm_model.joblib')
from joblib import load

# Wczytaj model z pliku
svm_model = load('svm_model.joblib')

# Użyj modelu do przewidywań
predictions = svm_model.predict(X_test)

#%% sieci neuronowe Tensor flow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inicjalizacja tokenizera
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df_train['recenzja'])

# Konwersja tekstów na sekwencje
X = tokenizer.texts_to_sequences(df_train['recenzja'])

# Padding sekwencji, aby miały tę samą długość
X = pad_sequences(X, maxlen=100)
Y = df_train['sentyment'].values

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Zdefiniowanie modelu
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))  # Warstwa osadzeń
model.add(Flatten())  # Spłaszczenie danych
model.add(Dense(1, activation='sigmoid'))  # Warstwa wyjściowa

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(X, Y, epochs=5, batch_size=32, validation_split=0.2)

df_test = wczytaj_dane('C:/Users/dariu/aclImdb/test/')
df_test['recenzja'] = df_test['recenzja'].apply(clean_text)

 # Użyj tej samej długości sekwencji co w treningu

 

model.save('moj_model.keras')

from tensorflow.keras.models import load_model
model = load_model('moj_model.keras')

# Użyj modelu do przewidywań lub dalszego trenowania
predictions = model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

# Ocena modelu
predictions = [round(pred[0]) for pred in predictions]
print(classification_report(Y_test, predictions))
print("Dokładność:", accuracy_score(Y_test, predictions))


# Zdefiniowanie modelu z większą iloscia warstw ukrytych
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
model.add(Dropout(0.5))  # Dodaj warstwę Dropout dla regularyzacji, aby zapobiec przetrenowaniu.
model.add(Conv1D(64, 5, activation='relu'))  # Warstwa konwolucyjna, która jest często używana w przetwarzaniu obrazów, ale może być również skuteczna w przetwarzaniu tekstu.
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))  # Warstwa LSTM to warstwa rekurencyjna, która jest dobrym wyborem dla danych sekwencyjnych, takich jak tekst.
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=5, batch_size=32, validation_split=0.2)

#test
df_test = wczytaj_dane('C:/Users/dariu/aclImdb/test/')
df_test['recenzja'] = df_test['recenzja'].apply(clean_text)
X_test = tokenizer.texts_to_sequences(df_test['recenzja'])
X_test = pad_sequences(X_test, maxlen=100)
  # Użyj tej samej długości sekwencji co w treningu
predictions = model.predict(X_test)
Y_test = df_test['sentyment'].values
from sklearn.metrics import classification_report, accuracy_score

# Ocena modelu
predictions = [round(pred[0]) for pred in predictions]
print(classification_report(Y_test, predictions))
print("Dokładność:", accuracy_score(Y_test, predictions))

'''Różne warstwy w modelach sieci neuronowych mają specyficzne role i są używane w różnych rodzajach architektur, zarówno w TensorFlow, jak i PyTorch:

    Warstwy Gęste (Dense Layers): Standardowe warstwy neuronów, gdzie każdy neuron jest połączony ze wszystkimi neuronami w poprzedniej warstwie.

    Warstwy Konwolucyjne (Convolutional Layers): Przetwarzają dane wejściowe w małych, nakładających się oknach, użyteczne w analizie obrazów i czasem w przetwarzaniu tekstu.

    Warstwy Rekurencyjne (RNN, LSTM, GRU): Pozwalają na przetwarzanie danych sekwencyjnych, przechowując informacje o poprzednich stanach.

    Warstwy Dropout: Pomagają zapobiegać przetrenowaniu przez losowe wyłączanie neuronów podczas treningu.

    Warstwy Osadzające (Embedding Layers): Używane do tworzenia gęstych reprezentacji wektorowych dla danych kategorialnych, szczególnie w przetwarzaniu języka naturalnego.

Podobne warstwy i koncepcje znajdują zastosowanie w różnych frameworkach uczenia maszynowego, w tym w PyTorch, który również pozwala na budowanie różnorodnych architektur sieci neuronowych za pomocą modularnych i elastycznych komponentów'''