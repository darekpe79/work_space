# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:01:53 2024

@author: dariu
"""
#%%Spacy
import spacy

# Ładowanie polskiego modelu
nlp = spacy.load("pl_core_news_sm")

# Przykładowy tekst w języku polskim
text = "Nazywam się Grzegorz Brzęszczyszczykiewicz, pochodzę z Chrząszczyżewoszczyc, pracuję w Łękołodzkim Urzędzie Powiatowym"
text = "Mark Zuckerberg założył Facebooka w 2004 roku. Microsoft został założony przez Billa Gatesa i Paula Allena. Elon Musk urodził się w RPA, ale teraz mieszka w USA."
# Przetwarzanie tekstu
doc = nlp(text)

# Wypisanie wykrytych nazw własnych i ich kategorii
for entity in doc.ents:
    print(entity.text, entity.label_)
    
    
    
import spacy
from spacy.pipeline import EntityRuler

# Ładowanie modelu językowego
nlp = spacy.load("pl_core_news_sm")  # Zmień na 'en_core_web_sm' dla angielskiego

# Tworzenie nowego EntityRuler


# Dodawanie wzorców
ruler = nlp.add_pipe("entity_ruler")

# Dodawanie wzorców
patterns = [
    {"label": "WORK_OF_ART", "pattern": "Zamek"},
    # Dodaj więcej wzorców dla innych znanych tytułów dzieł sztuki
]

# Dodanie wzorców do EntityRuler
ruler.add_patterns(patterns)

# Tekst do analizy
text = """Zamek, REŻ. FRANCISZEK SZUMIŃSKI ..."""

# Przetwarzanie tekstu
doc = nlp(text)

# Wyświetlanie wykrytych nazw własnych
for ent in doc.ents:
    print(ent.text, ent.label_)

    
import morfeusz2

# Inicjalizacja Morfeusza
morf = morfeusz2.Morfeusz()

# Przykładowe słowo
slowo = "kotek"

# Generowanie odmian słowa
analiza = morf.analyse(slowo)
for interp in analiza:
    # Pobranie informacji o odmianie
    info = interp[2]
    orth = info[0]  # forma odmieniona
    lemma = info[1]  # lemat
    tag = info[2]    # tag gramatyczny

    print(f"Odmiana: {orth}, Lemma: {lemma}, Tag: {tag}")

import morfeusz2

# Inicjalizacja Morfeusza
morf = morfeusz2.Morfeusz()

# Przykładowe słowo
slowo = "Warszawa"

# Generowanie odmian słowa przez przypadki
analiza = morf.analyse(slowo)
odmiany = set()  # Zbiór do przechowywania unikalnych odmian

for interp in analiza:
    info = interp[2]
    forma_odmieniona = info[0]
    tag = info[2]

    # Filtracja odmian przez przypadki
    # Szukamy rzeczownika (subst) w liczbie pojedynczej (sg)
    if "subst" in tag and "sg" in tag:
        odmiany.add(forma_odmieniona)

# Wyświetlenie odmian
print("Odmiany słowa 'kot' przez przypadki:")
for odmiana in odmiany:
    print(odmiana)

    
import spacy
from spacy.training import Example

# Załaduj istniejący model
nlp = spacy.load("pl_core_news_sm")

# Przykładowe dane treningowe: tekst oraz oznaczenia (start, end, label)
train_data = [
    ("Twitter is a great platform", {"entities": [(0, 7, "PLATFORM")]}),
    ("I just bought a Tesla", {"entities": [(18, 23, "BRAND")]}),
    # Dodaj więcej przykładów
]

# Aktualizacja modelu
ner = nlp.get_pipe("ner")
for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Rozpoczęcie treningu
nlp.disable_pipes(['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])  # Wyłączenie niepotrzebnych komponentów
optimizer = nlp.resume_training()
for itn in range(100):  # Liczba iteracji
    random.shuffle(train_data)
    losses = {}
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(losses)

# Testowanie modelu
test_text = "I love using Twitter"
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.label_)

    
#%%Transformers    
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Ładowanie tokienizera i modelu
tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
model = AutoModelForTokenClassification.from_pretrained("dkleczek/bert-base-polish-uncased-v1")

# Tworzenie pipeline NER
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Przykładowy tekst
text = "Jan Kowalski mieszka w Warszawie. On uwielbia podróżować do Krakowa i Gdańska."

# Wykrywanie nazw własnych
ner_results = ner_pipeline(text)

# Wypisywanie wyników
for entity in ner_results:
    print(entity)
    
#%% Stanza    
import stanza

# Pobieranie i instalowanie polskiego modelu
stanza.download('pl')

# Inicjalizacja pipeline Stanza
nlp = stanza.Pipeline(lang='pl', processors='tokenize,ner')

# Przykładowy tekst
text = "Jan Kowalski mieszka w Warszawie. On uwielbia podróżować do Krakowa i Gdańska."

# Przetwarzanie tekstu
doc = nlp(text)

# Wypisanie wykrytych nazw własnych
for sentence in doc.sentences:
    for entity in sentence.ents:
        print(f"Tekst: {entity.text}, Typ: {entity.type}")
    
#%%Flair 
from flair.data import Sentence
from flair.models import SequenceTagger

# Wczytanie pre-trenowanego modelu NER (wielojęzycznego)
tagger = SequenceTagger.load('flair/ner-multi')

# Przykładowy tekst
text = "Sherlock Holmes i doktor Watson w nowej kampanii reklamowej OLX Klasyczni bohaterowie stworzeni przez Arthura Conana Doyle’a pojawili się w nowej kampanii reklamowej serwisu ogłoszeniowego OLX. Sherlock Holmes i doktor Watson mają w zrealizowanych filmikach humorystycznych ostrzegać, przypominać i tłumaczyć, jak nie dać się oszukać w sieci. Seria krótkich filmów w zabawny sposób gra z wizerunkiem postaci stworzonych przez Arthura Conana Doyle’a. Jacek Borusiński z Mumio wciela się w cokolwiek średnio błyskotliwego Sherlocka, natomiast Sebastian Stankiewicz partneruje mu w roli Watsona i w tej wersji słynnego duetu to raczej on nosi głowę na karku. Ich detektywistyczne biuro odwiedzają współcześni bohaterowie padający ofiarą internetowych łotrów. „Bezpieczeństwo naszych użytkowników jest dla nas niezwykle ważne, dlatego regularnie wracamy do tego tematu w kolejnych kampaniach wideo. To już nasza piąta aktywacja poświęcona tej tematyce, a jednocześnie swego rodzaju powrót do korzeni, ponieważ OLX.pl, na bardzo wczesnym etapie swojego istnienia, dostępny był pod adresem Szerlok.pl” – mówi Michał Wiśniewski z OLX. Za kreację kampanii odpowiada agencja PZL. „Kradzieże tożsamości, wyłudzanie danych czy phishing to obrzydliwe przestępstwa. Jesteśmy w PZL przekonani, że poruszanie tych tematów śmiertelnie poważnie mogłoby podziałać na odbiorców wręcz odpychająco. Jak więc podać treść ostrzeżeń bez oceniania i dydaktyzmu? Zdecydowaliśmy się na komediową formę, mając nadzieję, że tak podane treści dotrą do odbiorców i zapobiegną w jakiejś mierze szerzeniu się e-zła” – mówi Sławomir Szczęśniak z PZL, współautor koncepcji kreatywnej. Produkcję klipów powierzono firmie Lucky Luciano Pictures. W ramach kampanii powstały cztery spoty, każdy poświęcony innemu oszustwu w sieci. Dwa z nich zostały już udostępnione w sieci. Kampania jest dostępna w mediach społecznościowych marki OLX. "

# Tworzenie obiektu Sentence i przetwarzanie tekstu
sentence = Sentence(text)
tagger.predict(sentence)

# Wypisanie wykrytych nazw własnych
for entity in sentence.get_spans('ner'):
    print(entity)

text = "Mark Zuckerberg założył Facebooka w 2004 roku. Microsoft został założony przez Billa Gatesa i Paula Allena. Elon Musk urodził się w RPA, ale teraz mieszka w USA."

# Tworzenie obiektu Sentence i przetwarzanie tekstu
sentence = Sentence(text)
tagger.predict(sentence)

# Wypisanie wykrytych nazw własnych
for entity in sentence.get_spans('ner'):
    print(f"Tekst: {entity.text}, Typ: {entity.tag}")
    
#%%    
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Nazywam się Grzegorz Brzęszczyszczykiewicz, pochodzę "\
    "z Chrząszczyżewoszczyc, pracuję w Łękołodzkim Urzędzie Powiatowym"
example= """OSIEMNASTKA, REŻ. NATALIA FIJEWSKA-ZDANOWSKA
Pięciu bohaterów, którzy na widok publiczny wyciągną swoje problemy. Pięciu bohaterów, którzy zmuszeni zostaną do znalezienia przyczyn nieudanego życia. I niech nikogo nie zmyli tytuł tego spektaklu, bo nie o imprezie osiemnastkowej będziemy tutaj rozmawiać, a o bohaterach którzy robić będą dobrą minę do złej gry. Zabawa urodzinowa jest tylko pretekstem do tego, aby zamienić się w rozmowy o egzystencji i niespełnionych pragnieniach
/fot. Rafał Latoszek/

Dla jednych będzie to przewidywalny początek, dla innych dobre wprowadzenie do rozgrywającej się akcji. Bo zaczynamy zwykłą rozmową pomiędzy trzema parami, które spotykają się przypadkiem na wspólnej imprezie osiemnastkowej swoich dzieci. Jest miło i przyjemnie, dialogi bawią, nietrudno też zaśmiać się z tych wszystkich żartów sytuacyjnych które się zadziewają. Lecz trwające coraz dłużej rozmowy zaczynają gęstnieć, a w powietrzu unosi się dziwny zapach żalu i goryczy, który przeżywają bohaterowie. Akcja przyśpieszy kiedy okaże się, że młodzież która tak świetnie bawić się miała na zorganizowanej imprezie zniknęła, a pilnujący ich rodzice obiecali do rana nie wychodzić z domu, który mieści się gdzieś na skraju lasu. Powiało grozą - nie będę ukrywać. Pojawiła się też cała masa pytań, gdzie zniknęły dzieci, dlaczego rodzice nie mogą opuścić budynku, co wydarzy się dalej, kto zginie. Napięcie zaczyna narastać, a pytania zamiast znikać, tylko się mnożyły. Byłam dobrej myśli do tego co zadzieje się dalej, bo oprócz momentów grozy mieliśmy również dobry humor, który trzymał wysoki poziom.

Aż tu nagle… Szok i niedowierzanie, bo spektakl zwolnił do maksimum, sprawy zaczęły zamiast się rozwiązywać nawarstwiać, pojawiło się całe mnóstwo wstawek o Janie Chrzcicielu i zapewne cytatów z Biblii, które ani nie są mi bliskie ani do tego dzieła nie pasowały. A szkoda, bo zamysł Natalii Fijewskiej-Zdanowskiej (scenariusz i reżysera), która głos oddaje ludziom dojrzałym z dość dużym już bagażem doświadczeń był naprawdę dobry. Bo co tu dużo mówić stykamy się z ludźmi, których marzenia pozostają w sferze zapomnienia, a najskrytsze pragnienia nigdy nie zostają wypowiedziane na głos. Reżyserka oddaje swój spektakl w ręce pokolenia niespełnionego, ludzi lat 70. i 80., które wychowało się na lęku i toksycznych relacjach. I to ich przeżycia powinny nas-widzów uwierać czy też wzbudzać jakąś refleksję. I zgodzę się, że na początku tak się dzieje, ale później akcja staje w miejscu problemy się nie rozwiązują, a dziejące się wydarzenia nagle znajdują ujście, bez większego ładu i składu.

Mówiąc o spektaklu Osiemnastka nie można jednak zapomnieć o aktorach, którzy byli bardzo przekonujący i prawdziwi w swoich rolach. To było tak dobre połączenie różnych charakterów i osobowości, że momentami na scenie działy się rzeczy piękne. Na tyle, że nie sposób było nie oddać się w ręce bohaterów i dać im po prostu prowadzić się po niuansach tego dzieła. I na próżno dzielić tutaj obsadę na obóz lepiej i gorzej grający, bo to co pokazali na scenie mogło być tylko w tej pierwszej z grup. Naprawdę to był kawał solidnej pracy, doświadczenia które dobrze skomponowało się z opowiadaną historią postaci. I żałuję (bardzo), że koniec tego dzieła, mimo fantastycznej gry aktorskiej, był mało przekonywująco (i słaby). Ale wybrzmieć to musi głośno, bo jakby odrzucić to co złe, to dla Zuzanny Fijewskiej-Malesza, Agaty Kołodziejczyk, Dominiki Łakomskiej, Jarosława Boberka i Pawła Pabisiaka przyszła bym jeszcze raz chętnie na taką imprezę. byty={"osoba":"Fijewska-Zdanowska, Natalia", "instytucja":"Teatr Rampa na Targówku"}"""
ner_results = nlp(example)
print(ner_results)


# Funkcja do łączenia tokenów w pełne jednostki


def combine_tokens(ner_results):
    combined_entities = []
    current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0, "start": None, "end": None}
    previous_end_of_word = False

    for token in ner_results:
        # Sprawdzamy, czy bieżący token jest końcem słowa
        end_of_word = "</w>" in token['word']
        cleaned_word = token['word'].replace("</w>", "")

        # Rozpoczęcie nowej jednostki
        if token['entity'].startswith('B-'):
            if current_entity['word']:
                # Obliczamy średnią ocenę dla skompletowanej jednostki
                current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
                combined_entities.append(current_entity)
            current_entity = {"word": cleaned_word, "type": token['entity'][2:], "score_sum": token['score'],
                              "token_count": 1, "start": token['start'], "end": token['end']}
            previous_end_of_word = end_of_word
        # Kontynuacja obecnej jednostki
        elif token['entity'].startswith('I-') and current_entity['type'] == token['entity'][2:]:
            # Dodajemy spację przed bieżącym tokenem, jeśli poprzedni token był końcem słowa
            if previous_end_of_word:
                current_entity['word'] += " " + cleaned_word
            else:
                current_entity['word'] += cleaned_word
            current_entity['end'] = token['end']
            current_entity['score_sum'] += token['score']
            current_entity['token_count'] += 1
            previous_end_of_word = end_of_word

    # Dodajemy ostatnią jednostkę, jeśli istnieje
    if current_entity['word']:
        current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
        combined_entities.append(current_entity)

    return combined_entities

# Wywołanie funkcji
combined_entities = combine_tokens


# Wywołanie funkcji
combined_entities = combine_tokens(ner_results)

# Wyświetlanie wyników
for entity in combined_entities:
    print(entity)
    
    
    
import re

def oznacz_byty(tekst, lista_bytow):
    for byt, typ in lista_bytow:
        pattern = re.compile(re.escape(byt), re.IGNORECASE)
        start = 0
        for match in pattern.finditer(tekst):
            start_idx = match.start()
            end_idx = match.end()
            if start_idx != start:  # Sprawdza, czy to nie jest kontynuacja poprzedniego bytu
                tekst = tekst[:start_idx] + f"[B-{typ}]" + tekst[start_idx:end_idx] + f"[/B-{typ}]" + tekst[end_idx:]
                start = end_idx + len(f"[B-{typ}]") + len(f"[/B-{typ}]")
            else:
                tekst = tekst[:start_idx] + f"[I-{typ}]" + tekst[start_idx:end_idx] + f"[/I-{typ}]" + tekst[end_idx:]
                start = end_idx + len(f"[I-{typ}]") + len(f"[/I-{typ}]")
    return tekst

# Przykładowe użycie
lista_bytow = [("J.R.R. Tolkien", "OSOBA"), ("Warszawa", "MIEJSCOWOŚĆ")]
tekst = "J.R.R. Tolkien i czarodziej Gandalf otrzymali swoje ulice na warszawskim Mordorze."
tekst_z_anotacjami = oznacz_byty(tekst, lista_bytow)
print(tekst_z_anotacjami)



import spacy
import re

# Ładowanie modelu języka polskiego
nlp = spacy.load('pl_core_news_sm')

# Przykładowy tekst
text = "OSIEMNASTKA, REŻ. NATALIA FIJEWSKA-ZDANOWSKA Pięciu bohaterów, którzy na widok publiczny wyciągną swoje problemy."

# Lista bytów do oznaczenia
entities = {
    "NATALIA FIJEWSKA-ZDANOWSKA": "osoba",
    "Teatr Rampa na Targówku": "instytucja"
}

# Funkcja do lematyzacji
def lemmatize(text):
    return " ".join([token.lemma_ for token in nlp(text.lower())])

# Lematyzacja bytów
lemmatized_entities = {lemmatize(entity): label for entity, label in entities.items()}

# Lematyzacja tekstu
lemmatized_text = lemmatize(text)

# Funkcja do oznaczania bytów w tekście
def annotate_text(text, entities):
    for entity, label in entities.items():
        text = re.sub(r'\b' + re.escape(entity) + r'\b', f"[{entity}]({label})", text)
    return text

# Oznaczanie bytów w tekście
annotated_text = annotate_text(lemmatized_text, lemmatized_entities)
print(annotated_text)

import spacy
from collections import Counter

# Załaduj model językowy (np. angielski)
nlp = spacy.load("pl_core_news_lg")

# Przykładowy tekst
text = """OSIEMNASTKA, REŻ. NATALIA FIJEWSKA-ZDANOWSKA
Pięciu bohaterów, którzy na widok publiczny wyciągną swoje problemy. Pięciu bohaterów, którzy zmuszeni zostaną do znalezienia przyczyn nieudanego życia. I niech nikogo nie zmyli tytuł tego spektaklu, bo nie o imprezie osiemnastkowej będziemy tutaj rozmawiać, a o bohaterach którzy robić będą dobrą minę do złej gry. Zabawa urodzinowa jest tylko pretekstem do tego, aby zamienić się w rozmowy o egzystencji i niespełnionych pragnieniach
/fot. Rafał Latoszek/

Dla jednych będzie to przewidywalny początek, dla innych dobre wprowadzenie do rozgrywającej się akcji. Bo zaczynamy zwykłą rozmową pomiędzy trzema parami, które spotykają się przypadkiem na wspólnej imprezie osiemnastkowej swoich dzieci. Jest miło i przyjemnie, dialogi bawią, nietrudno też zaśmiać się z tych wszystkich żartów sytuacyjnych które się zadziewają. Lecz trwające coraz dłużej rozmowy zaczynają gęstnieć, a w powietrzu unosi się dziwny zapach żalu i goryczy, który przeżywają bohaterowie. Akcja przyśpieszy kiedy okaże się, że młodzież która tak świetnie bawić się miała na zorganizowanej imprezie zniknęła, a pilnujący ich rodzice obiecali do rana nie wychodzić z domu, który mieści się gdzieś na skraju lasu. Powiało grozą - nie będę ukrywać. Pojawiła się też cała masa pytań, gdzie zniknęły dzieci, dlaczego rodzice nie mogą opuścić budynku, co wydarzy się dalej, kto zginie. Napięcie zaczyna narastać, a pytania zamiast znikać, tylko się mnożyły. Byłam dobrej myśli do tego co zadzieje się dalej, bo oprócz momentów grozy mieliśmy również dobry humor, który trzymał wysoki poziom.

Aż tu nagle… Szok i niedowierzanie, bo spektakl zwolnił do maksimum, sprawy zaczęły zamiast się rozwiązywać nawarstwiać, pojawiło się całe mnóstwo wstawek o Janie Chrzcicielu i zapewne cytatów z Biblii, które ani nie są mi bliskie ani do tego dzieła nie pasowały. A szkoda, bo zamysł Natalii Fijewskiej-Zdanowskiej (scenariusz i reżysera), która głos oddaje ludziom dojrzałym z dość dużym już bagażem doświadczeń był naprawdę dobry. Bo co tu dużo mówić stykamy się z ludźmi, których marzenia pozostają w sferze zapomnienia, a najskrytsze pragnienia nigdy nie zostają wypowiedziane na głos. Reżyserka oddaje swój spektakl w ręce pokolenia niespełnionego, ludzi lat 70. i 80., które wychowało się na lęku i toksycznych relacjach. I to ich przeżycia powinny nas-widzów uwierać czy też wzbudzać jakąś refleksję. I zgodzę się, że na początku tak się dzieje, ale później akcja staje w miejscu problemy się nie rozwiązują, a dziejące się wydarzenia nagle znajdują ujście, bez większego ładu i składu.

Mówiąc o spektaklu Osiemnastka nie można jednak zapomnieć o aktorach, którzy byli bardzo przekonujący i prawdziwi w swoich rolach. To było tak dobre połączenie różnych charakterów i osobowości, że momentami na scenie działy się rzeczy piękne. Na tyle, że nie sposób było nie oddać się w ręce bohaterów i dać im po prostu prowadzić się po niuansach tego dzieła. I na próżno dzielić tutaj obsadę na obóz lepiej i gorzej grający, bo to co pokazali na scenie mogło być tylko w tej pierwszej z grup. Naprawdę to był kawał solidnej pracy, doświadczenia które dobrze skomponowało się z opowiadaną historią postaci. I żałuję (bardzo), że koniec tego dzieła, mimo fantastycznej gry aktorskiej, był mało przekonywująco (i słaby). Ale wybrzmieć to musi głośno, bo jakby odrzucić to co złe, to dla Zuzanny Fijewskiej-Malesza, Agaty Kołodziejczyk, Dominiki Łakomskiej, Jarosława Boberka i Pawła Pabisiaka przyszła bym jeszcze raz chętnie na taką imprezę."""

# Przetwarzaj tekst
doc = nlp(text)

# Ekstrakcja nazw własnych
entities = [ent.text for ent in doc.ents]
entity_freq = Counter(entities)

# Analiza częstości słów
words = [token.text for token in doc if token.is_alpha and not token.is_stop]
word_freq = Counter(words)

# Wyświetl wyniki
print("Najczęstsze nazwy własne:", entity_freq.most_common(5))
print("Najczęstsze słowa:", word_freq.most_common(5))

#%% MODEL LDA
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import textwrap

# Załadowanie modelu i tokienizatora
model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Utworzenie pipeline NER
nlp1 = pipeline("ner", model=model, tokenizer=tokenizer)
example = """OSIEMNASTKA, REŻ. NATALIA FIJEWSKA-ZDANOWSKA
Pięciu bohaterów, którzy na widok publiczny wyciągną swoje problemy. Pięciu bohaterów, którzy zmuszeni zostaną do znalezienia przyczyn nieudanego życia. I niech nikogo nie zmyli tytuł tego spektaklu, bo nie o imprezie osiemnastkowej będziemy tutaj rozmawiać, a o bohaterach którzy robić będą dobrą minę do złej gry. Zabawa urodzinowa jest tylko pretekstem do tego, aby zamienić się w rozmowy o egzystencji i niespełnionych pragnieniach
/fot. Rafał Latoszek/

Dla jednych będzie to przewidywalny początek, dla innych dobre wprowadzenie do rozgrywającej się akcji. Bo zaczynamy zwykłą rozmową pomiędzy trzema parami, które spotykają się przypadkiem na wspólnej imprezie osiemnastkowej swoich dzieci. Jest miło i przyjemnie, dialogi bawią, nietrudno też zaśmiać się z tych wszystkich żartów sytuacyjnych które się zadziewają. Lecz trwające coraz dłużej rozmowy zaczynają gęstnieć, a w powietrzu unosi się dziwny zapach żalu i goryczy, który przeżywają bohaterowie. Akcja przyśpieszy kiedy okaże się, że młodzież która tak świetnie bawić się miała na zorganizowanej imprezie zniknęła, a pilnujący ich rodzice obiecali do rana nie wychodzić z domu, który mieści się gdzieś na skraju lasu. Powiało grozą - nie będę ukrywać. Pojawiła się też cała masa pytań, gdzie zniknęły dzieci, dlaczego rodzice nie mogą opuścić budynku, co wydarzy się dalej, kto zginie. Napięcie zaczyna narastać, a pytania zamiast znikać, tylko się mnożyły. Byłam dobrej myśli do tego co zadzieje się dalej, bo oprócz momentów grozy mieliśmy również dobry humor, który trzymał wysoki poziom.

Aż tu nagle… Szok i niedowierzanie, bo spektakl zwolnił do maksimum, sprawy zaczęły zamiast się rozwiązywać nawarstwiać, pojawiło się całe mnóstwo wstawek o Janie Chrzcicielu i zapewne cytatów z Biblii, które ani nie są mi bliskie ani do tego dzieła nie pasowały. A szkoda, bo zamysł Natalii Fijewskiej-Zdanowskiej (scenariusz i reżysera), która głos oddaje ludziom dojrzałym z dość dużym już bagażem doświadczeń był naprawdę dobry. Bo co tu dużo mówić stykamy się z ludźmi, których marzenia pozostają w sferze zapomnienia, a najskrytsze pragnienia nigdy nie zostają wypowiedziane na głos. Reżyserka oddaje swój spektakl w ręce pokolenia niespełnionego, ludzi lat 70. i 80., które wychowało się na lęku i toksycznych relacjach. I to ich przeżycia powinny nas-widzów uwierać czy też wzbudzać jakąś refleksję. I zgodzę się, że na początku tak się dzieje, ale później akcja staje w miejscu problemy się nie rozwiązują, a dziejące się wydarzenia nagle znajdują ujście, bez większego ładu i składu.

Mówiąc o spektaklu Osiemnastka nie można jednak zapomnieć o aktorach, którzy byli bardzo przekonujący i prawdziwi w swoich rolach. To było tak dobre połączenie różnych charakterów i osobowości, że momentami na scenie działy się rzeczy piękne. Na tyle, że nie sposób było nie oddać się w ręce bohaterów i dać im po prostu prowadzić się po niuansach tego dzieła. I na próżno dzielić tutaj obsadę na obóz lepiej i gorzej grający, bo to co pokazali na scenie mogło być tylko w tej pierwszej z grup. Naprawdę to był kawał solidnej pracy, doświadczenia które dobrze skomponowało się z opowiadaną historią postaci. I żałuję (bardzo), że koniec tego dzieła, mimo fantastycznej gry aktorskiej, był mało przekonywująco (i słaby). Ale wybrzmieć to musi głośno, bo jakby odrzucić to co złe, to dla Zuzanny Fijewskiej-Malesza, Agaty Kołodziejczyk, Dominiki Łakomskiej, Jarosława Boberka i Pawła Pabisiaka przyszła bym jeszcze raz chętnie na taką imprezę."""
max_length = 512  # Maksymalna długość fragmentu tekstu, dostosuj w zależności od modelu i ograniczeń pamięci
fragments = textwrap.wrap(example, max_length, break_long_words=False, replace_whitespace=False)

# Analiza każdego fragmentu osobno
ner_results = []
for fragment in fragments:
    ner_results.extend(nlp1(fragment))

print(ner_results)


# Funkcja do łączenia tokenów w pełne jednostki


def combine_tokens(ner_results):
    combined_entities = []
    current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0, "start": None, "end": None}
    previous_end_of_word = False

    for token in ner_results:
        # Sprawdzamy, czy bieżący token jest końcem słowa
        end_of_word = "</w>" in token['word']
        cleaned_word = token['word'].replace("</w>", "")

        # Rozpoczęcie nowej jednostki
        if token['entity'].startswith('B-'):
            if current_entity['word']:
                # Obliczamy średnią ocenę dla skompletowanej jednostki
                current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
                combined_entities.append(current_entity)
            current_entity = {"word": cleaned_word, "type": token['entity'][2:], "score_sum": token['score'],
                              "token_count": 1, "start": token['start'], "end": token['end']}
            previous_end_of_word = end_of_word
        # Kontynuacja obecnej jednostki
        elif token['entity'].startswith('I-') and current_entity['type'] == token['entity'][2:]:
            # Dodajemy spację przed bieżącym tokenem, jeśli poprzedni token był końcem słowa
            if previous_end_of_word:
                current_entity['word'] += " " + cleaned_word
            else:
                current_entity['word'] += cleaned_word
            current_entity['end'] = token['end']
            current_entity['score_sum'] += token['score']
            current_entity['token_count'] += 1
            previous_end_of_word = end_of_word

    # Dodajemy ostatnią jednostkę, jeśli istnieje
    if current_entity['word']:
        current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
        combined_entities.append(current_entity)

    return combined_entities

# Wywołanie funkcji
combined_entities = combine_tokens


# Wywołanie funkcji
combined_entities = combine_tokens(ner_results)

# Wyświetlanie wyników
for entity in combined_entities:
    print(entity)

entities = [entity['word'] for entity in combined_entities]  


# Przygotowanie danych: tokenizacja i usuwanie słów stopu
import spacy
import re

# Załaduj polski model
nlp = spacy.load('pl_core_news_sm')

# Pobierz polskie stop words
stop_words = nlp.Defaults.stop_words
def clean_text(text):
    # Przetwarzanie tekstu za pomocą spaCy
    doc = nlp(text)
    # Usunięcie znaków specjalnych i lematyzacja, zachowując tylko te tokeny, które nie są stop words
    lemmatized = []
    for token in doc:
        if token.text not in stop_words:
            lemma = token.lemma_.lower()
            lemma = re.sub(r'\W', ' ', lemma)  # Usunięcie znaków specjalnych z lematów
            lemma = re.sub(r'\s+[a-zA-Z]\s+', ' ', lemma)  # Usunięcie wszystkich pojedynczych liter
            lemma = re.sub(r'\s+', ' ', lemma, flags=re.I)  # Zamiana wielokrotnych spacji na pojedynczą spację
            lemmatized.append(lemma.strip())
    
    # Usunięcie pustych tokenów po czyszczeniu
    lemmatized = [token for token in lemmatized if token]

    return lemmatized
texts=clean_text(example)
documents = [texts]
extended_text = texts + entities
documents = [extended_text]
# Tworzenie słownika
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

# Tworzenie modelu LDA
lda = LdaModel(corpus, num_topics=7, id2word=dictionary, passes=30)

# Wyświetlanie tematów
topics = lda.print_topics(num_words=4)
for topic in topics:
    print(topic)
#%% Grafy Lematyzacja i ujdenolicanie
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import textwrap

# Załadowanie modelu i tokienizatora
model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

# Utworzenie pipeline NER
nlp1 = pipeline("ner", model=model, tokenizer=tokenizer)
example = """OSIEMNASTKA, REŻ. NATALIA FIJEWSKA-ZDANOWSKA </tytuł>
Pięciu bohaterów, którzy na widok publiczny wyciągną swoje problemy. Pięciu bohaterów, którzy zmuszeni zostaną do znalezienia przyczyn nieudanego życia. I niech nikogo nie zmyli tytuł tego spektaklu, bo nie o imprezie osiemnastkowej będziemy tutaj rozmawiać, a o bohaterach którzy robić będą dobrą minę do złej gry. Zabawa urodzinowa jest tylko pretekstem do tego, aby zamienić się w rozmowy o egzystencji i niespełnionych pragnieniach
/fot. Rafał Latoszek/

Dla jednych będzie to przewidywalny początek, dla innych dobre wprowadzenie do rozgrywającej się akcji. Bo zaczynamy zwykłą rozmową pomiędzy trzema parami, które spotykają się przypadkiem na wspólnej imprezie osiemnastkowej swoich dzieci. Jest miło i przyjemnie, dialogi bawią, nietrudno też zaśmiać się z tych wszystkich żartów sytuacyjnych które się zadziewają. Lecz trwające coraz dłużej rozmowy zaczynają gęstnieć, a w powietrzu unosi się dziwny zapach żalu i goryczy, który przeżywają bohaterowie. Akcja przyśpieszy kiedy okaże się, że młodzież która tak świetnie bawić się miała na zorganizowanej imprezie zniknęła, a pilnujący ich rodzice obiecali do rana nie wychodzić z domu, który mieści się gdzieś na skraju lasu. Powiało grozą - nie będę ukrywać. Pojawiła się też cała masa pytań, gdzie zniknęły dzieci, dlaczego rodzice nie mogą opuścić budynku, co wydarzy się dalej, kto zginie. Napięcie zaczyna narastać, a pytania zamiast znikać, tylko się mnożyły. Byłam dobrej myśli do tego co zadzieje się dalej, bo oprócz momentów grozy mieliśmy również dobry humor, który trzymał wysoki poziom.

Aż tu nagle… Szok i niedowierzanie, bo spektakl zwolnił do maksimum, sprawy zaczęły zamiast się rozwiązywać nawarstwiać, pojawiło się całe mnóstwo wstawek o Janie Chrzcicielu i zapewne cytatów z Biblii, które ani nie są mi bliskie ani do tego dzieła nie pasowały. A szkoda, bo zamysł Natalii Fijewskiej-Zdanowskiej (scenariusz i reżysera), która głos oddaje ludziom dojrzałym z dość dużym już bagażem doświadczeń był naprawdę dobry. Bo co tu dużo mówić stykamy się z ludźmi, których marzenia pozostają w sferze zapomnienia, a najskrytsze pragnienia nigdy nie zostają wypowiedziane na głos. Reżyserka oddaje swój spektakl w ręce pokolenia niespełnionego, ludzi lat 70. i 80., które wychowało się na lęku i toksycznych relacjach. I to ich przeżycia powinny nas-widzów uwierać czy też wzbudzać jakąś refleksję. I zgodzę się, że na początku tak się dzieje, ale później akcja staje w miejscu problemy się nie rozwiązują, a dziejące się wydarzenia nagle znajdują ujście, bez większego ładu i składu.

Mówiąc o spektaklu Osiemnastka nie można jednak zapomnieć o aktorach, którzy byli bardzo przekonujący i prawdziwi w swoich rolach. To było tak dobre połączenie różnych charakterów i osobowości, że momentami na scenie działy się rzeczy piękne. Na tyle, że nie sposób było nie oddać się w ręce bohaterów i dać im po prostu prowadzić się po niuansach tego dzieła. I na próżno dzielić tutaj obsadę na obóz lepiej i gorzej grający, bo to co pokazali na scenie mogło być tylko w tej pierwszej z grup. Naprawdę to był kawał solidnej pracy, doświadczenia które dobrze skomponowało się z opowiadaną historią postaci. I żałuję (bardzo), że koniec tego dzieła, mimo fantastycznej gry aktorskiej, był mało przekonywująco (i słaby). Ale wybrzmieć to musi głośno, bo jakby odrzucić to co złe, to dla Zuzanny Fijewskiej-Malesza, Agaty Kołodziejczyk, Dominiki Łakomskiej, Jarosława Boberka i Pawła Pabisiaka przyszła bym jeszcze raz chętnie na taką imprezę."""
max_length = 512  # Maksymalna długość fragmentu tekstu, dostosuj w zależności od modelu i ograniczeń pamięci
fragments = textwrap.wrap(example, max_length, break_long_words=False, replace_whitespace=False)

# Analiza każdego fragmentu osobno
ner_results = []
for fragment in fragments:
    ner_results.extend(nlp1(fragment))

print(ner_results)


# Funkcja do łączenia tokenów w pełne jednostki


def combine_tokens(ner_results):
    combined_entities = []
    current_entity = {"word": "", "type": None, "score_sum": 0, "token_count": 0, "start": None, "end": None}
    previous_end_of_word = False

    for token in ner_results:
        # Sprawdzamy, czy bieżący token jest końcem słowa
        end_of_word = "</w>" in token['word']
        cleaned_word = token['word'].replace("</w>", "")

        # Rozpoczęcie nowej jednostki
        if token['entity'].startswith('B-'):
            if current_entity['word']:
                # Obliczamy średnią ocenę dla skompletowanej jednostki
                current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
                combined_entities.append(current_entity)
            current_entity = {"word": cleaned_word, "type": token['entity'][2:], "score_sum": token['score'],
                              "token_count": 1, "start": token['start'], "end": token['end']}
            previous_end_of_word = end_of_word
        # Kontynuacja obecnej jednostki
        elif token['entity'].startswith('I-') and current_entity['type'] == token['entity'][2:]:
            # Dodajemy spację przed bieżącym tokenem, jeśli poprzedni token był końcem słowa
            if previous_end_of_word:
                current_entity['word'] += " " + cleaned_word
            else:
                current_entity['word'] += cleaned_word
            current_entity['end'] = token['end']
            current_entity['score_sum'] += token['score']
            current_entity['token_count'] += 1
            previous_end_of_word = end_of_word

    # Dodajemy ostatnią jednostkę, jeśli istnieje
    if current_entity['word']:
        current_entity['score'] = current_entity['score_sum'] / current_entity['token_count']
        combined_entities.append(current_entity)

    return combined_entities




# Wywołanie funkcji
combined_entities = combine_tokens(ner_results)

import networkx as nx
from spacy.lang.pl import Polish
import spacy
from collections import defaultdict
from fuzzywuzzy import fuzz
from collections import defaultdict
from definicje import *
entities = [entity['word'] for entity in combined_entities]  
entities_list=entities
# Ładowanie spaCy
nlp = spacy.load("pl_core_news_lg")

# Przetwarzanie tekstu
doc = nlp(example.lower())
lemmatized_text = " ".join([token.lemma_ for token in doc])

# Ponowne przetworzenie lematyzowanego tekstu, aby umożliwić analizę zdań


# Lematyzacja bytów
lemmatized_entities = []
for entity in entities:
    doc_entity = nlp(entity.lower())
    lemmatized_entity = " ".join([token.lemma_ for token in doc_entity])
    lemmatized_entities.append(lemmatized_entity)



# Ładowanie modelu spaCy z wektorami


# Przekształcanie bytów na obiekty doc spaCy
threshold = 80

def group_similar_entities(entities, threshold):
    groups = []
    for entity in entities:
        added = False
        for group in groups:
            if any(fuzz.token_sort_ratio(entity, member) > threshold for member in group):
                group.append(entity)
                added = True
                break
        if not added:
            groups.append([entity])
    return groups

entity_groups = group_similar_entities(lemmatized_entities, threshold)

# Zmiana tutaj: wybieranie najkrótszego bytu w grupie jako reprezentanta
representatives = [sorted(group, key=lambda x: len(x))[0] for group in entity_groups]

entity_to_representative_map = {}
for group in entity_groups:
    representative = sorted(group, key=lambda x: len(x))[0]
    for entity in group:
        entity_to_representative_map[entity] = representative

for entity, representative in entity_to_representative_map.items():
    print(f'"{entity}" zostanie zastąpione przez "{representative}"')

def replace_entities_with_representatives(text, map):
    # Sortowanie kluczy według długości tekstu malejąco, aby najpierw zastąpić dłuższe frazy
    sorted_entities = sorted(map.keys(), key=len, reverse=True)
    
    for entity in sorted_entities:
        representative = map[entity]
        # Zastąpienie klucza (bytu) jego wartością (reprezentantem) w tekście
        text = text.replace(entity, representative)
    
    return text



# Zastępowanie bytów w tekście ich reprezentantami
updated_text = replace_entities_with_representatives(lemmatized_text, entity_to_representative_map)

print(updated_text)

list_of_new_entities=(list(entity_to_representative_map.values()))
unique(list_of_new_entities)
lemmatized_doc = nlp(updated_text)
co_occurrence = defaultdict(int)

# Tworzenie grafu
G = nx.Graph()

# Dodanie węzłów dla każdego zlematyzowanego bytu

title_end_position = updated_text.find("< /tytuł >")

# Iteracja przez byty i zwiększenie ich wagi, jeśli występują przed tagiem </tytuł>

for entity in list_of_new_entities:
    # Sprawdzenie, czy byt znajduje się przed końcem tytułu
    entity_position = updated_text.find(entity)
    if entity_position != -1 and entity_position < title_end_position:
        # Dodanie bytu do grafu z domyślną wagą, jeśli jeszcze nie istnieje
        if entity not in G:
            G.add_node(entity, weight=1.0)
        # Zwiększenie wagi bytu o 0.5, jeśli znajduje się przed końcem tytułu
        G.nodes[entity]['weight'] = G.nodes[entity].get('weight', 1.0) + 0.5
for node, attrs in G.nodes(data=True):
    print(f"Waga bytu '{node}': {attrs.get('weight', 'Brak wagi')}")
# Ponowne obliczenie centralności i wizualizacja grafu
 
    
# Analiza współwystępowania zlematyzowanych bytów
for sent in lemmatized_doc.sents:
    sent_text = " ".join([token.lemma_ for token in sent])
    entities_in_sent = [entity for entity in list_of_new_entities if entity in sent_text]
    for i in range(len(entities_in_sent)):
        for j in range(i + 1, len(entities_in_sent)):
            co_occurrence[(entities_in_sent[i], entities_in_sent[j])] += 1
            G.add_edge(entities_in_sent[i], entities_in_sent[j], weight=co_occurrence[(entities_in_sent[i], entities_in_sent[j])])

# Rysowanie grafu
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', font_size=12, font_weight='bold')
plt.title("Graf współwystępowania bytów")
plt.show()
# Obliczanie centralności stopniowej
degree_centrality = nx.degree_centrality(G)

# Sortowanie i wyświetlanie
sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
print("Centralność bytów:")
for entity, centrality in sorted_centrality:
    print(f"{entity}: {centrality}")
    
#%% Proste zliczanie bytów w tytule +1 - moja waga

entity_counts = {entity: 0 for entity in list_of_new_entities}

# Znalezienie końca tytułu
title_end_pos = updated_text.find("< /tytuł >")

# Zliczanie wystąpień
for entity in list_of_new_entities:
    # Liczenie wszystkich wystąpień bytu
    total_occurrences = updated_text.count(entity)
    entity_counts[entity] += total_occurrences
    
    # Sprawdzenie, czy byt występuje w tytule i dodanie dodatkowego punktu
    if updated_text.find(entity) < title_end_pos:
        entity_counts[entity] += 1

# Sortowanie i wyświetlanie wyników
sorted_entity_counts = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)

print("Wystąpienia bytów z uwzględnieniem ważności tytułu:")
for entity, count in sorted_entity_counts:
    print(f"{entity}: {count}")    
    
    
#%%
import spacy
nlp = spacy.load("pl_core_news_sm")

doc = nlp("jedzcie kolację")
for t in doc:
    print(f"{t.text:<10} => {t.lemma_:<10}")
    
import spacy

# Załaduj model języka polskiego
nlp = spacy.load("pl_core_news_sm")

# Lista obiektów
obiekty = ['NATALIA FIJEWSKA - ZDANOWSKA',
 'Rafał Latoszek',
 'Janie Chrzcicielu',
 'Biblii',
 'Natalii Fijewskiej - Zdanowskiej',
 'Osiemnastka',
 'Zuzanny Fijewskiej - Malesza',
 'Agaty Kołodziejczyk',
 'Dominiki Łakomskiej',
 'Jarosława Boberka',
 'Pawła Pabisiaka']

# Zlematyzuj obiekty
zlematyzowane_obiekty = []

for obiekt in obiekty:
    doc = nlp(obiekt)
    zlematyzowany_obiekt = ' '.join([token.lemma_ for token in doc])
    zlematyzowane_obiekty.append(zlematyzowany_obiekt)

# Wyświetl zlematyzowane obiekty
for zlematyzowany_obiekt in zlematyzowane_obiekty:
    print(zlematyzowany_obiekt)
    
    
from morfeusz2 import Morfeusz

# Initialize Morfeusz
morfeusz = Morfeusz()

# List of entities
entities = [
    'NATALIA FIJEWSKA - ZDANOWSKA',
    'Rafał Latoszek',
    'Janie Chrzcicielu',
    'Biblii',
    'Natalii Fijewskiej - Zdanowskiej',
    'Osiemnastka',
    'Zuzanny Fijewskiej - Malesza',
    'Agaty Kołodziejczyk',
    'Dominiki Łakomskiej',
    'Jarosława Boberka',
    'Pawła Pabisiaka'
]

# Lemmatize entities
lemmatized_entities = []

def select_lemma(analysis):
    for entry in analysis:
        if 'imię' in entry[2][3] or 'nazwisko' in entry[2][3]:
            return entry[2][1]  # Return lemma if it's a proper name or surname
    return analysis[0][2][1]  # Fallback to the first lemma if no proper name or surname is found

for entity in entities:
    analysis = morfeusz.analyse(entity)
    lemmatized_entities.append(select_lemma(analysis))

print(lemmatized_entities)
import pymorphy2

# Inicjalizuj Pymorphy2
morph = pymorphy2.MorphAnalyzer()

# Lista obiektów
obiekty = ['NATALIA FIJEWSKA - ZDANOWSKA',
 'Rafał Latoszek',
 'Janie Chrzcicielu',
 'Biblii',
 'Natalii Fijewskiej - Zdanowskiej',
 'Osiemnastka',
 'Zuzanny Fijewskiej - Malesza',
 'Agaty Kołodziejczyk',
 'Dominiki Łakomskiej',
 'Jarosława Boberka',
 'Pawła Pabisiaka']

# Zlematyzuj obiekty
zlematyzowane_obiekty = []

for obiekt in obiekty:
    zlematyzowany_obiekt = morph.parse(obiekt)[0].normal_form
    zlematyzowane_obiekty.append(zlematyzowany_obiekt)

# Wyświetl zlematyzowane obiekty
for zlematyzowany_obiekt in zlematyzowane_obiekty:
    print(zlematyzowany_obiekt)

import spacy

# Załaduj polski model językowy
nlp = spacy.load("pl_core_news_lg")

# Przykładowy tekst w języku polskim
tekst = "Natalii Fijewskiej - Zdanowskiej pracowała z Rafałem Latoszkiem nad projektem."

# Przetwórz tekst za pomocą modelu
doc = nlp(tekst)

# Wydrukuj lematy dla każdego słowa w tekście
for token in doc:
    print(f"{token.text} -> {token.lemma_}")