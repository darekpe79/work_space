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
    
    
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_checkpoint = "pietruszkowiec/herbert-base-ner"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Nazywam się Grzegorz Brzęszczyszczykiewicz, pochodzę "\
    "z Chrząszczyżewoszczyc, pracuję w Łękołodzkim Urzędzie Powiatowym"
example= """ZAMEK, REŻ. FRANCISZEK SZUMIŃSKI

Zamek w reżyserii Franciszka Szumińskiego w Teatrze Dramatycznym w Warszawie przyniósł wiele rozczarowań. Stał się spektaklem pozbawionym jakichkolwiek znaczeń i pola do interpretacji. Zamiast tego, postawiono na dosłowność, która raziła i bolała zarazem. Dlatego tak bardzo marzyłam o tym, żeby to dzieło skończyło się jak najszybciej. Co oczywiście okazało się być marzeniem nie do spełnienia.
Wiem, mogłam opuścić widownię podczas trwania przerwy, ale gdzieś wewnątrz chciałam dać temu działu jeszcze jedną szansę. Licząc, że coś się zmieni. Że spektakl nabierze tempa, a aktorzy wyjdą po raz kolejny na scenę i zaczną prawdziwy popis swoich umiejętności. Oczywiście, niewiele z tego wszystkiego zostało, ba mam wrażenie że skończyło się jeszcze gorzej. Ale zacznijmy od początku… Nasz bohater K. (Modest Ruciński) przyjeżdża do pewnej, bliżej nie określonej, wioski, gdzie przychodzi zmierzyć mu się z wieloma absurdami jego przybycia. Niby został zatrudniony jako geometra, niby ktoś tam o jego przybyciu wie, ale z drugiej strony żadnego geometry w tym miejscu nie trzeba, a i pozwolenie na pracę ciężko zdobyć. Jedynym ratunkiem i wybawieniem dla K. może być Zamek, który określa zasady życia w tym miejscu, funkcjonowanie obywateli, a i wprowadza pewien system który działać ma zgodnie z tym co zostało nakazane. Dlatego też obywatele, żyją w zgodzie z nałożonymi prawami i obowiązkami.

Gdyby wszystko w tym spektaklu poszło jak należy, dzielilibyśmy z K. trudy jego zmagań z wyższą władzą, siłą która potrafi bez skrupułów zniszczyć człowieka, ale też dać mu jakieś strzępy nadziei, tylko po to aby dostosowanie się do panujących warunków było koniecznością. Dzielilibyśmy z K. ten trud podjęcia decyzji o opuszczeniu tej bliżej nieokreślonej wsi i próby dostania się do Zamku. Zastanowilibyśmy się nad symboliką przedstawionego świata czy mechanizmami, które się w nim pojawiają. Moglibyśmy interpretować oglądane dzieło według własnego widzimisię, bo przecież taki jest Kafka – wymusza szukanie głębszego sensu, porzucenie tego co widzialne, zaufanie temu co nieokreślone, niejasne i ukryte.

W tym spektaklu jakby wszystko działo się odwrotnie. Dostajemy dzieło pozbawione jakichkolwiek sensów. To, co widzę wydaje mi się prawdą narzuconą, trochę słabo zarysowaną, niestwarzającą tej możliwości czytania pomiędzy dialogami czy pomiędzy postaciami. A i w bohaterach tego dzieła trudno się czegokolwiek dopatrzeć, bo te postacie są po prostu miałkie i nijakie. Niby aktorzy próbują wzbudzić w swoich postaciach emocje, brzmieć naturalnie, tchnąć życie w to dzieło, opowiedzieć o czymś. Wszystko to, wydaje się jednak tak bardzo nieosiągalne. I tak naprawdę szkoda tego spektaklu, bo zadatki na dobre dzieło miał duże. Zaczynając od mrocznej i mocno zarysowanej scenografii autorstwa Julii Basisty, przez reżyserię światła Daniela Sanjuana-Ciepielewskiego, a kończąc na możliwościach tekstu Kafki. W ostatecznym jednak rozrachunku mierzymy się ze słabo zagranym i mówiąc wprost nudnym dziełem."""
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





