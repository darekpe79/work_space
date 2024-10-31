# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:01:50 2024

@author: dariu
"""

from pymarc import MARCReader, XMLWriter

# Otwieranie pliku MARC i pliku wyjściowego XML
with open('input.mrc', 'rb') as marc_file, open('output.xml', 'wb') as xml_file:
    reader = MARCReader(marc_file)
    writer = XMLWriter(xml_file)

    # Przechodzenie przez każdy rekord i zapisywanie go w formacie XML
    for record in reader:
        writer.write(record)

    writer.close()  # Zamknięcie XMLWriter jest konieczne



import requests
import xml.etree.ElementTree as ET

# URL repozytorium książek
api_url = "https://bibliotekanauki.pl/api/oai/books"

# Parametry zapytania
params = {
    "verb": "ListRecords",
    "metadataPrefix": "oai_dc"  # Można użyć "bits" dla bardziej szczegółowego formatu
}

# Wysłanie zapytania
response = requests.get(api_url, params=params)

# Sprawdzenie odpowiedzi
if response.status_code == 200:
    # Parsowanie odpowiedzi XML
    root = ET.fromstring(response.content)

    # Lista na pełne rekordy XML
    full_records_xml = []

    for record in root.findall(".//{http://www.openarchives.org/OAI/2.0/}record"):
        # Pobieramy cały element <record> jako surowy XML i dodajemy do listy
        full_records_xml.append(ET.tostring(record, encoding='utf-8').decode('utf-8'))

    # Wyświetlamy pierwsze 5 pełnych rekordów XML (jako surowe XML)
    for record_xml in full_records_xml[:5]:  # Pokaż 5 pierwszych rekordów dla przykładu
        print(record_xml)
        print("\n" + "-"*50 + "\n")  # Separator między rekordami dla czytelności

else:
    print("Błąd podczas pobierania danych:", response.status_code)

import requests
import xml.etree.ElementTree as ET

# URL repozytorium książek
api_url = "https://bibliotekanauki.pl/api/oai/articles"

# Parametry pierwszego zapytania
params = {
    "verb": "ListRecords",
    "metadataPrefix": "jats"  # lub "bits" dla bardziej szczegółowego formatu
}

# Ustawienia limitu
record_limit = 50  # Liczba maksymalnych rekordów do pobrania
counter = 0  # Licznik pobranych rekordów

# Lista na pełne rekordy XML
full_records_xml = []

while True:
    # Tworzymy URL z parametrami do podglądu
    request_url = requests.Request('GET', api_url, params=params).prepare().url
    print("Pobieranie z URL:", request_url)

    # Wysłanie zapytania
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        # Parsowanie odpowiedzi XML
        root = ET.fromstring(response.content)
        
        # Zbieranie rekordów
        for record in root.findall(".//{http://www.openarchives.org/OAI/2.0/}record"):
            full_records_xml.append(ET.tostring(record, encoding='utf-8').decode('utf-8'))
            counter += 1
            if counter >= record_limit:
                print("Osiągnięto limit rekordów.")
                break  # Wyjście z pętli, gdy osiągniemy limit

        # Sprawdzenie, czy osiągnięto limit po wewnętrznej pętli
        if counter >= record_limit:
            break

        # Sprawdzenie, czy istnieje resumptionToken
        token = root.find(".//{http://www.openarchives.org/OAI/2.0/}resumptionToken")
        if token is not None and token.text:
            # Ustawiamy parametr `resumptionToken` na wartość tokena
            params = {
                "verb": "ListRecords",
                "resumptionToken": token.text
            }
        else:
            break  # Jeśli brak tokena, kończymy pętlę
    else:
        print("Błąd podczas pobierania danych:", response.status_code)
        break

# Wyświetlanie liczby pobranych rekordów
print(f"Pobrano {counter} rekordów.")
# Wyświetlamy kilka rekordów dla podglądu
for record_xml in full_records_xml[:5]:
    print(record_xml)
    print("\n" + "-"*50 + "\n")  # Separator dla czytelności
