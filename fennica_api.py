import os

current = os.getcwd() #get working directory
os.chdir("C:\\Program Files\\NordVPN") #cd into nord directory
os.system("nordvpn -c -g 'Finland'") #change ip
os.chdir(current) #cd back into working directory

EUROPEAN_COUNTRIES = ["Poland", "Germany", "Austria", "Finland"]
for country in EUROPEAN_COUNTRIES:
    try:
        current = os.getcwd()  # Pobierz bieżący katalog roboczy
        os.chdir("C:\\Program Files\\NordVPN")  # Przejdź do katalogu NordVPN
        print("Rozłączanie aktualnego połączenia VPN...")
        os.system("nordvpn disconnect")  # Rozłącz
        command = f'nordvpn -c -g "{country}"'  # Komenda zmiany serwera
        os.system(command)
        os.chdir(current)  # Powrót do bieżącego katalogu roboczego
        print(f"Zmieniono IP na serwer w {country}.")
        
    except Exception as e:
        print(f"Błąd podczas zmiany IP: {e}")






import os
import requests
from tqdm import tqdm
from pymarc import MARCWriter, parse_xml_to_array
import xml.etree.ElementTree as ET
from io import BytesIO
from itertools import cycle
import time

# Parametry połączenia
base_url = 'https://oai-pmh.api.melinda.kansalliskirjasto.fi/bib'
chunk_size = 100000  # Liczba rekordów w jednym pliku
records_per_vpn_change = 55000  # Liczba rekordów między zmianami VPN
file_count = 1  # Numeracja plików
records_in_file = 0  # Licznik rekordów w bieżącym pliku
records_since_last_vpn_change = 0  # Licznik rekordów od ostatniej zmiany VPN
output_dir = "D:\\Nowa_praca"  # Katalog wyjściowy
start_date = "2023-12-01"  # Początek zakresu dat
end_date = "2023-12-31"    # Koniec zakresu dat
resumption_token = None
current_chunk_records = []  # Lista rekordów dla bieżącego pliku

# Kraje europejskie dla NordVPN
EUROPEAN_COUNTRIES = ["Poland", "Germany", "Austria", "Finland"]
country_cycle = cycle(EUROPEAN_COUNTRIES)  # Generator cykliczny

# Tworzenie katalogu wyjściowego, jeśli nie istnieje
os.makedirs(output_dir, exist_ok=True)

# Lista prefiksów UKD dla literatury
LITERATURE_UKD_PREFIXES = [
    "82", "830", "840", "850", "860", "870", "880", "890"
]
# Klasy YKL związane z literaturą
LITERATURE_YKL_CLASSES = [
    "80", "82", "83", "84", "85", "86"
]

# Funkcja zapisująca rekordy do plików MARC
def save_records_to_file(records, file_count):
    file_name = os.path.join(output_dir, f"fennica_records_20231{file_count}.mrc")
    with open(file_name, 'wb') as marc_file:
        writer = MARCWriter(marc_file)
        for record in records:
            try:
                writer.write(record)
            except Exception as e:
                print(f"Błąd podczas zapisywania rekordu: {e}")
        writer.close()
    print(f"Zapisano {len(records)} rekordów do pliku: {file_name}")

# Funkcja usuwająca przestrzenie nazw z XML
def remove_namespace(xml_string):
    xml_bytes = BytesIO(xml_string.encode('utf-8'))
    it = ET.iterparse(xml_bytes, events=['start', 'end'])
    for _, el in it:
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    return ET.tostring(it.root, encoding='utf-8')

def is_literary_record(record):
    """
    Sprawdza, czy rekord dotyczy literatury na podstawie pól 080 i 084.
    """
    for field in record.get_fields('080'):
        if field['a'] and any(field['a'].startswith(prefix) for prefix in LITERATURE_UKD_PREFIXES):
            return True
    for field in record.get_fields('084'):
        if field['a'] and any(field['a'].startswith(prefix) for prefix in LITERATURE_YKL_CLASSES):
            return True
    return False

def change_ip():
    """
    Zmienia IP za pomocą NordVPN iterując cyklicznie po krajach.
    """
    try:
        current = os.getcwd()  # Pobierz bieżący katalog roboczy
        os.chdir("C:\\Program Files\\NordVPN")  # Przejdź do katalogu NordVPN

        country = next(country_cycle)  # Pobierz następny kraj
        command = f'nordvpn -c -g "{country}"'
        os.system(command)
        print(f"Zmieniono IP na serwer w {country}.")
        
        os.chdir(current)  # Powrót do poprzedniego katalogu roboczego
    except Exception as e:
        print(f"Błąd podczas zmiany IP: {e}")

# Test funkcji

# Główna pętla pobierania danych
while True:
    # Przygotowanie parametrów zapytania
    params = {
        'verb': 'ListRecords',
        'metadataPrefix': 'marc21',
        'from': start_date,
        'until': end_date,
        'set': 'fennica'
    }
    if resumption_token:
        params['resumptionToken'] = resumption_token
    
    # Wysyłanie zapytania
    print(f"Wysyłanie zapytania... (plik {file_count})")
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Błąd: {response.status_code}")
        break
    
    # Przetwarzanie odpowiedzi
    try:
        # Parsowanie XML i usuwanie przestrzeni nazw
        root = ET.ElementTree(ET.fromstring(response.content)).getroot()
        raw_xml_cleaned = remove_namespace(ET.tostring(root, encoding='unicode'))
        
        # Pobranie rekordów z XML
        for record_elem in root.findall('.//{http://www.openarchives.org/OAI/2.0/}record'):
            metadata_elem = record_elem.find('.//{http://www.loc.gov/MARC21/slim}record')
            if metadata_elem is not None:
                marcxml = (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<collection xmlns="http://www.loc.gov/MARC21/slim">\n'
                    + ET.tostring(metadata_elem, encoding='unicode') +
                    '</collection>'
                )
                try:
                    record_list = parse_xml_to_array(BytesIO(marcxml.encode('utf-8')))
                    # Filtrowanie rekordów literackich
                    literary_records = [rec for rec in record_list if is_literary_record(rec)]
                    current_chunk_records.extend(literary_records)
                    records_in_file += len(literary_records)
                    records_since_last_vpn_change += len(literary_records)
                    
                    # Drukuj licznik każdego przetworzonego rekordu
                    for _ in literary_records:
                        print(f"Przetworzono rekord #{records_in_file}")
                    
                    # Sprawdź, czy osiągnięto rozmiar chunk
                    if records_in_file >= chunk_size:
                        save_records_to_file(current_chunk_records, file_count)
                        file_count += 1
                        records_in_file = 0
                        current_chunk_records = []  # Wyczyść listę dla następnego pliku
                    
                    # Zmiana IP i przerwa co 30,000 rekordów
                    if records_since_last_vpn_change >= records_per_vpn_change:
                        print("Zmiana serwera VPN...")
                        change_ip()
                        print("Pauza 5 minut...")
                        time.sleep(800)  # 5 minut przerwy
                        records_since_last_vpn_change = 0  # Zresetuj licznik po zmianie VPN
                except Exception as e:
                    print(f"Błąd podczas parsowania rekordu: {e}")
                    print(f"Surowy XML rekordu:\n{marcxml}")
    
    except Exception as e:
        print(f"Błąd podczas przetwarzania odpowiedzi: {e}")
        break
    
    # Pobranie tokenu stronicowania
    try:
        resumption_token_elem = root.find('.//{http://www.openarchives.org/OAI/2.0/}resumptionToken')
        if resumption_token_elem is not None:
            resumption_token = resumption_token_elem.text
            print(f"Pobrano resumptionToken: {resumption_token}")
        else:
            print("Brak resumptionToken. Pobieranie zakończone.")
            break
    except Exception as e:
        print(f"Błąd podczas pobierania resumptionToken: {e}")
        break

# Zapis ostatniej porcji rekordów, jeśli pozostały jakieś
if current_chunk_records:
    save_records_to_file(current_chunk_records, file_count)














#%%
import os
import requests
from tqdm import tqdm
from pymarc import MARCWriter, parse_xml_to_array
import xml.etree.ElementTree as ET
from io import BytesIO

# Parametry połączenia
base_url = 'https://oai-pmh.api.melinda.kansalliskirjasto.fi/bib'
chunk_size = 100000  # Liczba rekordów w jednym pliku
file_count = 1  # Numeracja plików
records_in_file = 0  # Licznik rekordów w bieżącym pliku
output_dir = "D:\\Nowa_praca"  # Katalog wyjściowy
start_date = "2023-01-01"  # Początek zakresu dat
end_date = "2023-12-31"    # Koniec zakresu dat

# Tworzenie katalogu wyjściowego, jeśli nie istnieje
os.makedirs(output_dir, exist_ok=True)

# Lista prefiksów UKD dla literatury
LITERATURE_UKD_PREFIXES = [
    "82", "830", "840", "850", "860", "870", "880", "890"
]
# Klasy YKL związane z literaturą
LITERATURE_YKL_CLASSES = [
    "80", "82", "83", "84", "85", "86"
]

# Funkcja zapisująca rekordy do plików MARC
def save_records_to_file(records, file_count):
    file_name = os.path.join(output_dir, f"fennica_records_3{file_count}.mrc")
    with open(file_name, 'wb') as marc_file:
        writer = MARCWriter(marc_file)
        for record in records:
            try:
                writer.write(record)
            except Exception as e:
                print(f"Błąd podczas zapisywania rekordu: {e}")
        writer.close()
    print(f"Zapisano {len(records)} rekordów do pliku: {file_name}")

# Funkcja usuwająca przestrzenie nazw z XML
def remove_namespace(xml_string):
    xml_bytes = BytesIO(xml_string.encode('utf-8'))
    it = ET.iterparse(xml_bytes, events=['start', 'end'])
    for _, el in it:
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    return ET.tostring(it.root, encoding='utf-8')

def is_literary_record(record):
    """
    Sprawdza, czy rekord dotyczy literatury na podstawie pól 080 i 084.
    """
    for field in record.get_fields('080'):
        if field['a'] and any(field['a'].startswith(prefix) for prefix in LITERATURE_UKD_PREFIXES):
            return True
    for field in record.get_fields('084'):
        if field['a'] and any(field['a'].startswith(prefix) for prefix in LITERATURE_YKL_CLASSES):
            return True
    return False

current_chunk_records = []  # Lista rekordów dla bieżącego pliku
resumption_token = None

while True:
    # Przygotowanie parametrów zapytania
    params = {
        'verb': 'ListRecords',
        'metadataPrefix': 'marc21',
        'from': start_date,
        'until': end_date,
        'set': 'fennica'
    }
    if resumption_token:
        params['resumptionToken'] = resumption_token
    
    # Wysyłanie zapytania
    print(f"Wysyłanie zapytania... (plik {file_count})")
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Błąd: {response.status_code}")
        break
    
    # Przetwarzanie odpowiedzi
    try:
        # Parsowanie XML i usuwanie przestrzeni nazw
        root = ET.ElementTree(ET.fromstring(response.content)).getroot()
        raw_xml_cleaned = remove_namespace(ET.tostring(root, encoding='unicode'))
        
        # Pobranie rekordów z XML
        for record_elem in root.findall('.//{http://www.openarchives.org/OAI/2.0/}record'):
            metadata_elem = record_elem.find('.//{http://www.loc.gov/MARC21/slim}record')
            if metadata_elem is not None:
                marcxml = (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<collection xmlns="http://www.loc.gov/MARC21/slim">\n'
                    + ET.tostring(metadata_elem, encoding='unicode') +
                    '</collection>'
                )
                try:
                    record_list = parse_xml_to_array(BytesIO(marcxml.encode('utf-8')))
                    # Filtrowanie rekordów literackich
                    literary_records = [rec for rec in record_list if is_literary_record(rec)]
                    current_chunk_records.extend(literary_records)
                    records_in_file += len(literary_records)
                    
                    # Drukuj licznik każdego przetworzonego rekordu
                    for _ in literary_records:
                        print(f"Przetworzono rekord #{records_in_file}")
                    
                    # Sprawdź, czy osiągnięto rozmiar chunk
                    if records_in_file >= chunk_size:
                        save_records_to_file(current_chunk_records, file_count)
                        file_count += 1
                        records_in_file = 0
                        current_chunk_records = []  # Wyczyść listę dla następnego pliku
                except Exception as e:
                    print(f"Błąd podczas parsowania rekordu: {e}")
                    print(f"Surowy XML rekordu:\n{marcxml}")
    
    except Exception as e:
        print(f"Błąd podczas przetwarzania odpowiedzi: {e}")
        break
    
    # Pobranie tokenu stronicowania
    try:
        resumption_token_elem = root.find('.//{http://www.openarchives.org/OAI/2.0/}resumptionToken')
        if resumption_token_elem is not None:
            resumption_token = resumption_token_elem.text
            print(f"Pobrano resumptionToken: {resumption_token}")
        else:
            print("Brak resumptionToken. Pobieranie zakończone.")
            break
    except Exception as e:
        print(f"Błąd podczas pobierania resumptionToken: {e}")
        break

# Zapis ostatniej porcji rekordów, jeśli pozostały jakieś
if current_chunk_records:
    save_records_to_file(current_chunk_records, file_count)

    

    

#%% deduplikacja

from pymarc import MARCReader
from tqdm import tqdm

# Inicjalizacja zbioru na unikalne identyfikatory
unique_ids_file1 = set()

# Wczytaj rekordy ze starego pliku i zbuduj zbiór 'records2'
old_file_path = 'D:/Nowa_praca/08.02.2024_marki/fi_fennica_08-02-2024.mrc'

with open(old_file_path, 'rb') as data:
    reader = MARCReader(data)
    for record in tqdm(reader, desc="Przetwarzanie rekordów"):
        try:
            # Pobierz identyfikator z pola '001'
            my001 = record.get_fields('001')
            if my001:  # Jeśli pole '001' istnieje
                record_id = my001[0].value()
                # Usuń pierwszą literę, jeśli istnieje
                if len(record_id) > 1:
                    record_id = record_id[1:]
                # Dodaj przetworzony identyfikator do zbioru
                unique_ids_file1.add(record_id)
        except Exception as e:
            print(f"Error processing record in old file: {e}")
            pass

print(f"Collected {len(unique_ids_file1)} unique IDs (without first letter) from the old file.")
import os
from pymarc import MARCReader
from tqdm import tqdm

# Ścieżka do katalogu z nowymi plikami MARC
file2_directory = 'D:/Nowa_praca/update_fennica/'

# Zbiór ID ze starego pliku (records2)


# Listy na duplikaty i nowe rekordy
# Zbiory do deduplikacji drugiego zbioru
unique_ids_file2 = set()
duplicates_between_sets = []  # Duplikaty między zbiorami
internal_duplicates = []  # Wewnętrzne duplikaty w drugim zbiorze
new_records = []  # Nowe rekordy

# Iteracja przez wszystkie pliki w katalogu
for file_name in os.listdir(file2_directory):
    if file_name.endswith('.mrc'):  # Sprawdzanie plików MARC
        file_path = os.path.join(file2_directory, file_name)
        print(f"Przetwarzanie pliku: {file_name}")

        # Wczytaj rekordy z pliku
        with open(file_path, 'rb') as f:
            reader = MARCReader(f)
            for record in tqdm(reader, desc=f"Przetwarzanie {file_name}"):
                try:
                    field_001 = record.get_fields('001')
                    if field_001:
                        record_id = field_001[0].value()
                        if record_id in unique_ids_file1:
                            # Duplikat między zbiorami
                            duplicates_between_sets.append(record_id)
                        elif record_id in unique_ids_file2:
                            # Duplikat wewnętrzny
                            internal_duplicates.append(record_id)
                        else:
                            # Nowy rekord
                            unique_ids_file2.add(record_id)
                            new_records.append(record)
                except Exception as e:
                    print(f"Błąd przetwarzania rekordu: {e}")
output_file = 'uniqueFennica.mrc'
# Zapis nowych rekordów do pliku wynikowego
with open(output_file, 'wb') as f:
    writer = MARCWriter(f)
    for record in new_records:
        try:
            writer.write(record)
        except Exception as e:
            print(f"Błąd podczas zapisywania rekordu: {e}")
    writer.close()

# Podsumowanie wyników
print(f"Znaleziono {len(duplicates_between_sets)} duplikatów między zbiorami.")
print(f"Znaleziono {len(internal_duplicates)} wewnętrznych duplikatów w drugim zbiorze.")
print(f"Znaleziono {len(new_records)} nowych rekordów i zapisano do {output_file}.")

#%%650 N G

from pymarc import MARCReader, Field, TextWriter, MARCWriter
import pandas as pd
import re
from pymarc import MARCReader, TextWriter, Field, Subfield,MARCWriter
# Wczytanie Excela
excel_path = "D:/Nowa_praca/fennica/all_650_new_karolina.xlsx"
arkusz1 = pd.read_excel(excel_path, sheet_name="fin_do_laczenia")
arkusz2 = pd.read_excel(excel_path, sheet_name="wszystko_karolina")

# Wyciągamy tylko potrzebne kolumny
arkusz1 = arkusz1[['all', 'desk_650']]
arkusz2 = arkusz2[['all', 'KPto650', 'nationalityto650']]

# Łączenie arkuszy na podstawie kolumny "all"
merged = pd.merge(arkusz1, arkusz2, on="all", how="left")
# Funkcja do przetwarzania pola MARC na format zgodny z desk_650
def clean_text(value):
    """
    Usuwa zbędne spacje, kropki i przecinki z początku i końca tekstu.
    """
    if not value:
        return None
    return re.sub(r'[.,]+$', '', value.strip())  # Usuwa kropki/przecinki na końcu i zbędne spacje

# Funkcja do wyciągania podpola 'a'
def extract_subfield_a(value):
    """
    Wyciąga zawartość podpola 'a' z wartości MARC (desk_650 lub pole 650).
    """
    value = re.sub(r'^\\[0-9]*', '', value)  # Usunięcie wskaźnika (\7, \0)
    match = re.search(r'\$a([^$]+)', value)  # Szukanie zawartości podpola 'a'
    if match:
        return clean_text(match.group(1))  # Czyszczenie i zwracanie zawartości podpola 'a'
    return None

# Czyszczenie desk_650 w Excelu
merged['desk_650_normalized'] = merged['desk_650'].apply(lambda x: clean_text(extract_subfield_a(x)))




def is_duplicate_field(record, new_field):
    """
    Sprawdza, czy rekord MARC zawiera już pole identyczne z new_field.
    """
    for field in record.get_fields(new_field.tag):
        # Porównujemy wskaźniki i podpola
        if field.indicators == new_field.indicators and field.subfields == new_field.subfields:
            return True
    return False

# Funkcja przetwarzająca MARC dla danego wskaźnika i kolumny Excela
def process_marc(input_file, output_mrk, output_mrc, merge_column, indicator_value):
    """
    Przetwarza plik MARC, dodając pola 650 na podstawie podanej kolumny Excela.
    """
    with open(input_file, "rb") as marc_file:
        reader = MARCReader(marc_file)

        # Przygotowanie writerów
        with open(output_mrk, "w", encoding="utf-8") as text_output, \
             open(output_mrc, "wb") as binary_output:
            
            text_writer = TextWriter(text_output)
            mrc_writer = MARCWriter(binary_output)

            # Liczniki
            total_records = 0
            records_with_new_fields = 0

            # Iteracja przez rekordy MARC
            for record in reader:
                total_records += 1
                added_new_field = False  # Flaga dla rekordu

                for field in record.get_fields('650'):
                    # Wyciągnięcie podpola 'a'
                    subfield_a = clean_text(extract_subfield_a(str(field)))

                    if subfield_a:
                        # Dopasowanie do Excela
                        nowy_wiersz = merged.loc[merged['desk_650_normalized'] == subfield_a]

                        if not nowy_wiersz.empty:
                            # Obsługa tłumaczeń
                            new_fields = nowy_wiersz[merge_column].dropna().unique()
                            for new_value in new_fields:
                                if new_value:
                                    # Formatowanie i czyszczenie wartości
                                    formatted_value = clean_text(new_value).capitalize()

                                    # Tworzenie pola 650
                                    new_field = Field(
                                        tag='650',
                                        indicators=[' ', ' '],
                                        subfields=[
                                            Subfield('a', formatted_value),
                                            Subfield('2', indicator_value)
                                        ]
                                    )

                                    # Sprawdzanie duplikatów
                                    if not is_duplicate_field(record, new_field):
                                        record.add_ordered_field(new_field)
                                        added_new_field = True

                # Zliczanie rekordów z nowymi polami
                if added_new_field:
                    records_with_new_fields += 1

                # Zapis rekordu
                text_writer.write(record)
                mrc_writer.write(record)

            # Zamknięcie writerów
            text_writer.close()
            mrc_writer.close()

    # Wynik w konsoli
    print(f"{indicator_value}: Total records processed: {total_records}")
    print(f"{indicator_value}: Records with new fields: {records_with_new_fields}")

# Czyszczenie kolumny 'desk_650'
merged['desk_650_normalized'] = merged['desk_650'].apply(lambda x: clean_text(extract_subfield_a(x)))

# Pierwszy proces: ELB-g
process_marc(
    input_file="D:/Nowa_praca/update_fennica/uniqueFennica.mrc",
    output_mrk="wynikowy_elb_g.mrk",
    output_mrc="wynikowy_elb_g.mrc",
    merge_column="KPto650",
    indicator_value="ELB-g"
)

# Drugi proces: ELB-n (bazując na wynikowym pliku z ELB-g)
process_marc(
    input_file="wynikowy_elb_g.mrc",
    output_mrk="wynikowy_elb_n_g.mrk",
    output_mrc="wynikowy_elb_n_g.mrc",
    merge_column="nationalityto650",
    indicator_value="ELB-n")

#%%380
import os
from pymarc import MARCReader, MARCWriter, Field, Subfield

# Pliki wejściowe i wyjściowe
input_file = "C:/Users/darek/wynikowy_elb_n_g.mrc"  # Zmień na ścieżkę do pliku wejściowego MARC
output_file = "fennica_650_380_381.mrc"

# Mapa wartości 084_YKL do nowych pól 380 i 381
yk_to_new_fields = {
    "83": {"380": {"a": "Literature", "i": "Major genre", "l": "eng"}, 
           "381": {"a": "Drama", "i": "Major genre", "l": "eng"}},
    "84": {"380": {"a": "Literature", "i": "Major genre", "l": "eng"}, 
           "381": {"a": "Fiction", "i": "Major genre", "l": "eng"}},
    "80,81,85": {"380": {"a": "Literature", "i": "Major genre", "l": "eng"}, 
                 "381": {"a": "Other", "i": "Major genre", "l": "eng"}},
    "82": {"380": {"a": "Literature", "i": "Major genre", "l": "eng"}, 
           "381": {"a": "Lyrical poetry", "i": "Major genre", "l": "eng"}},
    "86": {"380": {"a": "Secondary literature", "i": "Major genre", "l": "eng"}, 
           "381": None},
}

# Specjalne przypadki dla pola 380
special_380_cases = {
    "arvostelu": {"a": "Secondary literature", "i": "Major genre", "l": "eng"}
}

# Funkcja sprawdzająca duplikaty
def is_duplicate_field(record, new_field):
    """
    Sprawdza, czy rekord MARC zawiera już pole identyczne z new_field.
    """
    for field in record.get_fields(new_field.tag):
        # Porównujemy wskaźniki i podpola
        if field.indicators == new_field.indicators and field.subfields == new_field.subfields:
            return True
    return False

# Funkcja dodająca nowe pola do rekordu
def add_new_fields(record, counters):
    """
    Dodaje nowe pola do rekordu, zliczając dodane pola 380 i 381.
    """
    # Obsługa pola 084_YKL
    for field in record.get_fields("084"):
        if field["a"]:
            field_value = field["a"].strip()  # Oczyszczanie wartości
            for yk_class, new_fields in yk_to_new_fields.items():
                # Sprawdzenie czy wartość 084$a zaczyna się od klucza yk_class
                if any(field_value.startswith(prefix) for prefix in yk_class.split(",")):
                    # Dodaj pole 380
                    if new_fields["380"]:
                        subfields_380 = [
                            Subfield("i", new_fields["380"]["i"]),
                            Subfield("a", new_fields["380"]["a"]),
                            Subfield("l", new_fields["380"]["l"])  # Dodanie języka
                        ]
                        new_field_380 = Field(tag="380", indicators=[" ", " "], subfields=subfields_380)
                        if not is_duplicate_field(record, new_field_380):  # Deduplication
                            record.add_ordered_field(new_field_380)
                            counters["380"] += 1
                    # Dodaj pole 381
                    if new_fields["381"]:
                        subfields_381 = [
                            Subfield("i", new_fields["381"]["i"]),
                            Subfield("a", new_fields["381"]["a"]),
                            Subfield("l", new_fields["381"]["l"])  # Dodanie języka
                        ]
                        new_field_381 = Field(tag="381", indicators=[" ", " "], subfields=subfields_381)
                        if not is_duplicate_field(record, new_field_381):  # Deduplication
                            record.add_ordered_field(new_field_381)
                            counters["381"] += 1
    
    # Obsługa specjalnych przypadków pola 380
    for field in record.get_fields("380"):
        if field["a"] and field["a"] in special_380_cases:
            special_case = special_380_cases[field["a"]]
            subfields_380 = [
                Subfield("i", special_case["i"]),
                Subfield("a", special_case["a"]),
                Subfield("l", special_case["l"])  # Dodanie języka
            ]
            new_field_380 = Field(tag="380", indicators=[" ", " "], subfields=subfields_380)
            if not is_duplicate_field(record, new_field_380):  # Deduplication
                record.add_ordered_field(new_field_380)
                counters["380"] += 1

# Otwórz plik wejściowy i wyjściowy
counters = {"380": 0, "381": 0}  # Liczniki dodanych pól
with open(input_file, "rb") as reader_file, open(output_file, "wb") as writer_file:
    reader = MARCReader(reader_file)
    writer = MARCWriter(writer_file)
    
    for record in reader:
        add_new_fields(record, counters)  # Dodaj nowe pola z deduplikacją
        writer.write(record)   # Zapisz zmodyfikowany rekord

    writer.close()

# Wynik w konsoli
print(f"Rekordy zostały zapisane do pliku: {output_file}")
print(f"Dodano pola 380: {counters['380']}")
print(f"Dodano pola 381: {counters['381']}")



#%% VIAFS
from tqdm import tqdm
#viaf_combination
#tworzenie slowniczka
fields_to_check={}
my_marc_files = ["D:/Nowa_praca/08.02.2024_marki/pbl_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_articles_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_chapters_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles0_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles1_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles2_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles3_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles4_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_books__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_chapters__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/es_articles__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/es_ksiazki__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/fi_arto__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/fi_fennica_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_articles_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_chapters__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/pbl_articles_08-02-2024.mrc"]
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('700','600','100')
            
            
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_a=field.get_subfields('a')
                sub_d=field.get_subfields('d')
                sub_1=field.get_subfields('1')
                if sub_a and sub_d and sub_1:
                    text=''
                    for sub in sub_a+sub_d:
                        
                        for l in sub:
                            
                            if l.isalnum():
                                text=text+l
                    fields_to_check[text]=sub_1[0]



#uzycie slowniczka i dodanie viafow oczywistych                                
my_marc_files =["C:/Users/darek/fennica_650_380_381.mrc"]
counter=0
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'new_viaf.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'new_viaf.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('700','600','100')
            
            
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_a=field.get_subfields('a')
                sub_d=field.get_subfields('d')
                sub_1=field.get_subfields('1')
                if sub_1:
                    continue
                else:
                    #print(field)
                    if sub_a and sub_d:
                        text=''
                        for sub in sub_a+sub_d:
                            
                            for l in sub:
                                
                                if l.isalnum():
                                    text=text+l
                        if text in fields_to_check:
                            counter+=1
                            print(text, fields_to_check[text])
                    
                            field.add_subfield('1', fields_to_check[text])
            #print(record)
            data1.write(record.as_marc())
            writer.write(record)    
writer.close()                    
#%%correct field "d"  
my_marc_files = ["C:/Users/darek/fennica_650_380_381new_viaf.mrc"]
records_double_d=set()
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'d_unify2.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'d_unify2.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            print(record)
            
            
            my = record.get_fields('700','600','100')
            for field in my:
                #print(record['001'].value())
               # field.add_subfield('d', '1989-2022')
                #sub_a=field.get_subfields('a')
                sub_d=field.get_subfields('d')
                if len(sub_d)>1:
                    field.delete_subfield('d')
                    
                    if field['d'][0].isnumeric():
                        field['d']="("+field['d']+")"
                    #records_double_d.add(record['001'].value())
                    
                    
                    continue
                else:
                #field.delete_subfield('d')
                

                    if sub_d:
                        if field['d'][0].isnumeric():
                            field['d']="("+field['d'].strip('.')+")"
                   
            data1.write(record.as_marc()) 
            writer.write(record)    
writer.close() 

from pymarc import MARCReader, MARCWriter, Field, Subfield, Record
import string

def shift_subfield_code(code: str) -> str:
    """
    Funkcja przesuwa literowy kod subpola o 1 pozycję w alfabecie: a->b, b->c, ..., y->z.
    Cyfry i inne znaki (np. '5', '0') pozostają bez zmian.
    Zostawiamy 'z' jako 'z' (możesz ewentualnie zaimplementować zawijanie).
    """
    # Jeżeli kod jest pojedynczą literą (a-z):
    if len(code) == 1 and code.isalpha():
        # Znajdź indeks w alfabecie
        idx = string.ascii_lowercase.find(code.lower())
        if idx == -1:
            # Nie znaleziono w a-z
            return code
        
        # Jeżeli to 'z', zostawiamy jako 'z' (lub zawijaj do 'a' – wg potrzeb)
        if code.lower() == 'z':
            return 'z'
        
        # Przesunięcie o 1
        new_char = string.ascii_lowercase[idx + 1]
        
        # Jeśli kod był wielką literą (raczej w MARC subfields się nie spotyka), to podtrzymujemy
        if code.isupper():
            new_char = new_char.upper()
        return new_char
    else:
        # Jeśli to cyfra lub inny znak, nie zmieniamy
        return code

def modify_or_add_995(file_path, output_path):
    with open(file_path, 'rb') as marc_file, open(output_path, 'wb') as out_file:
        reader = MARCReader(marc_file, to_unicode=True, force_utf8=True)
        writer = MARCWriter(out_file)
        
        for record in reader:
            fields_995 = record.get_fields('995')
            
            if not fields_995:
                #
                # 1. Rekord NIE ma pola 995
                #    -> tworzymy nowe pole 995 i dodajemy subfield 'a' = 'Kansalliskirjasto'
                #
                new_995 = Field(
                    tag='995',
                    indicators=[' ', ' '],
                    subfields=[Subfield(code='a', value='Kansalliskirjasto')]
                )
                record.add_field(new_995)
            else:
                #
                # 2. Rekord ma przynajmniej jedno pole 995
                #    -> pracujemy z PIERWSZYM polem 995 (jeśli trzeba obsłużyć wszystkie – pętla)
                #
                field_995 = fields_995[0]
                subfields_list = field_995.subfields  # to jest lista obiektów Subfield od pymarc 4.x

                # Sprawdzamy, czy istnieje subfield a z wartością "Kansalliskirjasto"
                sub_a_exists = any(
                    (sf.code == 'a' and sf.value == 'Kansalliskirjasto') 
                    for sf in subfields_list
                )

                if not sub_a_exists:
                    #
                    # 2a. Nie ma subfield a="Kansalliskirjasto"
                    #     -> przesuwamy literowe kody subpól, dodajemy nowe subfield 'a'
                    #
                    new_subfields = []
                    
                    # Przesuwamy literowe kody
                    for sf in subfields_list:
                        shifted_code = shift_subfield_code(sf.code)
                        new_subfields.append(Subfield(code=shifted_code, value=sf.value))
                    
                    # Wstawiamy *na początek* nowy subfield a="Kansalliskirjasto"
                    new_subfields.insert(0, Subfield(code='a', value='Kansalliskirjasto'))
                    
                    # Tworzymy nowe pole 995 z przesuniętymi kodami + nowym subfieldem 'a'
                    new_field_995 = Field(
                        tag='995',
                        indicators=field_995.indicators,  # zachowaj oryginalne wskaźniki
                        subfields=new_subfields
                    )
                    
                    # Usuwamy stare pole 995 i dodajemy nowe
                    record.remove_field(field_995)
                    record.add_field(new_field_995)
            
            # Zapisujemy zmodyfikowany rekord
            writer.write(record)
        
        writer.close()

# Przykład wywołania
if __name__ == "__main__":
    input_path = 'C:/Users/darek/fennica_650_380_381new_viafd_unify2.mrc'
    output_path = 'C:/Users/darek/fennica_650_380_381new_viafd_unify+995.mrc'
    modify_or_add_995(input_path, output_path)






# Modyfikacja pliku MARC


# URL API
base_url = 'https://oai-pmh.api.melinda.kansalliskirjasto.fi/bib'

# Parametry ListRecords
start_date = '2024-01-01'
end_date = '2024-12-31'
params = {
    'verb': 'ListRecords',
    'metadataPrefix': 'marc21',  # Format metadanych MARC 21
    'set': 'fennica',            # Wybrany zestaw
    'from': start_date,
    'until': end_date
}

# Wysłanie zapytania
response = requests.get(base_url, params=params)

# Sprawdzanie odpowiedzi
if response.status_code == 200:
    print("URL żądania:", response.url)
    print("\nRekordy z zestawu Fennica w formacie MARC21 (pierwsze 1000 znaków):")
    print(response.text[:1000])  # Wyświetlenie części odpowiedzi
else:
    print(f"Błąd: {response.status_code}")
