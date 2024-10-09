# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:42:23 2024

@author: dariu
"""

from pymarc import MARCReader
from collections import Counter

def count_unique_journals_by_issn(marc_file):
    issn_list = []

    with open(marc_file, 'rb') as fh:
        reader = MARCReader(fh)
        for record in reader:
            # Field 773 is used for host item entry (e.g., the journal in which an article appears)
            if '773' in record:
                fields_773 = record.get_fields('773')
                for field in fields_773:
                    # Subfield 'x' contains the ISSN of the host item
                    if 'x' in field:
                        issn = field['x'].strip()
                        issn_list.append(issn)

    # Count unique ISSNs
    unique_issns = set(issn_list)
    print(f"Total number of journal articles: {len(issn_list)}")
    print(f"Total number of unique journals (by ISSN): {len(unique_issns)}")

    # Optionally, print the count of articles per ISSN
    issn_counts = Counter(issn_list)
    for issn, count in issn_counts.most_common():
        print(f"ISSN {issn}: {count} articles")

# Replace 'your_dataset.mrc' with the path to your MARC file
count_unique_journals_by_issn('D:/Nowa_praca/Espana/periodcs/articles_do_dodania_995.mrc')
from pymarc import MARCReader
import csv

def extract_journals_to_csv(marc_file, output_csv):
    """
    Extracts journal titles and their ISSNs from a MARC21 file and saves them to a CSV file.

    Parameters:
    - marc_file: Path to the input MARC21 file.
    - output_csv: Path to the output CSV file.
    """
    # Używamy słownika do przechowywania unikalnych ISSN i tytułów czasopism
    journals = {}

    # Otwieramy plik MARC21 w trybie binarnym
    with open(marc_file, 'rb') as fh:
        reader = MARCReader(fh)
        for record in reader:
            # Sprawdzamy, czy rekord zawiera pole 773
            if '773' in record:
                for field in record.get_fields('773'):
                    # Wyciągamy ISSN z subfield 'x'
                    issn = field.get('x', '').strip()
                    
                    # Wyciągamy tytuł czasopisma z subfield 't' lub 'a'
                    title = field.get('t', '').strip() or field.get('a', '').strip()
                    
                    # Jeśli oba, ISSN i tytuł, są dostępne, dodajemy do słownika
                    if issn and title:
                        journals[issn] = title
                    # Jeśli tylko tytuł jest dostępny, możemy dodać go bez ISSN
                    elif title:
                        # Używamy ISSN jako pustego ciągu, jeśli brak ISSN
                        journals.setdefault('', set()).add(title)

    # Otwieramy plik CSV do zapisu
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Zapisujemy nagłówki kolumn
        writer.writerow(['ISSN', 'Journal Title'])
        
        # Iterujemy przez słownik i zapisujemy dane do CSV
        for issn, title in sorted(journals.items()):
            if issn:
                writer.writerow([issn, title])
            else:
                # Jeśli ISSN jest pusty, zapisujemy tylko tytuł
                for t in title:
                    writer.writerow(['', t])

    print(f"Extracted {len(journals)} unique journals to {output_csv}")

# Przykładowe użycie funkcji
extract_journals_to_csv(
    'D:/Nowa_praca/08.02.2024_marki/fi_arto__08-02-2024.mrc',
    'finnish_journals.csv'
)


from pymarc import MARCReader
import csv
from tqdm import tqdm

def extract_journals_to_csv(marc_files, output_csv):
    """
    Extracts journal titles and their ISSNs from multiple MARC21 files and saves them to a CSV file.

    Parameters:
    - marc_files: List of paths to the input MARC21 files.
    - output_csv: Path to the output CSV file.
    """
    # Initialize a dictionary to store unique ISSN and journal title pairs
    journals = {}

    # Iterate over each MARC file provided in the list
    for marc_file in marc_files:
        print(f"Processing file: {marc_file}")
        with open(marc_file, 'rb') as fh:
            reader = MARCReader(fh)
            # Wrap the reader with tqdm for progress indication
            for record in tqdm(reader, desc=f"Processing records in {marc_file}", unit="record"):
                # Check if the record contains field 773 (Host Item Entry)
                if '773' in record:
                    for field in record.get_fields('773'):
                        # Extract ISSN from subfield 'x'
                        issn = field.get('x', '').strip()
                        
                        # Extract journal title from subfield 't' or 'a'
                        title = field.get('t', '').strip() or field.get('a', '').strip()
                        
                        # If both ISSN and title are available, add to the dictionary
                        if issn and title:
                            journals[issn] = title
                        # If only title is available, add it without ISSN
                        elif title:
                            # Use ISSN as an empty string if ISSN is missing
                            journals.setdefault('', set()).add(title)

    # Open the CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['ISSN', 'Journal Title'])
        
        # Iterate through the journals dictionary and write each entry to the CSV
        for issn, titles in sorted(journals.items()):
            if issn:
                writer.writerow([issn, titles])
            else:
                # If ISSN is missing, write only the journal title(s)
                for title in titles:
                    writer.writerow(['', title])

    print(f"Extracted {len(journals)} unique journals to {output_csv}")
    return journals
# Example usage
marc_files = [
    'D:/Nowa_praca/08.02.2024_marki/es_ksiazki__08-02-2024.mrc',
    'D:/Nowa_praca/08.02.2024_marki/es_articles__08-02-2024.mrc'  # Replace with your second MARC file path
]

output_csv = 'finnish_journals_combined.csv'

# Call the function with the list of MARC files and the desired output CSV file name
journals=extract_journals_to_csv(marc_files, output_csv)
