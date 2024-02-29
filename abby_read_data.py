# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 09:29:24 2024

@author: dariu
"""

import xml.etree.ElementTree as ET

# Ścieżka do Twojego pliku XML
file_path = 'D:/Nowa_praca/synopsisSJ_abbyy.xml'

# Otwieranie i wczytywanie pliku XML
tree = ET.parse(file_path)
root = tree.getroot()

# Definiujemy przestrzeń nazw, aby można było korzystać z niej podczas wyszukiwania
namespaces = {'ns': 'http://www.abbyy.com/FineReader_xml/FineReader6-schema-v1.xml'}

# Wyszukiwanie wszystkich bloków tekstowych
text_blocks = root.findall('.//ns:block[@blockType="Text"]', namespaces)

for block in text_blocks[:2]:
    # Dla każdego bloku tekstowego wyszukujemy paragrafy
    paragraphs = block.findall('.//ns:text/ns:par', namespaces)
    for par in paragraphs:
        # Dla każdego paragrafu wyszukujemy linie
        lines = par.findall('.//ns:line', namespaces)
        for line in lines:
            # Dla każdej linii konstruujemy tekst z charParams
            text_line = ''.join(char.text for char in line.findall('.//ns:charParams', namespaces))
            print(text_line)
            
            
#lepsza wersja dla Czarka:         
            
block_counter = 1
for block in text_blocks[:13]:  # Przetwarzamy tylko dwa pierwsze bloki
    print(f"--- Blok Tekstowy {block_counter} ---")
    # Dla każdego bloku tekstowego wyszukujemy paragrafy
    paragraph_counter = 1
    paragraphs = block.findall('.//ns:text/ns:par', namespaces)
    for par in paragraphs:
        print(f"  Paragraf {paragraph_counter}:")
        # Dla każdego paragrafu wyszukujemy linie
        line_counter = 1
        lines = par.findall('.//ns:line', namespaces)
        for line in lines:
            # Konstruujemy tekst z charParams dla każdej linii
            text_line = ''.join([char.text if char.text is not None else '' for char in line.findall('.//ns:charParams', namespaces)])
            print(f"    Linia {line_counter}: {text_line}")
            line_counter += 1
        paragraph_counter += 1
    block_counter += 1
    
    
    
    
    
page_counter = 1
for page in root.findall('.//ns:page', namespaces)[1:2]:
    print(f"--- Strona {page_counter} ---")
    block_counter = 1
    text_blocks = page.findall('.//ns:block[@blockType="Text"]', namespaces)
    for block in text_blocks:
        print(f"  --- Blok Tekstowy {block_counter} ---")
        paragraph_counter = 1
        paragraphs = block.findall('.//ns:text/ns:par', namespaces)
        for par in paragraphs:
            print(f"    Paragraf {paragraph_counter}:")
            line_counter = 1
            lines = par.findall('.//ns:line', namespaces)
            for line in lines:
                text_line = ''.join([char.text if char.text is not None else '' for char in line.findall('.//ns:charParams', namespaces)])
                print(f"      Linia {line_counter}: {text_line}")
                line_counter += 1
            paragraph_counter += 1
        block_counter += 1
    page_counter += 1
    
    
    
page = root.find('.//ns:page', namespaces)  # Znajdź tylko pierwszą stronę
if page is not None:
    print("--- Strona 1 ---")
    block_counter = 1
    text_blocks = page.findall('.//ns:block[@blockType="Text"]', namespaces)
    for block in text_blocks:
        print(f"  --- Blok Tekstowy {block_counter} ---")
        paragraph_counter = 1
        paragraphs = block.findall('.//ns:text/ns:par', namespaces)
        for par in paragraphs:
            print(f"    Paragraf {paragraph_counter}:")
            line_counter = 1
            lines = par.findall('.//ns:line', namespaces)
            for line in lines:
                text_line = ''.join([char.text if char.text is not None else '' for char in line.findall('.//ns:charParams', namespaces)])
                print(f"      Linia {line_counter}: {text_line}")
                line_counter += 1
            paragraph_counter += 1
        block_counter += 1
