# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:42:49 2024

@author: dariu
"""

from pymarc import XmlHandler, map_xml,parse_xml,MARCWriter,TextWriter
from tqdm import tqdm



# Przykładowe użycie do drukowania rekordów
file_path = 'C:/Users/dariu/aut_ph.xml'

parserxml=XmlHandler()
def do_it(r):
    r['001'].data='dupa'
    print(r)
    

output_file='wysrany.mrk'
def do_it(record):
    record['001'].data = 'dupa'
    print(record)
    writer.write(record)

with open(output_file, 'wt', encoding="utf-8") as marc_file:
    writer = TextWriter(marc_file)    
    map_xml(do_it, file_path)
     


            
marc_file='dupa'

parserxml=XmlHandler()
parse_xml(file_path, parserxml)    
chunk_size = 10000
counter = 0
file_count = 1

for record in parserxml.records:
    if counter % chunk_size == 0 and counter > 0:
        # Zapisywanie rekordów do pliku co 10000 rekordów
        marc_file_name = f"{marc_file}_{file_count}.mrk"
        with open(marc_file_name, 'wt', encoding="utf-8") as marc_file1:
            writer = TextWriter(marc_file1)
            for rec in parserxml.records[counter - chunk_size:counter]:
                writer.write(rec)
            writer.close()
        file_count += 1
    counter += 1

# Zapisywanie pozostałych rekordów
if counter % chunk_size != 0:
    marc_file_name = f"{marc_file}_{file_count}.mrk"
    with open(marc_file_name, 'wt', encoding="utf-8") as marc_file1:
        writer = TextWriter(marc_file1)
        for rec in parserxml.records[counter - (counter % chunk_size):counter]:
            writer.write(rec)
        writer.close()
    
    

