# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:31:36 2023

@author: dariu
"""


from pymarc import MARCReader, TextWriter
from tqdm import tqdm

my_marc_files = ["D:/Nowa_praca/21082023_nowe marki nowy viaf/sp_ksiazki_composed_unify2_do_wyslanianew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_books_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_articles_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_fennica_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_arto_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/es_articles_sorted_31.05.2023_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_chapters_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_books_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles4_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles3_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles2_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles1_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles0_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_chapters_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_books_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_articles_21-02-2023composenew_viafnew_viaf_processed.mrc"]  # Your list of file paths goes here

for my_marc_file in tqdm(my_marc_files):
    counter = 0  # Initialize counter for each file
    writer = TextWriter(open(my_marc_file + '_sample.mrk', 'wt', encoding="utf-8"))

    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        
        for record in tqdm(reader):
            # Increment counter for each record read
            counter += 1

            # Save the record to the new MARC text file
            writer.write(record)
            
            # If counter hits 100, break out of the loop for the current file
            if counter >= 150:
                break

    # Close the TextWriter for the current file
    writer.close()
    
    
from tqdm import tqdm
import ijson
import json
import jellyfish
from definicje import *
with open ('C:/Users/dariu/Desktop/BNvsBibNauki/28062023results_final_all.json', 'r', encoding='utf-8') as f:
    nikodem=json.load(f)
import json

# Load the JSON data
with open('C:/Users/dariu/Desktop/BNvsBibNauki/bibnau_bn_final_results.json', 'r',encoding='utf-8') as f:
    data = json.load(f)

# Take first 5 records
first_five_records = {k: v for i, (k, v) in enumerate(data.items()) if i < 150}
for k, v in data.items():
    unique(v[0][1])
# Extract the first 60,000 key-value pairs
first_60000_records = {k: v for i, (k, v) in enumerate(data.items()) if i < 60000}

# Extract the rest of the key-value pairs
remaining_records = {k: v for i, (k, v) in enumerate(data.items()) if i >= 60000}
  
# Save the first 5 records to another JSON file
with open('Linked_metadata_of_the_Science_ Library_and_the_ National_Library_part2', 'w', encoding='utf-8') as f:
    json.dump(remaining_records, f, ensure_ascii=False, indent=4)
    

    
with open('try.json', 'r',encoding='utf-8') as f:
    data = json.load(f)
    

