# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:52:15 2023

@author: dariu
"""

from pymarc import MARCReader, MARCWriter, Field, Subfield

def move_x_after_t_in_773(record):
    field_773 = record.get('773', None)

    # If the record doesn't have a 773 field, return as is
    if not field_773:
        return record

    x_subfield = next((subfield for subfield in field_773.subfields if subfield.code == 'x'), None)
    t_subfield = next((subfield for subfield in field_773.subfields if subfield.code == 't'), None)

    # If either subfield x or t doesn't exist, return the record as is
    if not x_subfield or not t_subfield:
        return record

    # Remove the $x subfield
    field_773.subfields.remove(x_subfield)

    # Find the position of the $t subfield and insert the $x subfield after it
    t_position = field_773.subfields.index(t_subfield)
    field_773.subfields.insert(t_position + 1, x_subfield)

    return record

# Function to process a MARC file
def process_marc_file(input_path, output_path):
    with open(input_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
        reader = MARCReader(input_file)
        writer = MARCWriter(output_file)

        for record in reader:
            modified_record = move_x_after_t_in_773(record)
            writer.write(modified_record)

        writer.close()
def process_marc_files(file_list):
    for file_path in file_list:
        # For demonstration purposes, I'm assuming you want to overwrite each file with its processed data.
        # If you'd like a different output naming convention, adjust the output_path accordingly.
        output_path = file_path.replace('.mrc', '_processed.mrc')
        process_marc_file(file_path, output_path)

# Call the function specifying the input and output file paths
#process_marc_file('C:/Users/dariu/773_proba.mrc', 'path_to_output.mrc')
files_to_process = ["D:/Nowa_praca/21082023_nowe marki nowy viaf/sp_ksiazki_composed_unify2_do_wyslanianew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_articles_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_books_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_chapters_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles0_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles1_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles2_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles3_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles4_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_books_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_chapters_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/es_articles_sorted_31.05.2023.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_arto_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_fennica_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_articles_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_books_21-02-2023composenew_viafnew_viaf.mrc"]
process_marc_files(files_to_process)