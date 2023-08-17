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
process_marc_file('C:/Users/dariu/773_proba.mrc', 'path_to_output.mrc')
files_to_process = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_chapters_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_libri_marc_bn_articles_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc"]
process_marc_files(files_to_process)