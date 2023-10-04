# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:11:30 2023

@author: dariu
"""

from pymarc import MARCReader
from tqdm import tqdm
from pymarc import MARCReader,JSONReader
from tqdm import tqdm
from pymarc import Record, Field, Subfield
import requests
import json
from pymarc import MARCReader
from pymarc import parse_json_to_array
from pymarc import TextWriter
from pymarc import XMLWriter
from pymarc import JSONWriter
from io import BytesIO
import warnings
from pymarc import MARCReader
from pymarc import Record, Field 
import pandas as pd
from definicje import *
#first delete all major genre
from pymarc import MARCReader, MARCWriter, Field

with open('C:/Users/dariu/proba.mrc', 'rb') as fh:
    reader = MARCReader(fh)
    with open('zmodyfikowany_plik2.mrc', 'wb') as out:
        writer = MARCWriter(out)

        for record in reader:
            # Sprawdzenie pól 380 i 381
            for tag in ['380', '381']:
                fields_to_delete = []
                for field in record.get_fields(tag):
                    subfield_i = field.get_subfields('i')
                    # Jeśli podpole 'i' istnieje i ma wartość "Major genre", dodaj pole do listy do usunięcia
                    if subfield_i and 'Major genre' in subfield_i[0]:
                        fields_to_delete.append(field)

                # Usuń wybrane pola z rekordu
                for field in fields_to_delete:
                    record.remove_field(field)

            # Zapisz zmodyfikowany rekord do nowego pliku
            writer.write(record)

        writer.close()




field655=pd.read_excel('D:/Nowa_praca/655_381_380_excele/29.08.2023nowe_BN_655.xlsx', sheet_name='Arkusz2',dtype=str)
listy655=dict(zip(field655['desk655'].to_list(),field655['action1'].to_list()))
genre655=dict(zip(field655['desk655'].to_list(),field655['action2'].to_list()))
field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/29.08.2023_literature_nationalities.xlsx', sheet_name='Sheet1',dtype=str)
dictionary_to_check=dict(zip(field650['dane oryginalne'].to_list(),field650['narodowosc'].to_list()))



my_marc_files=["C:/Users/dariu/zmodyfikowany_plik.mrc"]

# Initialize the final dictionary
dictionary_final = {}
secondary_and_literature=[]
literature=[]
secondary=[]
other=[]
new_categories_tocheck=set()
# Iterate through MARC files
for my_marc_file in tqdm(my_marc_files):
    with open(my_marc_file, 'rb') as file, open('zmodyfikowany_plikver2.mrc', 'wb') as out:
        reader = MARCReader(file)
        for record in tqdm(reader):
            #field_008 = record['008'].data

            # Extract year and language from field_008
            # year_str = field_008[7:11]
            # language = field_008[35:38]

            # # Check if the year is within the range and in Polish
            
            # #if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
            year = int(1)
            fields_655 = record.get_fields('655')
            fields_650 = record.get_fields('650')
            fields_386 = record.get_fields('386')
         
            nationalities_from_655 = []
            nationalities_from_650 = []
            nationalities_from_386 = []

            # Extract nationalities from different fields
            for field in fields_386:
                subfields_a = field.get_subfields('a')
                for subfield in subfields_a:
                    if subfield in dictionary_to_check:
                        nationalities_from_386.append(dictionary_to_check[subfield])

            for field in fields_655:
                subfields_a = field.get_subfields('a')
                for subfield in subfields_a:
                    if subfield in dictionary_to_check:
                        nationalities_from_655.append(dictionary_to_check[subfield])

            for field in fields_650:
                subfields_a = field.get_subfields('a')
                for subfield in subfields_a:
                    if subfield in dictionary_to_check:
                        nationalities_from_650.append(dictionary_to_check[subfield])
            # Initialize set to store categories for each record
            categories_for_record = set()
            starts_with_artykul = False
            found_in_listy655 = False
            
            for field in fields_655:
                subfields_a = field.get_subfields('a')
                subfield_x = field.get_subfields('x')
                subfield_2 = field.get_subfields('2')
                subfield_y = field.get_subfields('y')
                if (len(subfields_a) > 0 and len(subfield_2) > 0 and len(subfield_x) == 0 and len(subfield_y) == 0) or (len(subfields_a) > 0 and len(subfield_2) == 0 and len(subfield_x) == 0 and len(subfield_y) == 0) :
# Do things when only 'a' and '2' are present
                    for subfield in subfields_a:
                        subfield=subfield.strip()
                        
                       # Check first condition: starts with "artykuł z czasopisma"
                        if subfield.startswith("Artykuł z czasopisma"):
                           starts_with_artykul = True
                           
                
                       # Check second condition: is in listy655
                        if listy655.get(subfield) == "Literature":
                           found_in_listy655 = True
                           categories_for_record.add("literature")
                           categories_for_record.add(genre655.get(subfield))
                           if genre655.get(subfield)==None:
                               new_categories_tocheck.add(subfield)
                           
                           
                        elif listy655.get(subfield) == "Secondary literature":
                            categories_for_record.add("secondary")
                            
                         
                            
                        elif listy655.get(subfield) == "Other":
                            categories_for_record.add("other")
                        # elif subfield in listy655:
                        #     categories_for_record.add("literature")
                    continue  # Exit the loop after processing                    
                
                if subfield_x:
                    categories_for_record.add("secondary")
                    continue
                     
                elif len(subfield_y) > 0 and len(subfield_x) == 0:
                    is_secondary = False 
                
                    for subfield in subfields_a:
                        if subfield.startswith("Szkice literackie") or subfield.startswith("Publicystyka"):
                            is_secondary = True  # Set the flag
                    if is_secondary:
                        categories_for_record.add("secondary")
                    else:
                        categories_for_record.add("literature")
                        for subfield in subfields_a:
                            subfield=subfield.strip()
                            if listy655.get(subfield) == "Other":
                                continue
                            else:
                                categories_for_record.add(genre655.get(subfield))
                                if genre655.get(subfield)==None:
                                    new_categories_tocheck.add(subfield)
                    continue
                    

                else:
                    categories_for_record.add("other")
            if starts_with_artykul and found_in_listy655:
                
                categories_for_record.remove('secondary')
                
                    
            final_category = None
            #usuwamy None jesli elem nie bylo w slowniku:
            categories_for_record.discard(None)
            #jesli mamy secondary i cos jeszcze ale bez okreslenia dokladnego gatunku to cos jeszcze usuwamy
            if 'secondary' in categories_for_record and len(categories_for_record)==2:
                categories_for_record.discard('literature')
                categories_for_record.discard('other')
            #jesli jest sama literatura bez dokladnego okreslenia usuwam:
            elif 'literature' in categories_for_record and len(categories_for_record)==1:
                categories_for_record.discard('literature')
            if categories_for_record:    
                if "secondary" in categories_for_record and "literature" in categories_for_record:
                    final_category = "secondary_and_literature"
                elif "secondary" in categories_for_record:
                    final_category = "secondary"
                elif "literature" in categories_for_record:
                    final_category = "literature"
                elif "other" in categories_for_record and len(categories_for_record) == 1:
                    final_category = "other"
                
                if final_category == "secondary_and_literature":
                    record.add_ordered_field(
                        Field(
                            tag='380',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield('i', 'Major genre'),
                                Subfield('a', 'Secondary literature'),
                                Subfield('l', 'eng')
                            ]
                        )
                    )
                    record.add_ordered_field(
                        Field(
                            tag='380',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield('i', 'Major genre'),
                                Subfield('a', 'Literature'),
                                Subfield('l', 'eng')
                            ]
                        ))
                    
                    categories_for_record.remove('secondary')
                    categories_for_record.remove('literature')
                    for category in categories_for_record:
                        record.add_ordered_field(
                            Field(
                                tag='381',
                                indicators=['\\', '\\'],
                                subfields=[
                                    Subfield('i', 'Major genre'),
                                    Subfield('a', category),
                                    Subfield('l', 'eng')
                                ]
                            ))
                        
                
                elif final_category == "literature":
                    record.add_ordered_field(
                        Field(
                            tag='380',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield('i', 'Major genre'),
                                Subfield('a', 'Literature'),
                                Subfield('l', 'eng')
                            ]
                        ))
                    
                    categories_for_record.remove('literature')
                    for category in categories_for_record:
                        record.add_ordered_field(
                            Field(
                                tag='381',
                                indicators=['\\', '\\'],
                                subfields=[
                                    Subfield('i', 'Major genre'),
                                    Subfield('a', category),
                                    Subfield('l', 'eng')
                                ]
                            ))
                
                
                elif final_category == "secondary":
                    record.add_ordered_field(
                        Field(
                            tag='380',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield('i', 'Major genre'),
                                Subfield('a', 'Secondary literature'),
                                Subfield('l', 'eng')
                            ]
                        )
                    )
                    categories_for_record.remove('secondary')
    
                    nationalities = list(set(nationalities_from_655 + nationalities_from_386 + nationalities_from_650))
                    for nationality in nationalities:
                        if nationality not in dictionary_final:
                            dictionary_final[nationality] = {}
                        if year not in dictionary_final[nationality]:
                            dictionary_final[nationality][year] = {
                                "literature": 0, 
                                "secondary": 0, 
                                "secondary_and_literature": 0,
                                "other": 0  # Adding an entry for 'other' as per your categories
                            }
                        
                        # Increment 'secondary' if no fields_655 are present and the nationality is defined
                    if not fields_655 and nationality: 
                        dictionary_final[nationality][year]["secondary"] += 1
                        record.add_ordered_field(
                            Field(
                                tag='380',
                                indicators=['\\', '\\'],
                                subfields=[
                                    Subfield('i', 'Major genre'),
                                    Subfield('a', 'Secondary literature'),
                                    Subfield('l', 'eng')
                                ]
                            )
                        )
            out.write(record.as_marc())        
                    # Update the count for the final category in the dictionary
                    # if final_category:
                    #     dictionary_final[nationality][year][final_category] += 1
                
                    # # If the nationality is "austrian literature" and the final_category is "literature", 
                    # # add the record to the all_selected list
                    # if nationality == "austrian literature" and final_category == "literature":
                    #     literature.append(record)
                    # if nationality == "american literature" and final_category == "other":
                    #     other.append(record)
                    # if nationality == "austrian literature" and final_category == "secondary":
                    #     secondary.append(record)
                    # if nationality == "austrian literature" and final_category == "secondary_and_literature":
                    #     secondary_and_literature.append(record)
                        
                    # if not fields_655 and nationality=="austrian literature": 
                        # secondary.append(record)    
with open('four_categories_new data_from_Czarek_files.json', 'w', encoding='utf-8') as f:
    json.dump(dictionary_final, f, ensure_ascii=False) 
    
    
new=set()

new.add(Field(

        tag = '380', 

        indicators = ['\\','\\'],

        subfields = [
                       Subfield('i', 'Major genre'),
                        Subfield('a', 'Secondary literature'),
                        Subfield('l', 'eng'),
                    ]
        )) 
Field(

        tag = '380', 

        indicators = ['\\','\\'],

        subfields = [
                       Subfield('i', 'Major genre'),
                        Subfield('a', 'Secondary literature'),
                        Subfield('l', 'eng'),
                    ]
        ) 

lista=['d','d','a']
deduplicated_list = []
seen = []

for item in lista:
    if item not in seen:
        deduplicated_list.append(item)
        seen.append(item)
