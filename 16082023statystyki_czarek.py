# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:38:02 2023

@author: dariu
"""
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
field655=pd.read_excel('D:/Nowa_praca/655_381_380_excele/29.08.2023nowe_BN_655.xlsx', sheet_name='Arkusz2',dtype=str)
listy655=dict(zip(field655['desk655'].to_list(),field655['action1'].to_list()))
field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/29.08.2023_literature_nationalities.xlsx', sheet_name='Sheet1',dtype=str)
dictionary_to_check=dict(zip(field650['dane oryginalne'].to_list(),field650['narodowosc'].to_list()))


dictionary_final = {}

with open('C:/Users/dariu/Downloads/bibs-all (1).marc', 'rb') as file:
    reader = MARCReader(file)
    for record in tqdm(reader):
        field_008 = record['008'].data

        # Extract year and language from field_008
        year_str = field_008[7:11]
        language = field_008[35:38]

        # Check if the year is between 1989 and 2023 and is all digits
        if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
            year = int(year_str)
            fields_655 = record.get_fields('655')

            for field in fields_655:
                subfields = field.get_subfields('a')
                for subfield in subfields:
                    if subfield in dictionary_to_check:
                        nationality = dictionary_to_check[subfield]
                        
                        if nationality not in dictionary_final:
                            dictionary_final[nationality] = {}
                        
                        if year not in dictionary_final[nationality]:
                            dictionary_final[nationality][year] = {"literature": 0, "secondary": 0}

                        if listy655.get(subfield) == "Secondary literature":
                            dictionary_final[nationality][year]["secondary"] += 1
                        else:
                            dictionary_final[nationality][year]["literature"] += 1                           
                            
#check nationality in 650 then rest in 655:
dictionary_final = {}

with open('C:/Users/dariu/Downloads/bibs-all (1).marc', 'rb') as file:
    reader = MARCReader(file)
    for record in reader:
        field_008 = record['008'].data

        # Extract year and language from field_008
        year_str = field_008[7:11]
        language = field_008[35:38]

        # Check if the year is between 1989 and 2023 and is all digits
        if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
            year = int(year_str)
            fields_655 = record.get_fields('655')
            fields_650 = record.get_fields('650')
            
            # Extract nationalities from field 650
            nationalities = []
            for field in fields_650:
                subfields = field.get_subfields('a')
                for subfield in subfields:
                    if subfield in dictionary_to_check:
                        nationalities.append(dictionary_to_check[subfield])

            for nationality in nationalities:
                if nationality not in dictionary_final:
                    dictionary_final[nationality] = {}

                if year not in dictionary_final[nationality]:
                    dictionary_final[nationality][year] = {"literature": 0, "secondary": 0}

                for field in fields_655:
                    subfields = field.get_subfields('a')
                    for subfield in subfields:
                        if listy655.get(subfield) == "Secondary literature":
                            dictionary_final[nationality][year]["secondary"] += 1
                        else:
                            dictionary_final[nationality][year]["literature"] += 1

#check nationality in 650 and in 655 than the rest (secondary, literature) in 655:
dictionary_final = {}

with open('C:/Users/dariu/Downloads/bibs-all (1).marc', 'rb') as file:
    reader = MARCReader(file)
    for record in tqdm(reader):
        field_008 = record['008'].data

        # Extract year and language from field_008
        year_str = field_008[7:11]
        language = field_008[35:38]

        # Check if the year is between 1989 and 2023 and is all digits
        if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
            year = int(year_str)
            nationalities = []
            
            # Collect nationalities from 655 field
            fields_655 = record.get_fields('655')
            for field in fields_655:
                subfields = field.get_subfields('a')
                for subfield in subfields:
                    if subfield in dictionary_to_check:
                        nationalities.append(dictionary_to_check[subfield])

            # Collect nationalities from 650 field as well
            fields_650 = record.get_fields('650')
            for field in fields_650:
                subfields = field.get_subfields('a')
                for subfield in subfields:
                    if subfield in dictionary_to_check:
                        nationalities.append(dictionary_to_check[subfield])

            # Process the found nationalities
            for nationality in set(nationalities):  # Using set to ensure no duplicates
                if nationality not in dictionary_final:
                    dictionary_final[nationality] = {}

                if year not in dictionary_final[nationality]:
                    dictionary_final[nationality][year] = {"literature": 0, "secondary": 0}

                for field in fields_655:
                    subfields = field.get_subfields('a')
                    for subfield in subfields:
                        if listy655.get(subfield) == "Secondary literature":
                            dictionary_final[nationality][year]["secondary"] += 1
                        else:
                            dictionary_final[nationality][year]["literature"] += 1

#CHECKING ONLY 655 but with x to get secondary
'''only 655 with x:

    For each 655 field, we'll get the value of subfield x.
    If the value of subfield x matches either "tematyka" or "historia", we'll increment the "secondary" counter.
    If not, we'll check listy655 to decide between "literature" and "secondary" counters, as before.'''

dictionary_final = {}

with open('C:/Users/dariu/Downloads/bibs-all (1).marc', 'rb') as file:
    reader = MARCReader(file)
    for record in tqdm(reader):
        field_008 = record['008'].data

        # Extract year and language from field_008
        year_str = field_008[7:11]
        language = field_008[35:38]

        # Check if the year is between 1989 and 2023 and is all digits
        if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
            year = int(year_str)
            fields_655 = record.get_fields('655')

            for field in fields_655:
                subfields_a = field.get_subfields('a')
                subfield_x = field.get_subfields('x')
                
                for subfield in subfields_a:
                    if subfield in dictionary_to_check:
                        nationality = dictionary_to_check[subfield]

                        if nationality not in dictionary_final:
                            dictionary_final[nationality] = {}

                        if year not in dictionary_final[nationality]:
                            dictionary_final[nationality][year] = {"literature": 0, "secondary": 0}

                        # Check if subfield x contains "tematyka" or "historia"
                        if any(value in ["tematyka", "historia"] for value in subfield_x):
                            dictionary_final[nationality][year]["secondary"] += 1
                        elif listy655.get(subfield) == "Secondary literature":
                            dictionary_final[nationality][year]["secondary"] += 1
                        else:
                            dictionary_final[nationality][year]["literature"] += 1


#"approach with checking 'x' but also checking 650  and ???last change in last lines: elif subfield in listy655???:"
#there can be zeros when there is no 655
dictionary_final = {}

with open('C:/Users/dariu/Downloads/bibs-all (1).marc', 'rb') as file:
    reader = MARCReader(file)
    for record in tqdm(reader):
        field_008 = record['008'].data

        # Extract year and language from field_008
        year_str = field_008[7:11]
        language = field_008[35:38]

        # Check if the year is between 1989 and 2023 and is all digits
        if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
            year = int(year_str)
            fields_655 = record.get_fields('655')
            fields_650 = record.get_fields('650')

            nationalities = []  # Using a list to handle multiple nationalities

            # Find nationality in 655
            for field in fields_655:
                subfields_a = field.get_subfields('a')
                for subfield in subfields_a:
                    if subfield in dictionary_to_check:
                        nationalities.append(dictionary_to_check[subfield])

            # If nationality is not found in 655, check 650
            if not nationalities:
                for field in fields_650:
                    subfields_a = field.get_subfields('a')
                    for subfield in subfields_a:
                        if subfield in dictionary_to_check:
                            nationalities.append(dictionary_to_check[subfield])

            # If nationality was identified either in 655 or 650
            for nationality in set(nationalities):
                if nationality not in dictionary_final:
                    dictionary_final[nationality] = {}

                if year not in dictionary_final[nationality]:
                    dictionary_final[nationality][year] = {"literature": 0, "secondary": 0}

                for field in fields_655:
                    subfields_a = field.get_subfields('a')
                    subfield_x = field.get_subfields('x')
                    for subfield in subfields_a:
                        # Check if subfield x contains "tematyka" or "historia"
                        if any(value in ["tematyka", "historia"] for value in subfield_x):
                            dictionary_final[nationality][year]["secondary"] += 1
                        elif listy655.get(subfield) == "Secondary literature":
                            dictionary_final[nationality][year]["secondary"] += 1
                        # elif subfield in listy655:  # Added this condition to ensure subfield exists in listy655
                        #     dictionary_final[nationality][year]["literature"] += 1
                        else:
                            dictionary_final[nationality][year]["literature"] += 1

#LAST BEST FOR ME (650 and 655) when there is no 655 field and we have nationality from 650 count this as our secondary
'''    Extract nationality from both 650 and 655 fields.
    Prioritize the classification of records based on the 655 field.
    Only use the nationality from the 650 field if there is no nationality in the 655 field.
    
                        else:
                            dictionary_final[nationality][year]["literature"] += 1
                        # elif subfield in listy655:
                       #     dictionary_final[nationality][year]["literature"] += 1
                       
                       ultimetly i added undefine'''
                       
my_marc_files=["D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_chapters_2023-08-07.mrc",
"D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_articles_2023-08-29.mrc",
"D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_books_2023-08-29.mrc"]
dictionary_final = {}
for my_marc_file in tqdm(my_marc_files):
    with open(my_marc_file, 'rb') as file:
        reader = MARCReader(file)
        for record in tqdm(reader):
            field_008 = record['008'].data
    
            # Extract year and language from field_008
            year_str = field_008[7:11]
            language = field_008[35:38]
    
            # Check if the year is between 1989 and 2023 and is all digits
            if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
                year = int(year_str)
                fields_655 = record.get_fields('655')
                fields_650 = record.get_fields('650')
                fields_386 = record.get_fields('386')
    
                nationalities_from_655 = []
                nationalities_from_650 = []
                nationalities_from_386 = []
    
               
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
    
                # Find nationality in 650 if not found in 655
                if not nationalities_from_655 or nationalities_from_386:
                    for field in fields_650:
                        subfields_a = field.get_subfields('a')
                        for subfield in subfields_a:
                            if subfield in dictionary_to_check:
                                nationalities_from_650.append(dictionary_to_check[subfield])
    
                # Combine nationalities found
                nationalities = nationalities_from_655 or nationalities_from_386 or nationalities_from_650
    
                for nationality in set(nationalities):
                    if nationality not in dictionary_final:
                        dictionary_final[nationality] = {}
    
                    if year not in dictionary_final[nationality]:
                        dictionary_final[nationality][year] = {"literature": 0, "secondary": 0}
                        
                    for field in fields_655:
                        subfields_a = field.get_subfields('a')
                        subfield_x = field.get_subfields('x')
                        for subfield in subfields_a:
                            # Check if subfield x contains "tematyka" or "historia"
                            if any(value in ["tematyka", "historia"] for value in subfield_x):
                                dictionary_final[nationality][year]["secondary"] += 1
                            elif listy655.get(subfield) == "Secondary literature":
                                dictionary_final[nationality][year]["secondary"] += 1
                            # else:
                            #     dictionary_final[nationality][year]["literature"] += 1
    #Alternative end after (instead of last else, maybe add undefined beacuse it is literature for sure based on disctionary to check?):
                            elif subfield in listy655:
                                dictionary_final[nationality][year]["literature"] += 1
                #optional:
                            # else:
                            #      dictionary_final[nationality][year]["undefined"] += 1
    
                    # If nationality is from 650 and there's no 655, count as secondary
                    if not fields_655 and nationality in nationalities_from_650:
                        dictionary_final[nationality][year]["secondary"] += 1
with open('all_new data_from_Czarek_files.json', 'w', encoding='utf-8') as f:
    json.dump(dictionary_final, f, ensure_ascii=False)





#NEW CODE 3 categories
# Initialize dictionary_final
from pymarc import MARCReader
from tqdm import tqdm

my_marc_files=["D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_chapters_2023-08-07.mrc",
"D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_articles_2023-08-29.mrc",
"D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_books_2023-08-29.mrc"]

# Initialize the final dictionary
dictionary_final = {}
all_selected=[]
# Iterate through MARC files
for my_marc_file in tqdm(my_marc_files):
    with open(my_marc_file, 'rb') as file:
        reader = MARCReader(file)
        for record in tqdm(reader):
            field_008 = record['008'].data

            # Extract year and language from field_008
            year_str = field_008[7:11]
            language = field_008[35:38]

            # Check if the year is within the range and in Polish
            if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
                year = int(year_str)
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

                for field in fields_655:
                    subfields_a = field.get_subfields('a')
                    subfield_x = field.get_subfields('x')
                    
                    if subfield_x:
                        categories_for_record.add("secondary")
                        continue
                    for subfield in subfields_a:
                        if listy655.get(subfield) == "Secondary literature":
                            categories_for_record.add("secondary")
                        elif subfield in listy655:
                            categories_for_record.add("literature")

                # Determine the final category
                final_category = None
                if "secondary" in categories_for_record and "literature" in categories_for_record:
                    final_category = "secondary_and_literature"
                elif "secondary" in categories_for_record:
                    final_category = "secondary"
                elif "literature" in categories_for_record:
                    final_category = "literature"

                # Update dictionary_final
                nationalities = list(set(nationalities_from_655 + nationalities_from_386 + nationalities_from_650))
                for nationality in nationalities:
                    if nationality not in dictionary_final:
                        dictionary_final[nationality] = {}
                    if year not in dictionary_final[nationality]:
                        dictionary_final[nationality][year] = {"literature": 0, "secondary": 0, "secondary_and_literature": 0}
                    
                    if not fields_655 and nationality: #in nationalities_from_650:
                        dictionary_final[nationality][year]["secondary"] += 1
                    if final_category:
                        dictionary_final[nationality][year][final_category] += 1
                        
                    if nationality == "austrian literature" and final_category == "literature":
                        all_selected.append(record)
                    # if nationality == "austrian literature":
                    #     all_selected.append(record)
                        
                    

with open('three_categories_new data_from_Czarek_files.json', 'w', encoding='utf-8') as f:
    json.dump(dictionary_final, f, ensure_ascii=False)


# 4 categories:
    
from pymarc import MARCReader
from tqdm import tqdm

my_marc_files=["D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_chapters_2023-08-07.mrc",
"D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_articles_2023-08-29.mrc",
"D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_books_2023-08-29.mrc"]

# Initialize the final dictionary
dictionary_final = {}
secondary_and_literature=[]
literature=[]
secondary=[]
other=[]
# Iterate through MARC files
for my_marc_file in tqdm(my_marc_files):
    with open(my_marc_file, 'rb') as file:
        reader = MARCReader(file)
        for record in tqdm(reader):
            field_008 = record['008'].data

            # Extract year and language from field_008
            year_str = field_008[7:11]
            language = field_008[35:38]

            # Check if the year is within the range and in Polish
            if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
                year = int(year_str)
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
                            
                           # Check first condition: starts with "artykuł z czasopisma"
                            if subfield.startswith("Artykuł z czasopisma"):
                               starts_with_artykul = True
                               
                    
                           # Check second condition: is in listy655
                            if listy655.get(subfield) == "Literature":
                               found_in_listy655 = True
                               categories_for_record.add("literature")
                            
                               
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
                        continue
                        

                    else:
                        categories_for_record.add("other")
                if starts_with_artykul and found_in_listy655:
                    
                    categories_for_record.remove('secondary')
                    
                        
                final_category = None
                
                if "secondary" in categories_for_record and "literature" in categories_for_record:
                    final_category = "secondary_and_literature"
                elif "secondary" in categories_for_record:
                    final_category = "secondary"
                elif "literature" in categories_for_record:
                    final_category = "literature"
                elif "other" in categories_for_record and len(categories_for_record) == 1:
                    final_category = "other"
                
                    
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
                    
                    # Update the count for the final category in the dictionary
                    if final_category:
                        dictionary_final[nationality][year][final_category] += 1
                
                    # If the nationality is "austrian literature" and the final_category is "literature", 
                    # add the record to the all_selected list
                    if nationality == "austrian literature" and final_category == "literature":
                        literature.append(record)
                    if nationality == "american literature" and final_category == "other":
                        other.append(record)
                    if nationality == "austrian literature" and final_category == "secondary":
                        secondary.append(record)
                    if nationality == "austrian literature" and final_category == "secondary_and_literature":
                        secondary_and_literature.append(record)
                        
                    if not fields_655 and nationality=="austrian literature": 
                        secondary.append(record)    
with open('four_categories_new data_from_Czarek_files.json', 'w', encoding='utf-8') as f:
    json.dump(dictionary_final, f, ensure_ascii=False)                

#%% CHECKS THINGS

                        
                        
                        
                        
with open('austrian_secondary_literature_other.mrk', 'w', encoding='utf-8') as output_file:
    writer = TextWriter(output_file)

    # Write each record to the file
    for record in secondary_and_literature:
        writer.write(record)

    writer.close()
    
total_count = 0    
for year, categories in dictionary_final["austrian literature"].items():
    for category, count in categories.items():
        total_count += count

my_marc_files=["D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_chapters_2023-08-07.mrc",
"D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_articles_2023-08-29.mrc",
"D:/Nowa_praca/statystyki_Czarek/libri_marc_bn_books_2023-08-29.mrc"]
set_380=set()
# Initialize the final dictionary
dictionary_final = {}
all_selected=[]
# Iterate through MARC files
for my_marc_file in tqdm(my_marc_files):
    with open(my_marc_file, 'rb') as file:
        reader = MARCReader(file)
        for record in tqdm(reader):
            field_008 = record['008'].data

            # Extract year and language from field_008
            year_str = field_008[7:11]
            language = field_008[35:38]

            # Check if the year is within the range and in Polish
            if year_str.isdigit() and 1989 <= int(year_str) <= 2023 and language == 'pol':
                year = int(year_str)
                fields_380 = record.get_fields('380')


                nationalities_from_655 = []
                nationalities_from_650 = []
                nationalities_from_386 = []

                # Extract nationalities from different fields
                for field in fields_380:
                    subfields_a = field.get_subfields('a')
                    for sub in subfields_a:
                        set_380.add(sub)
                        