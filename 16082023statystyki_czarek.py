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
field655=pd.read_excel('D:/Nowa_praca/655_381_380_excele/bn_art_655_doMrk.xlsx', sheet_name='do_mrk',dtype=str)
listy655=dict(zip(field655['desk655'].to_list(),field655['action2'].to_list()))
field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/650__do_pracy_wszystko.xlsx', sheet_name='bn2',dtype=str)
listy650=dict(zip(field650['desk_650'].to_list(),field650['to_use'].to_list()))
dictionary_to_check={}
patterna=r'(?<=\$a).*?(?=\$|$)' 
for k,v in listy650.items():
    k=re.findall(patterna, k)[0]
    #print(v)
    if type(v)!=float:
        dictionary_to_check[k]=v

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
                        #     dictionary_final[nationality][year]["literature"] += 1'''
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

            nationalities_from_655 = []
            nationalities_from_650 = []

            # Find nationality in 655
            for field in fields_655:
                subfields_a = field.get_subfields('a')
                for subfield in subfields_a:
                    if subfield in dictionary_to_check:
                        nationalities_from_655.append(dictionary_to_check[subfield])

            # Find nationality in 650 if not found in 655
            if not nationalities_from_655:
                for field in fields_650:
                    subfields_a = field.get_subfields('a')
                    for subfield in subfields_a:
                        if subfield in dictionary_to_check:
                            nationalities_from_650.append(dictionary_to_check[subfield])

            # Combine nationalities found
            nationalities = nationalities_from_655 or nationalities_from_650

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
                        else:
                            dictionary_final[nationality][year]["secondary"] += 1

                # If nationality is from 650 and there's no 655, count as secondary
                if not fields_655 and nationality in nationalities_from_650:
                    dictionary_final[nationality][year]["secondary"] += 1

print(dictionary_final)


with open('dictionary_final_last_approach.json', 'w', encoding='utf-8') as file:
    json.dump(dictionary_final, file, ensure_ascii=False, indent=4)