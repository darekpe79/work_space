# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:30:57 2024

@author: dariu
"""

import requests
import json

def get_krs_data(krs_number, register_type):
    url = f"https://api-krs.ms.gov.pl/api/krs/OdpisPelny/{krs_number}?rejestr={register_type}&format=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

# Przykład użycia
krs_number = "0000123456"  # Zamień na rzeczywisty numer KRS
register_type = "P"  # Typ rejestru: "P" dla przedsiębiorców, "S" dla stowarzyszeń

krs_data = get_krs_data('0000059307', "P")
if krs_data:
    print(krs_data)
def save_to_json(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to file: {e}")
        
save_to_json(krs_data, "krs_data.json")


import xml.etree.ElementTree as ET
import pandas as pd
import re

def parse_element(element):
    data_dict = {}
    for child in element:
        if len(child):
            data_dict[child.tag] = parse_element(child)
        else:
            data_dict[child.tag] = child.text
    return data_dict

def xml_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return parse_element(root)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def clean_column_names(column_name):
    # Remove namespace URLs
    return re.sub(r'\{.*?\}', '', column_name)

def xml_to_excel(xml_file, excel_file):
    data_dict = xml_to_dict(xml_file)
    flat_dict = flatten_dict(data_dict)
    df = pd.DataFrame([flat_dict])
    # Clean column names
    df.columns = [clean_column_names(col) for col in df.columns]
    df.to_excel(excel_file, index=False)





# Użycie funkcji
xml_file = 'C:/Users/dariu/Downloads/De Heus Sp. z o.o. SF za 2022.xml'
excel_file = 'sciezka_do_pliku.xlsx'
xml_to_excel(xml_file, excel_file)
