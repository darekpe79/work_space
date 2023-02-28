import re
from difflib import *
from tqdm import tqdm


'''
Supported data types:
    - str
    - pandas dataframe
    - list
    - dict
    
Normalization modes:
    - NFD - canonical decomposition
    - NFC - first applies a canonical decomposition, then composes pre-combined characters again
    - NFKD - compatibility decomposition
    - NFKC -  applies the compatibility decomposition, followed by the canonical composition
'''

import unicodedata as ud
import json
import pandas as pd
from definicje import *


def decode_str(str_input: str, mode = "NFC") -> str:
    try:
        return ud.normalize(mode, str_input)
    except TypeError:
        return str_input
    
def decode_df(df_input: pd.core.frame.DataFrame, mode = "NFC") -> pd.core.frame.DataFrame:
    output_df = pd.DataFrame()
    for col in df_input.columns:
        col_normalized = [decode_str(value, mode) for value in df_input[col]]
        output_df[col] = col_normalized
    return output_df

def decode_list(list_input: list, mode = "NFC") -> list:
    output_list = []
    for elem in list_input:
        if type(elem) == str:
            output_list.append(decode_str(elem))
        elif type(elem) == list:
            output_list.append(decode_list(elem))
        elif type(elem) == dict:
            output_list.append(decode_dict(elem))
        elif type(elem) == pd.core.frame.DataFrame:
            output_list.append(decode_df(elem))   
        else: output_list.append(elem)
    return output_list
      
def decode_dict(dict_input: dict, mode = "NFC") -> dict:
    output_dict = {}
    for key, value in dict_input.items():
        if type(value) == str:
            output_dict[decode_str(key)] = decode_str(value)
        elif type(value) == list:
            output_dict[decode_str(key)] = decode_list(value)
        elif type(value) == dict:
            output_dict[decode_str(key)] = decode_dict(value)
        elif type(value) == pd.core.frame.DataFrame:
            output_dict[decode_str(key)] = decode_df(value)
        else: output_dict[decode_str(key)] = value
    return output_dict

def compose_data(data, mode = "NFC"):
    data_type = type(data)
    if data_type == str:
        return decode_str(data)
    elif data_type == list:
        return decode_list(data)
    elif data_type == dict:
        return decode_dict(data)
    elif data_type == pd.core.frame.DataFrame:
        return decode_df(data)
    else:
        raise Exception("Not supported data type!")
#%%Tworzenie listy rekordów z pliku 

def mark_to_list(path):
    records = []
    with open(path, 'r', encoding = 'utf-8') as mrk:
        record = []
        for line in mrk.readlines():
            if line == '\n':
                pass
            elif line.startswith('=LDR') and record: 
                records.append(record)
                record = []
                record.append(line)
            else:
                record.append(line)
        records.append(record)   
    final_output = []  
    for record in records:      
        cleared_record=[]
        for i, field in enumerate(record):
            if not field.startswith('='):
                cleared_record[-1]=cleared_record[-1][:-1]+field
                
            else:
                cleared_record.append(field)
        final_output.append(cleared_record)
        
    return final_output








def mark_to_list1(path):
    records = []
    with open(path, 'r', encoding = 'utf-8') as mrk:
        record = []
        for line in mrk.readlines():
            if line != '\n':
                record.append(line)
            else:
                records.append(record)
                record = []
    return records

#%% lista do listy słowników

def list_of_dict_from_list_of_lists (records):
    recs2table = []
    for record in records:
        rec_dict = {}
        for field in record:
            if field[1:4] in rec_dict.keys():
                rec_dict[field[1:4]] = '❦'.join([rec_dict[field[1:4]], field[6:].strip()])
            else:
                rec_dict[field[1:4]] = field[6:].strip()
        recs2table.append(rec_dict)
    return recs2table
     
        
#%% Zapisujemy rekordy do pliku mark
def sortdict(dictionary):
    copy=dictionary.copy()
    output_dict = {'LDR': dictionary.get('LDR')}
    del copy['LDR']
    output_dict.update(dict(sorted(copy.items())))
    
    return output_dict

def to_file (file_name, list_of_dict_records):
    ''' list of dict records to file mrk'''
    file1 = open(file_name, "w", encoding='utf-8') 
    for record in list_of_dict_records:
        
        dictionary=sortdict(record)
        dictionary2=compose_data(dictionary)
        
        for key, value in dictionary2.items():
            for field in value.split('❦'):
                line='='+key+'  '+field+'\n'
                file1.writelines(line)
        file1.writelines('\n')
    
    file1.close()

def id_of_rec(dict):
    wszystkie_id=[]
    for record in dict:
        try:
            wszystkie_id.append(record['001'])
        except KeyError:
            pass
    return wszystkie_id 

def LDR_monography(list_dict):
    monography=[]
    for record in list_dict:
        if "LDR" in record.keys():
            if any(re.findall("^.{7}m", record['LDR'][0])):
                
                monography.append(record)
    return monography
def LDR_article(list_dict):
    article=[]
    for record in tqdm(list_dict):
        if "LDR" in record.keys():
            if any(re.findall("^.{7}b", record['LDR'][0])):
                
                article.append(record)
    return article             
    
            
def matcher(str1, str2):
    match=SequenceMatcher(a=str1, b=str2)
    return match.ratio() 

def lang_detect(texts):
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    # russian
    if re.search("[\u0400-\u0500]+", texts):
        return "ru"
    return None

def unique(list1):
  
    
    unique_list = []
      
    
    for x in list1:
        
        if x not in unique_list:
            unique_list.append(x)
    list1[:]=unique_list           
            
            
           
def list_of_dict_from_list_of_lists2 (records):
    recs2table = []
    for record in records:
        rec_dict = {}
        for field in record:
            if field[1:4] in rec_dict.keys():
                rec_dict[field[1:4]].append(field[6:].strip())
            else:
                rec_dict[field[1:4]] = [field[6:].strip()]
        recs2table.append(rec_dict)
    return recs2table

def list_of_dict_from_file (path):
    records=mark_to_list(path)
    recs2table = []
    for record in records:
        rec_dict = {}
        for field in record:
            if field[1:4] in rec_dict.keys():
                rec_dict[field[1:4]].append(field[6:].strip())
            else:
                rec_dict[field[1:4]] = [field[6:].strip()]
        recs2table.append(rec_dict)
    return recs2table

def to_file2 (file_name, list_of_dict_records):
    ''' list of dict records to file mrk'''
    file1 = open(file_name, "w", encoding='utf-8') 
    for record in list_of_dict_records:
        
        dictionary=sortdict(record)
        dictionary2=compose_data(dictionary)
        
        for key, value in dictionary2.items():
            for field in value:
                line='='+key+'  '+field+'\n'
                file1.writelines(line)
        file1.writelines('\n')
    
    file1.close()        
            
            
def get_indexes(l, val):
    return [i for i,value in enumerate(l) if value==val]         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            