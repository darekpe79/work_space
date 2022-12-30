import re
from difflib import *
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
def to_file (file_name, list_of_dict_records):
    ''' list of dict records to file mrk'''
    file1 = open(file_name, "w", encoding='utf-8') 
    for record in list_of_dict_records:
        
        
        for key, value in record.items():
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
            if any(re.findall("^.{7}m", record['LDR'])):
                
                monography.append(record)
    return monography
def LDR_article(list_dict):
    monography=[]
    for record in list_dict:
        if "LDR" in record.keys():
            if any(re.findall("^.{7}b", record['LDR'])):
                
                monography.append(record)
    return monography             
    
            
def matcher(str1, str2):
    match=SequenceMatcher(a=str1, b=str2)
    return match.ratio()            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            