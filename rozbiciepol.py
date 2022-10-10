from definicje import *
import pandas as pd
import os
from tqdm import tqdm
plik=r'/home/darek/Nowa_praca /fennica/arto_odsiane_24082021/rekordyarto1odsiane.mrk'
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)
val336=[]
for rekord in dictrec:
    for key, val in rekord.items():
        for field in val.split('â¦'):
            if key=='336':
                val336.append(val)
                
#%%
folderpath = r"/home/darek/Nowa_praca /fennica/fennica_odsiane_24082021" 
os.listdir(folderpath)
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
all_files =[]
all_files1=[]
for path in filepaths:
    with open(path, 'r', encoding='utf-8') as f:
        file = f.readlines()
        all_files.append(file)
        all_files1.extend(file)
records=[]
record=[]        
for line in all_files1:
    
    if line != '\n':
        record.append(line)
    else:
        records.append(record)
        record = []
        
dictrec=list_of_dict_from_list_of_lists(records)
val336=[]
for rekord in dictrec:
    for key, val in rekord.items():
        for field in val.split('â¦'):
            if key=='336':
                val336.append(val)
                
df = pd.DataFrame(val336, columns =['336']) 
df.to_excel("field336.xlsx", sheet_name='Sheet_name_1') 
#/(?<=a).*?(?=[$])
#(?<=b).*?(?=[$])
#%%
import re
s = r"\\$ateksti$btxt$2rdacontent"

pattern = '(?<=a).*?(?=[$])'


substring = re.search('(?<=a).*?(?=[$])', s)

print(substring)