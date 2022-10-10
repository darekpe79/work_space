import json
from tqdm import tqdm
import requests
from definicje import *
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
from time import time
import unicodedata as ud
df = pd.read_excel (r"C:\Users\darek\ujednolicanie-zderzenie_tabel-Czarek-ja__BŁĘDY\Darek_Brak_dat_BN_02.03.2022_BŁĘDY_Zderzenie_Czarek.xlsx",sheet_name=4) 
def split_it(year):
    data_ujednolicona=re.search(r'\(?[\(?\d\? ]{2,5}[-–.](\(?[\d\?]{3,5}\)?| \))?', year)
    if data_ujednolicona:
        return (data_ujednolicona.group()).strip('() ')
    else:
        return 'brak'
def split_itD(year):
    data_ujednolicona=re.search(r'(?<=\$d).*?(?=\$|$)', year)
    if data_ujednolicona:
        return (data_ujednolicona.group()).strip('() ')
    else:
        return 'brak'
def split_itViaf(viaf):
    try:
        Viaf=re.search(r'\d+', viaf)
        if Viaf:
            return (Viaf.group()).strip('() ')
        else:
            return 'brak'   
    except:
        return 'brak'
    
df['imie_nazwisko_PBL']=df.pbl_nazwisko+' '+df.pbl_imie

df['viafID'] = df['viaf'].apply(split_itViaf)
df['100_bez_viaf']=df['100'].str.replace(r'\$1http.+', '', regex=True)



#df['datapoleD'] = df['100'].apply(split_itD)

df.to_excel("BN_BEZ_DAT_doViafowania_pewne.xlsx", engine='xlsxwriter')  
#%%
df = pd.read_excel (r"C:\Users\darek\ujednolicanie-zderzenie_tabel-Czarek-ja__BŁĘDY\Darek_Brak_dat_BN_02.03.2022_BŁĘDY_Zderzenie_Czarek.xlsx",sheet_name=3) 
names=df['nazwisko data'].to_list()
names2=df['nazwa_czarek'].to_list()

ratio_list=[]
for (name1,name2) in zip(names, names2):
    #print (name1,name2)
    ratio=matcher(ud.normalize('NFD',name1.strip(' ')),ud.normalize('NFD',name2.strip(' ') ))
    ratio_list.append(ratio)
    
    
    
    
    

        
df['ratio'] = ratio_list
df.to_excel("Zderzenie_Czarek_BN.xlsx", engine='xlsxwriter')  

ratio=SequenceMatcher(None,'Gębuś Ryszard'.encode(), 'Gębuś Ryszard'.encode() ).ratio()
ud.normalize('NFC','Gębuś Ryszard') == ud.normalize('NFC','Gębuś Ryszard') 
'GębuśRyszard'=='GębuśRyszard'



