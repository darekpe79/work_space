# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:58:15 2022

@author: darek
"""
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
from concurrent.futures import ThreadPoolExecutor
val2=[]
pliki=["F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles4.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles0.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles1.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles2.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles3.mrk"]
#plik=r"F:\Nowa_praca\NOWI CZESI_FENNICAPBL_BN_14.04.2022\cz_articles0.mrk"
for plik in pliki:
    
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    path2=plik.split('\\')
    
    
    
    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key=='773':
    
                
                val2.extend(val.split('❦'))


    
                

df = pd.DataFrame((val2), columns =['773'])
    
df.to_excel("PBLarticles_duze2.xlsx", sheet_name='Sheet_name_1') 



patternISSN=r'(?<=\$x)\d{4}-[a-zA-Z0-9_.-]{4}'
patterntitle=r'(?<=\$t).*?(?=\$|$)'   
issns=[]
tytul=[]
name=[]
for names in tqdm(val2):
    for names2 in names.split('❦'):
        name.append(names2)
        title=re.findall(patterntitle, names2)   
        issn=re.findall(patternISSN, names2)
        try:
            tytuly=title[0].rstrip('. - /')
        except:
            tytuly='brak'
        try:
            issny=issn[0]
        except:
            issny='brak'
        issns.append(issny)
        tytul.append(tytuly)
df = pd.DataFrame(list(zip(tytul,issns, name)),
               columns =['tytul','issn', '773'])
df.to_excel("PBL_773_ISSNrozbiteCzesi0.xlsx", sheet_name='Sheet_name_1') 
new=df.loc[df['issn'] != 'brak']
new.to_excel("Czesi0.xlsx", sheet_name='Sheet_name_1')  
new2 = new.drop_duplicates(subset = ["issn"])
new2.to_excel("Czesicali_bez_brak_bez_duplikatow.xlsx", sheet_name='Sheet_name_1')
lista_issn=new2['issn'].tolist()
lista_tytulow=new2['tytul'].tolist()
dict_issn=dict(zip(lista_issn, lista_tytulow))
 
# ZACIAGANIE ISSN   


df_issny=pd.read_excel(r"C:\Users\darek\ujednolicanie_BN_czasopisma\BN_773_ISSNrozbiteLibri_bez_brak_bez_duplikatow.xlsx")
dict_issn=dict(zip(df_issny.issn, df_issny.tytul))

def issn (issn):
    slownik={'issn':'','tytul':'','nowy_tytul':'','ratio':''}   
    
        
    tytul=dict_issn[issn]
    slownik['tytul']=tytul
    slownik['issn']=issn
        
       
        
    url = 'https://portal.issn.org/resource/ISSN/{}?format=json'.format(issn)
    
    response = requests.request("GET", url)
    response.encoding = 'utf-8'
    try:
        response=response.json()
        
        graph=response['@graph']
        switch=False
        
        for g in graph:
            if 'KeyTitle' in g['@id']:
                switch=True
                tytul2=g['value']
                if type(tytul2) is list:
                    slownik['nowy_tytul']=tytul2
                    slownik['ratio']='brak'
                else:
                    
                    ratio=matcher(ud.normalize('NFD',tytul),ud.normalize('NFD',tytul2 ))
                    slownik['ratio']=ratio
                    slownik['nowy_tytul']=tytul2
                
        if switch==False:
            slownik['nowy_tytul']='brak'
            slownik['ratio']='brak'
    except:
        slownik['nowy_tytul']='brak'
        slownik['ratio']='brak'
    return slownik
            
        
with ThreadPoolExecutor() as executor:
    
    results=list(tqdm(executor.map(issn,dict_issn.keys()),total=len(dict_issn)))
        
                
                

df_to_excel=pd.DataFrame.from_dict(results)    
df_to_excel.to_excel("czech0_ujednolicone_czasopisma.xlsx", engine='xlsxwriter')    
# Opracowanie sciagnietego materialu:

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

df_issny=pd.read_excel(r"C:\Users\darek\czech0_ujednolicone_czasopisma.xlsx")

dict_issn=dict(zip(df_issny.issn, df_issny.nowy_tytul)) 
lista_cala=[]
dictionary={'issn':[],'nowa_nazwa':[] }
for issn, nazwa in dict_issn.items():
    if nazwa.startswith('['):
        
        nazwa=nazwa.replace(", '", "")
        print(nazwa)
        nazwa=nazwa.strip("[']").split("'")
        print(nazwa)
        lista=[]
        for n in nazwa:
            if lang_detect(n):
                continue
            
            lista.append(n.strip(' .'))
            #code3 = (n.encode('ascii', 'ignore')).decode("utf-8")
            
        lista_cala.append(lista[0])
        
        dictionary['issn'].append(issn)
        dictionary['nowa_nazwa'].append(lista[0])
    else:
        dictionary['issn'].append(issn)
        
        lista_cala.append(nazwa.strip(' .'))
        dictionary['nowa_nazwa'].append(nazwa.strip(' .'))
df_to_excel=pd.DataFrame.from_dict(dictionary)    
df_to_excel.to_excel("Czech_ujednolicone_czasopisma_po_selekcji.xlsx", engine='xlsxwriter') 


### Ujednolicanie MARC
df_ujednolicone=pd.read_excel(r"C:\Users\darek\Czech_ujednolicone_czasopisma_po_selekcji.xlsx",sheet_name=0)
dict_pole100_viaf=dict(zip(df_ujednolicone['issn'], df_ujednolicone.nowa_nazwa)) 
pliki=["F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles4.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles0.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles1.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles2.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles3.mrk"]
#plik=r"F:\Nowa_praca\NOWI CZESI_FENNICAPBL_BN_14.04.2022\cz_articles0.mrk"
for plik in pliki:

    
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    path2=plik.split('/')
    patternISSN=r'(?<=\$x)\d{4}-[a-zA-Z0-9_.-]{4}'
    patterntitle=r'(?<=\$t).*?(?=\$|$)' 
    cale=[]
    for rekord in tqdm(dictrec):
        
        for key, val in rekord.items():
            if key=='773':
                
                 val2=val.split('❦')
                 listavalue=[]

                 for value in val2:
                     if '$x' in value:
                         issn = re.findall(patternISSN, value)
                         if issn:
                             if issn[0] in dict_pole100_viaf:
                         
                         
                             
                                 #print(value)
                                 #podpole_ISSN=re.findall(patternISSN, value)
                                 #podpole_TYTUL=re.findall(patterntitle, value)
                                 ujedn773=dict_pole100_viaf[issn[0]]
                                 #print(podpole_TYTUL)
                                 value=value+'$s'+ujedn773
                                 #print(value)
        
                                
    
    
    
    
    
                             
                     listavalue.append(value)
                             
                             #print(poleviaf)
                 rekord[key]='❦'.join(listavalue)
                    
                         
        cale.append(rekord)
    
    to_file(path2[-1], cale)
        

            

            
    
        

            
            
            
