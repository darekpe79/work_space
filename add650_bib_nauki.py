# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:07:56 2022

@author: darek
"""
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
import regex as re
from datetime import date

today = date.today()
with open('C:/Users/dariu/Downloads/bibnau_bn_final_results.json', encoding='utf-8') as fh:
    dataname = json.load(fh)
listadospr=[]
for k, v in dataname.items():
    for num in v:
        listadospr.append(num[0])
listadospr=set(listadospr)
        
# dd/mm/YY
d1 = today.strftime("%d-%m-%Y")
field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/650__do_pracy_wszystko.xlsx', sheet_name='czech2',dtype=str)
listy=dict(zip(field650['desk_650'].to_list(),field650['tonew650_national'].to_list()))
def isNaN(num):
    return num!= num
dicto={}
dictionary650=field650.to_dict('records')
for elem in dictionary650:
    
    
    if isNaN(elem['tonew650_genre']) is  False:
        if elem['desk_650'] in dicto:
            dicto[elem['desk_650']].append(elem['tonew650_genre'])
        else:
            #dicto[elem['desk_650']]=[]
            dicto[elem['desk_650']]=[elem['tonew650_genre']]
    if isNaN(elem['tonew650_national']) is False:
        if elem['desk_650'] in dicto:
            dicto[elem['desk_650']].append(elem['tonew650_national'])
        else:
            dicto[elem['desk_650']]=[elem['tonew650_national']]
            

towork={}        
for key, val in dicto.items():
    if val:
        towork[key]=val
        
def capfirst(s):
    return s[:1].upper() + s[1:]    
        
        


paths=["D:/Nowa_praca/marki_04_01_2023/cz_chapters_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles0_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles1_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles2_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles3_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles4_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_books_2022-08-26_30-12-2022.mrk"
]




#val100=[]
counter=set()
wspolne=[]
for plik in paths:
    record=list_of_dict_from_file(plik)
    nametopath=plik.split('/')[-1].split('.')[0]+'_'
    
    for rec in tqdm(record):
   
            
        if '650' not in rec:
            continue
        else:
            for key,val in rec.items():
                if key=='650':
                    for v in val:
                        
                        if v in towork:
                            counter.add(rec['001'][0])
                            
                            
                            for v2 in towork[v]:
                                #print(r'\7$a'+v2+r'$2Libri')
                                toappend=r'\7$a'+capfirst(v2)
                                rec[key].append(toappend)
                    unique(rec[key])
                
    to_file2(nametopath+d1+'.mrk',record)           
                
                
with open('listaBibnauk_BibNar+libri.json' , 'w', encoding='utf-8') as data:
    
    json.dump(wspolne,data,indent=4,ensure_ascii=False)
wspolne_z_id={}
for k, v in dataname.items():
    for num in v:
        if num[0] in wspolne:
            wspolne_z_id[k]=num[0]
with open('listaBibnauk_BibNar+libri.json' , 'w', encoding='utf-8') as data:
    
    json.dump(wspolne_z_id,data,indent=4,ensure_ascii=False)
            
     