# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:37:17 2022

@author: dariu
"""

import json
from tqdm import tqdm
import requests
from definicje import *
import pandas as pd
import os
from tqdm import tqdm
#from googletrans import Translator
from itertools import zip_longest
import regex as re
from pprint import pprint
import pprint
from time import time


#%%

#proba
files=[
"D:/Nowa_praca/marki 29.06.2022Przed ubogaceniemGenre/BN_articles.mrk",
"D:/Nowa_praca/marki 29.06.2022Przed ubogaceniemGenre/BN_books.mrk",
"D:/Nowa_praca/marki 29.06.2022Przed ubogaceniemGenre/BN_chapters.mrk"]
patternISSN=r'(?<=\$x)\d{4}-[a-zA-Z0-9_.-]{4}'
patterntitle=r'(?<=\$t).*?(?=\$|$)'  
cale={}
bez_issn=set()
rekordy_655=unique([])
for plik in files:

    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    


##dla jednego pliku

    path2=plik.split('/')
    pattern4=r'(?<=\$7).*?(?=\$|$)'
    pattern_daty=r'\(?[\(?\d\? ]{2,5}[-–.](\(?[\d\?]{3,5}\)?| \))?'
    pattern_daty_marc=r'(?<=\$d).*?(?=\$|$)'
    pattern_a_marc=r'(?<=\$a).*?(?=\$|$)'
    pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'

    switch=False
    tylkoViaf=[]
    #proba=[]
    counter=0
    
    for rekord in tqdm(dictrec):
        
        for key, val in rekord.items():
            if key=='655':
                
                 val2=val.split('❦')

                 for value in val2:

                         
                    name= re.findall(pattern_a_marc, value)
                    #issn = re.findall(patternISSN, value)
                    if name:
                       rekordy_655.append(rekord['001'])
                        
                       cale[name[0]]=key
                        
#u=unique(rekordy_655)
u2=set(rekordy_655)
excel=pd.DataFrame.from_dict(cale, orient='index')
excel.to_excel('całepol650.xlsx', sheet_name='655') 
#%%
major_gen=[r'\\$iMajor genre$aLyrical poetry$leng',
r'\\$iMajor genre$aFiction$leng',r'\\$iMajor genre$aOther$leng',r'\\$iMajor genre$aDrama$leng']
literature_str=r'\\$iMajor genre$aLiterature$leng'
secondary_str=r'\\$iMajor genre$aSecondary literature$leng'
genre381=set()
genre380=set()
counter=0
for plik in ["C:/Users/dariu/marki_02.09.2022/fennica_2022-09-02.mrk",
"C:/Users/dariu/marki_02.09.2022/arto_2022-09-02.mrk"]:
    path=plik.split('/')[-1].split('.')[-2]
    
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists2(lista)
    #to_file2('new.mrk',dictrec)

    for rekord in tqdm(dictrec):
        counter+=1
        
        
        

        

        nowe={}

        for key, val in rekord.items():
            nowe[key]=val
            
            if key=='381':
                for v in val:
                    if v in major_gen:
                        genre381.add(nowe['001'][0])
            if key=='380':
                for v in val:
                    if v==literature_str or v==secondary_str:
                        genre380.add(nowe['001'][0])
                
#%% nowy PBL
genre=pd.read_excel('D:/Nowa_praca/pbl655_nowy_po_mapowaniu.xlsx', sheet_name=0)  
genre_list=genre.zrobione.tolist()
lista=[]       
for g in genre_list:
    for gen in g.split('|'):
        lista.append(gen)
l=set(lista)
l2=list(cale.keys())   
l3=l2+lista   
l=set(l3)  
excel=pd.DataFrame(l)
excel.to_excel('całepol655_pomapowaniuPBLnaBN.xlsx', sheet_name='655') 
       
