# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:33:01 2023

@author: dariu
"""

from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm

from itertools import zip_longest
import regex as re
import requests
from time import time
#from alphabet_detector import AlphabetDetector
#ad = AlphabetDetector()
from concurrent.futures import ThreadPoolExecutor
import threading

paths=["D:/Nowa_praca/marki_04_01_2023/cz_articles1_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles2_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles3_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles4_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_books_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_chapters_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/fennica_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/pbl_articles_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/pbl_books_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/arto_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/bn_articles_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/bn_books_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/bn_chapters_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles0_2022-08-26_30-12-2022.mrk"]
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$h).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
zViaf={}
bezviaf={}
viafs_list=[]
result={}
#val100=[]
for plik in paths:
  #  lista=mark_to_list(plik)
  #  dictrec=list_of_dict_from_list_of_lists(lista)
    dictrec=list_of_dict_from_file(plik) 

    for rekord in tqdm(dictrec):
        value=rekord.get('041')
        if value:

            for key, val in rekord.items():
                
                if key=='041':
        
                    
                    for names in val:
                        
                        #val100.append(vi)
    
                        
                        
                        name_a = re.findall(pattern3, names)
    
                        second_h=re.findall(pattern4, names)
    
                        
                        if name_a and second_h:
                            for n in name_a:
                                if '43a'+n not in result:
                                    result['43a'+n]={'language':n,'field':'43', 'subfiled':'a', 'counter':1}
                                else:
                                    result['43a'+n]['counter']+=1
                                
                            for s in second_h:
                                if '43h'+s not in result:
                                    result['43h'+s]={'language':s,'field':'43', 'subfiled':'h', 'counter':1}
                                else:
                                    result['43h'+s]['counter']+=1
                        elif name_a:
                            for n in name_a:
                                if '43a'+n not in result:
                                    result['43a'+n]={'language':n,'field':'43', 'subfiled':'a', 'counter':1}
                                else:
                                    result['43a'+n]['counter']+=1
                
        else:
             val008=rekord['008'][0][35:38]
             if '008'+val008 not in result:
                 result['008'+val008]={'language':val008,'field':'008', 'subfiled':'brak', 'counter':1}
             else:
                 result['008'+val008]['counter']+=1

                        
                            
                            
                                
                                
                    else:
                        name='brak'
                    


                    if viaf:
                        viaf=viaf[0]
                    else:
                        viaf='brak'
                    if viaf=='brak':
                        if name not in bezviaf:
                            bezviaf[name]=1
                        else:
                            bezviaf[name]+=1
                    else:
                        if name not in zViaf:
                            zViaf[name]=[viaf,1]
                        else:
                            zViaf[name][1]+=1
                            
viaf_nazwa_df=pd.DataFrame.from_dict(zViaf, orient='index') 
bez_viaf_nazwa_df=pd.DataFrame.from_dict(bezviaf, orient='index')

viaf_nazwa_df.to_excel("wszystko_z_VIAF.xlsx", sheet_name='Sheet_name_1')
bez_viaf_nazwa_df.to_excel("wszystko_bez_VIAF.xlsx", engine='xlsxwriter')
podpola=['a','h']
field=['041']
paths=["E:/Python/do_prob.mrk"]
for path in paths:
    dictrec=list_of_dict_from_file(path) 
    for rekord in tqdm(dictrec[:1]):
        id_rec=rekord['001']
        recstat={}
        for key, val in rekord.items():
            
            
            if key in field:
                
                pattern=r'(?<=\$).{2,}?(?=\$|$)' 
                for v in val:
                    subfield=re.findall(pattern, v)
                    recstat={}
                    for p in subfield:
                        
                        if p[0] in podpola:
                            print(p, key)
                            if key+p[0]+p[1:] not in recstat:
                                print(val)
                                recstat[key+p[0]+p[1:]]={p[0]:p[1:],'counter':1}
                            else:
                                if key+p[0]+p[1:] in recstat:
                                    recstat[key+p[0]+p[1:]]['counter']+=1
                                

def recstats (record, field, subfield):
    id_rec=rekord['001']
    recstat={}

    for key, val in rekord.items():
        
        
        if key in field:
            
            pattern=r'(?<=\$).{2,}?(?=\$|$)' 
            for v in val:
                subfields=re.findall(pattern, v)
                
                for p in subfields:
                    
                    if p[0] in subfield:
                        
                        if key+p[0]+p[1:] not in recstat:
                            
                            recstat[key+p[0]+p[1:]]={'field':key, 'subfield':p[0], p[0]:p[1:],'counter':1,'rec_id':id_rec[0]}
                        else:
                            if key+p[0]+p[1:] in recstat:
                                recstat[key+p[0]+p[1:]]['counter']+=1
    return recstat
x=recstats(rekord,['041'],['a','h'])
def recsstats (path,field,subfield):
    dictrec=list_of_dict_from_file(path) 
    for rekord in tqdm(dictrec[:1]):
        data=recstats(rekord,['041'],['a','h'])
        for key, value in data.items():
            print(value)
            
    
    
# F1 rekord, pole, podpole – dict
# F2 wejscie lista i pola i podpola-  lista dictow
# F3 lista dict - stats 


           

# x=''
# for n in lista[1:]:
#     print (n)
#     x+=' or key=='+f"'{n}'"
# pr=f"if key=='{lista[0]}'"+x+':'     
# exec(pr)



if key in lista:

podpola=['a','b']

    #pierwszy znak tylko spr
        
    pattern=r'(?<=\$).{2,}?(?=\$|$)' 
    subfield=re.findall(pattern, value)
    recstat={}
    for p in subfield:
        if p[0] in podpola:
            recstat[p[0]]=p[1:]
            
            

if len(lista)==1:
    print(1)
elif len(lista)==2:
    print(2)
    
    