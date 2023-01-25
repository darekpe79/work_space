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
pattern4='(?<=\$d).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
zViaf={}
bezviaf={}
viafs_list=[]

#val100=[]
for plik in paths:
  #  lista=mark_to_list(plik)
  #  dictrec=list_of_dict_from_list_of_lists(lista)
    dictrec=list_of_dict_from_file(plik) 

    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key=='700' or key=='100' or key=='600':
    
                
                for names in val:
                    
                    #val100.append(vi)

                    
                    
                    name = re.findall(pattern3, names)

                    viaf=re.findall(pattern5, names)

                    
                    if name:
                        name=name[0]
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


paths="D:/Nowa_praca/do_prob.mrk"
dictrec=list_of_dict_from_file(paths) 

lista=['700','600','100']
# x=''
# for n in lista[1:]:
#     print (n)
#     x+=' or key=='+f"'{n}'"
# pr=f"if key=='{lista[0]}'"+x+':'     
# exec(pr)

stri='larea'
print(stri[1:])
budowany={}
budowany[stri[0]]=stri[1:]

if key in lista:

podpola=['a','b']

    #pierwszy znak tylko spr
        
    pattern=r'(?<=\$).{2,}?(?=\$|$)' 
    subfield=re.findall(pattern, value)
    budowany={}
    for p in subfield:
        if p[0] in podpola:
            budowany[p[0]]=p[1:]
            
            

if len(lista)==1:
    print(1)
elif len(lista)==2:
    print(2)
    
    