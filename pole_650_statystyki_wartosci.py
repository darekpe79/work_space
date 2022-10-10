# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:04:50 2022

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
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/pbl_articles_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/pbl_books_2022-09-02.mrk"]
patternISSN=r'(?<=\$x)\d{4}-[a-zA-Z0-9_.-]{4}'
patterntitle=r'(?<=\$t).*?(?=\$|$)'  
cale={}

for plik in files:

    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists2(lista)
    


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

    
    for rekord in tqdm(dictrec):
        
        for key, val in rekord.items():
            if key=='650':
                
                for v in val:
                    podzielone=v.split('$')
                    
                    do_pracy=''
                    for el in podzielone[1:]:
                        do_pracy+=", "+el[1:]
                        #print (el)
                        
                    
                    field_a=re.findall(pattern_a_marc, v)
                    if field_a:
                         if v not in cale:
                             cale[v]=[1,field_a[0],do_pracy.strip(' ,')]
                         else:
                            cale[v][0]+=1
excel=pd.DataFrame.from_dict(cale, orient='index') 
excel.to_excel("PBL_cale_650.xlsx", sheet_name='Sheet_name_1')
cal={} 
cal['lala']=[1,'pop']
cal['lala'][0]+=1#[cal['lala'][0]+1,'pop']