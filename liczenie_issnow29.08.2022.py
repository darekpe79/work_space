# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 13:00:49 2022

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
files=["D:/Nowa_praca/marki 29.06.2022Przed ubogaceniemGenre/cz_articles4.mrk",
"D:/Nowa_praca/marki 29.06.2022Przed ubogaceniemGenre/cz_articles0.mrk",
"D:/Nowa_praca/marki 29.06.2022Przed ubogaceniemGenre/cz_articles1.mrk",
"D:/Nowa_praca/marki 29.06.2022Przed ubogaceniemGenre/cz_articles2.mrk",
"D:/Nowa_praca/marki 29.06.2022Przed ubogaceniemGenre/cz_articles3.mrk"]
patternISSN=r'(?<=\$x)\d{4}-[a-zA-Z0-9_.-]{4}'
patterntitle=r'(?<=\$t).*?(?=\$|$)'  
cale={}
bez_issn=set()
wszyscy=set()
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
            if key=='773':
                
                 val2=val.split('❦')

                 for value in val2:

                         
                    name= re.findall(patterntitle, value)
                    issn = re.findall(patternISSN, value)
                    if name and issn:
                        cale[issn[0]]=name[0]
                    else:
                        name=re.findall(patterntitle, value)
                        if name:
                            bez_issn.add(name[0])
                             
excel=pd.DataFrame.from_dict(cale, orient='index')
excel.to_excel('podziałaniachpolacy_issn.xlsx', sheet_name='655')   
excel2= pd.DataFrame(list(bez_issn))
excel2.to_excel('odziałaniachpolacy_bez_issn.xlsx', sheet_name='655') 