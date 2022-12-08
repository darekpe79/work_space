# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:59:36 2022

@author: dariu
"""
import json
from tqdm import tqdm
import requests
from definicje import *
import pandas as pd
import os
from tqdm import tqdm

from itertools import zip_longest
import regex as re
from time import time
import unicodedata as ud
from concurrent.futures import ThreadPoolExecutor
pliki=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles4_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles0_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles1_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles2_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles3_2022-08-26.mrk"]
patternISSN=r'(?<=\$x)\d{4}-[a-zA-Z0-9_.-]{4}'
patterntitle=r'(?<=\$t).*?(?=\$|$)' 
output={}
output2={}

for ks in pliki:
    ksiazki1=list_of_dict_from_file(ks)
    
    for rekord in tqdm(ksiazki1):
        
            
        
        for key, value in rekord.items():

                
            if key=='773':
            
                for v in value:
                    podpole_ISSN=re.findall(patternISSN, v)
                    podpole_TYTUL=re.findall(patterntitle, v)
                    if podpole_ISSN and podpole_TYTUL:
                        output[podpole_TYTUL[0]]=[podpole_ISSN[0]]
                    elif podpole_TYTUL:
                        output[podpole_TYTUL[0]]=['brak']
                    elif podpole_ISSN:
                        output[v]=[podpole_ISSN[0],'brak tytułu']
                    else:
                        output[v]=['brak','brak']
excel=pd.DataFrame.from_dict(output,orient='index')

excel.to_excel('773_cz_articles.xlsx', sheet_name='classification')       
name='john-doe'
first,last=name.split('-')                