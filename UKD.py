# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:42:11 2022

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
import requests
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from time import sleep

patternnum=r'(?<=<div class="nodetag">).*?(?=<\/div>)'
#https://finto.fi/yso/en/page/p3537
response = requests.get(url='https://udcsummary.info/php/index.php?lang=pl')
bs=BeautifulSoup(response.content)
soup = BeautifulSoup(response.text, 'html.parser')
ds= soup.find_all('div',{"id":"classtree"}) 
for d in ds:
    print(d)
    
ukdnum=[]    
test = soup.findAll('script') 
for t in test:
    print(type(t.text))
    ident=re.findall(patternnum, t.text)
    for ids in ident:
        idstofind="\'"+ids+"\'"
        ukdnum.append(idstofind)
        

namelist={}
for t in test:
    #print(type(t.text))
    for ukd in ukdnum:
        patt=rf'(?<=&nbsp;&nbsp;).*?(?={ukd})'
        name=re.findall(patt, t.text)
        if name:
            namelist[ukd]=name[0]
        namelist.append(name)
    
excel=pd.DataFrame.from_dict(namelist, orient='index') 
excel.to_excel("ukd.xlsx", sheet_name='Sheet_name_1')    
    
#%%
classifications={'080': 'Universal Decimal Classification Number',
'082': 'Dewey Decimal Classification Number',
'083': 'Additional Dewey Decimal Classification Number',
'084': 'Other Classification Number'}
    
ukd_og = pd.read_excel ("C:/Users/dariu/ukd_og.xlsx", sheet_name='Arkusz1')
dict_ = dict(zip(ukd_og['num'].to_list(),ukd_og['class'].to_list()))
removed_value = dict_.pop(5)
pliki=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/bn_chapters_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/pbl_articles_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/pbl_books_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/bn_articles_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/bn_books_2022-08-26.mrk"]
pattern_a_marc=r'(?<=\$a).*?(?=\$|$)' 
output=[]
output2={}
for ks in pliki:
    ksiazki1=list_of_dict_from_file(ks)
    
    for rekord in tqdm(ksiazki1):
        
            
        
        for key, value in rekord.items():
            if key=='080':
            
                for v in value:
                    if type(v) is list:
                        print(v)

                        
         
                    field_a=re.findall(pattern_a_marc, v)
                    print(field_a)
                    
                    if field_a:
                        for num, classi in dict_.items():
                            #classif=''
                            print(field_a)
                            if field_a[0].startswith(str(num)):
                                classif=dict_[num]
                                if field_a[0] not in  output2:
                                    output2[field_a[0]]=[1,classif]
                                else:
                                    output2[field_a[0]][0]+=1
                            elif field_a[0].startswith(str('5')):
                                if field_a[0].startswith(str('51')):
                                    classif='MATHEMATICS'

                                    
                                else:
                                    classif='NATURAL SCIENCES'
                                if field_a[0] not in  output2:
                                    output2[field_a[0]]=[1,classif]
                                else:
                                    output2[field_a[0]][0]+=1
                                    
                                
                          
excel=pd.DataFrame.from_dict(output2,orient='index')

excel.to_excel('polclassification2.xlsx', sheet_name='classification')
    
    
    
    
    

