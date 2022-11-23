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
response = requests.get(url='https://udcsummary.info/php/index.php?id=67277&lang=en#')
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
    
    
    
    
    
    

