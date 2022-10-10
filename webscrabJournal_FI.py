# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:34:33 2022

@author: darek
"""

import pandas as pd
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm


#for genre in tqdm(genre_list):
no='article432'   
URL=fr'https://kansalliskirjasto.finna.fi/Record/journalfi.{no}#details'
page=get(URL)
bs=BeautifulSoup(page.content)
#print(bs.prettify()[:100])
bs.title.string
block=bs.find_all('div', {'class':"subjectLine"})
if block:
    for b in block:
        print(b.text.strip('\n'))
        
recordmarclist=[]        
if block:
    for b in block:
        field653='=653  00$a '+b.text.strip('\n')
        recordmarclist.append(field653)





