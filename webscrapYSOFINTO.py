# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:01:07 2022

@author: darek
"""

import pandas as pd
from random import randint
from time import sleep
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm
genre = pd.read_excel (r"C:\Users\darek\statsy_genre_form_Finland.xlsx", sheet_name=0)
genre_list=genre.genre.to_list()
output={}
for genre in tqdm(genre_list):
    URL=Fr'https://finto.fi/yso/en/search?clang=fi&q={genre}'
    page=get(URL)
    bs=BeautifulSoup(page.content)
    #print(bs.prettify()[:100])
    bs.title.string
    block=block=bs.find_all('div', {'class':'search-result'})
    count=0
    
    for element in block:
        count+=1
        #print(count)
        #print(element)
        finska=element.find_all('a',{'class':'prefLabel'})
        inne=element.find_all('span',{'versal prefLabel proplabel'})
        symbol_jez=element.find_all('span',{'versal'})
        for nazwa in finska:
            #print(nazwa.text)
            output[nazwa.text]=[]
        for name in inne:
            output[nazwa.text].append(name.text)
 
    
excel=pd.DataFrame.from_dict(output, orient='index')
excel.to_excel('genre_fin_swe_eng.xlsx', sheet_name='format1')