# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:31:17 2021

@author: darek
"""
import pandas as pd
from random import randint
from time import sleep
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm
import time

import re

text = "The ISSN of this journal is 1234-567X."
pattern = r'\b\d{4}-\d{3}[\dX]\b'
match = re.findall(pattern, text)

URL=r'https://dialnet.unirioja.es/revistas/submateria/2310?registrosPorPagina=50&inicio=251'
header={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
           'AppleWebKit/537.36 (KHTML, like Gecko) '\
           'Chrome/75.0.3770.80 Safari/537.36'}
page=get(URL, headers=header)
bs=BeautifulSoup(page.content, features="lxml")
block=bs.find_all('span', {'class':'titulo'})
print(block)


lista=[]

for element in block:
    #print(element)
    a_tag = element.find('a', href=True)
    lista.append(a_tag['href'])
    a_tag.text
czasopisma={}
for url in tqdm(lista): 
    delay = randint(36, 66)
    print(delay)
    time.sleep(delay)
    URL=fr'https://dialnet.unirioja.es{url}'
    header={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
               'AppleWebKit/537.36 (KHTML, like Gecko) '\
               'Chrome/75.0.3770.80 Safari/537.36'}
    page=get(URL, headers=header)
    bs=BeautifulSoup(page.content, features="lxml")   
    #print(bs)
    info=bs.find(id='informacion')
    if 'País: España' in info.text:
        tytul=bs.find('span', {'class':'titulo'}).text
        print(tytul)
        try:
            sub_tytul=bs.find('span', {'class':'subtitulo'}).text
        except:
            sub_tytul='brak'
    
        informacion=info.find_all('span')
        issn_list=[]
        for info in informacion:
            match = re.findall(pattern, info.text)
            if match:
                issn=match[0]
                issn_list.append(issn)
        if issn_list:
            czasopisma[issn+' '+tytul]=[tytul,sub_tytul]+issn_list
        else:
            czasopisma[tytul]=[tytul,sub_tytul]
places=pd.DataFrame.from_dict(czasopisma, orient='index')
places.to_excel("Filologías_Generalidades_periodicos_dialnet.xlsx")            

issn=['lala','fala']
issn[1:]
for i in issn:
    print(i+i)
    tytul=element.find('h5', {'class':'item-result-title'})
    
    if tytul:
        wlasciwy=tytul.text
        
        #wlasciwy=wlasciwy.split('    ')
        #print(wlasciwy)
        wlasciwy=wlasciwy.replace('\nKey-title \xa0', '')#strip('\nKey-title \xa0')
        #print(wlasciwy.strip())
        #break

    tytul2=bs.find_all('div', {'class':'item-result-content-text'})
    for tyt in tytul2:
        if 'Title proper:' in tyt.text:
            print(tyt.text)
    
    
    
    
    
    #issns2=issns.find('p')
    #issns=issns2.text
    
            
        
    
    
        
    if issns==None:
        issns=bs.find('div', {'sidebar-accordion-list-selected-item'}).text
        issns=issns.strip('ISN :')
        #print(issns)
        
        
    elif issns is not None:
        issns2=issns.find('p')
        issns=issns2.text
        issns=issns.strip('ISN :')
    else:
        issns='brak'
    
    
    


    




    

