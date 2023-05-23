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

URL=r'https://dialnet.unirioja.es/revistas/submateria/2340?registrosPorPagina=50&inicio=151'
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
places.to_excel("Filologías_Filologías_clásicas_antiguas_dialnet.xlsx")            

#%% OAI harvesting with senor Niko

from sickle import Sickle
sickle = Sickle('https://dialnet.unirioja.es/oai/OAIHandler')
issny=pd.read_excel('D:/Nowa_praca/Espana/all_issns_dialnet.xlsx', sheet_name='Arkusz1',dtype=str)
listy=issny.issn.to_list()
	
'0049-5719' in listy

records = sickle.ListRecords(metadataPrefix='oai_dc', set='23',ignore_deleted=True)
counter=0
records_lista=[]
for record in records:
    record.header
    print(record.header.identifier)
    

    try:
        sources=record.metadata['source'] 
    except KeyboardInterrupt:
        break
    except:
        sources=[]
    if sources:
        for source in sources:
            
            match = re.findall(pattern, source)
            if match:
                for issn in match:
                    
                    if issn in listy:
                        records_lista.append(record.metadata)
                        
                    
    counter+=1
    if counter==100:
        break

with open ('articles_espana.json', 'w', encoding='utf-8') as file:
    json.dump(records_lista,file,ensure_ascii=False)   









