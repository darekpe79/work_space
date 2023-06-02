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
    delay = randint(56, 66)
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
#%% PALABRAS CLAVES
import pandas as pd
from random import randint
from time import sleep
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm
import time
from langdetect import detect
import re
from pymarc import MARCReader,JSONReader
from tqdm import tqdm
from pymarc import Record, Field, Subfield
import requests
import json
from pymarc import MARCReader
from pymarc import parse_json_to_array
from pymarc import TextWriter
from pymarc import XMLWriter
from pymarc import JSONWriter
from io import BytesIO
import warnings
from pymarc import MARCReader
from pymarc import Record, Field 
import pandas as pd
from definicje import *
from nordvpn_switcher import initialize_VPN,rotate_VPN,terminate_VPN
import sys

sys.setrecursionlimit(10000)
initialize_VPN(save=1,area_input=['complete rotation'])
rotate_VPN()
my_marc_files = ["D:/Nowa_praca/Espana/ksiazki i artykuly do wyslania_17.05.2023/article_do_wysylki.mrc"]
switch=False
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'nowe6507.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'nowe6507.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        counter=0
        counter2=0
        for record in tqdm(reader):
             if record['001'].value()=='spart13000':
                 print(record)
                 switch=True
                
            
             if switch:
                print(record['001'].value())
                counter2+=1
                print(counter2)
                
                
               # print(record)
                
                check = record.get_fields('650')
                if not check:
                    if counter2 % 11==0:
                         rotate_VPN()
                    
                    delay = randint(55, 65)
                    print(delay)
                    time.sleep(delay)
                    print(record)
                    my = record.get_fields('856')
                    if my:
                        counter+=1
                        URL=my[0].value().replace("oaiart", "articulo")+'&orden=0&info=link'
                        header={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
                                   'AppleWebKit/537.36 (KHTML, like Gecko) '\
                                   'Chrome/75.0.3770.80 Safari/537.36'}
                        try:
                            page=get(URL, headers=header)
                            bs=BeautifulSoup(page.content, features="lxml")
                            print(bs)
                            block=bs.find_all('td', {'class':'metadataFieldValue dc_subject'})
                            key_words_to_use=[]
                            
                            if block:
                                for b in block[0].text.split(';'):
                                    key_words_to_use.append(b)
                                print(key_words_to_use)
                                for word in key_words_to_use:
                                    my_new_650_field = Field(
            
                                                tag = '650', 
            
                                                indicators = ['0','4'],
            
                                                subfields = [Subfield('a', word),])
                                                
                                                
                                    record.add_ordered_field(my_new_650_field)     
                        except:
                            print('pominięty--',(record['001'].value()))
                             
                data1.write(record.as_marc()) 
                writer.write(record)    
writer.close() 

#naprawa pola 100
my_marc_files = ["C:/Users/dariu/articles_poskrobaniu_650.mrc"]
switch=False
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'articlegood100.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'articlegood100.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        counter=0
        counter2=0
        for record in tqdm(reader):

                
                
               # print(record)
                
                check = record.get_fields('100')
                if len(check)>1:
                    
                    
                    for field in check[1:]:
                        print(field.value())
                        my_new_700_field = Field(

                                    tag = '700', 

                                    indicators = [' ',' '],

                                    subfields = [Subfield('a', field.value()),])
                                    
                        record.remove_field(field)           
                        record.add_ordered_field(my_new_700_field)            
                data1.write(record.as_marc()) 
                writer.write(record)    
writer.close()

#sortowanie rekordów po 001:
    
with open('C:/Users/dariu/articles_poskrobaniu_650articlegood100.mrc', 'rb') as f:
    reader = MARCReader(f)
    records = list(reader)
    
    
    records.sort(key=lambda record: int(re.findall(r'\d+', record['001'].value())[0]))
writer = TextWriter(open('articles_sorted_31.05.2023.mrk','wt',encoding="utf-8"))   
with open('articles_sorted_31.05.2023.mrc', 'wb') as f:
    
    for record in records:


        f.write(record.as_marc()) 
        writer.write(record)    
writer.close()

dictionary={'kcy':1, "ksy2":3, "kay3":2}

lista=list(dictionary.items())
lista[0][1]
lista.sort(key=lambda x:x[0][1])