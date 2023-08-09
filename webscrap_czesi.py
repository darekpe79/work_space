# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 08:41:16 2022

@author: darek
"""

import pandas as pd
from random import randint
from time import sleep
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm
import regex as re
import time
from definicje import *
from ast import literal_eval
pat=r'(?<=\$7[a-zA-Z]{2}).*?(?=\$|$)'

genre = pd.read_excel (r"D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/01082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze,fi).xlsx", sheet_name='Arkusz1')
genre_list=genre.desk655.to_list()
ids=[]
for things in genre.cze_id.to_list():
    things=literal_eval(things) 
    if things:
        for thing in things:
            ids.append(thing)
    
results={}
for number in tqdm(ids):
    
    # num=re.findall(pat, number)
    # if number:
    #     goodnum=''.join(filter(str.isdigit, num[0]))
    #     print(goodnum)
    #     pad_string = goodnum.zfill(9)

        
     URL=f'https://aleph.nkp.cz/F/?func=find-c&local_base=aut&ccl_term=ica={number}'
    # URL=r'https://aleph.nkp.cz/F/?func=direct&doc_number={}&local_base=AUT'.format(pad_string)
     page=get(URL)
     bs=BeautifulSoup(page.content, features="lxml")
     #genre_eng=bs.find("td", text="Angl. ekvivalent").find_next_sibling("td").text
     table=bs.find("td", text="Angl. ekvivalent")
     if table:
         genre_eng=table.find_next_sibling("td").text
         print(genre_eng)
         results[number]=genre_eng
     else:
         results[number]='no Eng. '+URL
     time.sleep(4)
        
    
    
    
excel=pd.DataFrame.from_dict(results, orient='index')
excel.to_excel('07082023_elb_en_cz_uzupelnienie.xlsx', sheet_name='new')     
            
URL=r'https://aleph.nkp.cz/F/?func=direct&doc_number=000133957&local_base=AUT'.format(goodnum)

#%% czesi ksiazki z marka 655 i ze strony tlumaczenie plus google_translate
patgen=r'(?<=\$a).*?(?=\$|$)' 
patident=r'(?<=\$7[a-zA-Z]{2}).*?(?=\$|$)'
output={}
for plik in [r"C:\Users\dariu\Desktop\praca\marki_29_07_2022\cz_chapters.mrk"]:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    
    
    for rekord in tqdm(dictrec):
        
        #if '084' not in rekord:
            #nieobrobione.append(rekord)
        
        neeew=[]
        nowe={}
        for key, val in rekord.items():
            nowe[key]=val
            rekord_num=rekord['001']
            
            if key=='655':
                #output[rekord_num]=[]
                #print(val)
                v=val.split('❦')
                #new_value_list=[]
                for genre_field in v:

                    
                    if genre_field in output:
                        counter=output[genre_field][0]+1
                        output[genre_field][0]=counter
                    else:
                            
                        output[genre_field]=[1]
                        
                        gen=re.findall(patgen, genre_field)
                        ident=re.findall(patident, genre_field)
                        if ident:
                            goodnum=''.join(filter(str.isdigit, ident[0]))
                            print(goodnum)
                            pad_string = goodnum.zfill(9)
                            print(pad_string)
                        
                            
  
                            URL=r'https://aleph.nkp.cz/F/?func=direct&doc_number={}&local_base=AUT'.format(pad_string)
                            page=get(URL)
                            bs=BeautifulSoup(page.content, features="lxml")
                            #genre_eng=bs.find("td", text="Angl. ekvivalent").find_next_sibling("td").text
                            table=bs.find("td", text="Angl. ekvivalent")
                            if table:
                                genre_eng=table.find_next_sibling("td").text
                                print(genre_eng)
                                output[genre_field].append(genre_eng)
                            else:
                                output[genre_field].append('no Eng. '+URL)
                        else:
                            output[genre_field].append('no Eng.')
                            

                        time.sleep(1.5)
                    

                   

excel=pd.DataFrame.from_dict(output, columns=['counter','genre'], orient='index')  
excel.to_excel('655_articles_cz.xlsx', sheet_name='format1')                      
page=get(URL)
bs=BeautifulSoup(page.content, features="lxml")
bs.find("td", text="Angl. ekvivalent").find_next_sibling("td").text

dicto={}
dicto['lala']=[1]
liczba=dicto['lala'][0]+1
dicto['lala'][0]=liczba
dicto['lala'].append('tłum')
