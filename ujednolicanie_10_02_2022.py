# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:07:56 2022

@author: darek
"""
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
from time import time
from alphabet_detector import AlphabetDetector
plik=r"F:\Nowa_praca\libri\Iteracja 2021-07\oviafowane02.02.2022BN_PBL\libri_marc_bn_chapters_2021-08-05!100_600_700z_VIAF_i_bez_viaf_good995.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)
val100=[]
probal=[]
for rekord in tqdm(dictrec):
    for key, val in rekord.items():
        if key=='700' or key=='100' or key=='600':

            v=val.split('❦')
            for vi in v:
                val100.append(vi)


df = pd.DataFrame(val100,columns =['100_700_600'])
df.to_excel("sprawdzanie_marc_articles_2021_2021-08-05.xlsx", sheet_name='Sheet_name_1') 
#wydobycie nazw osobowych:
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$d).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
original_names=[]
name_date_list=[]
viafs_list=[]
for names in tqdm(val100):
    original_names.append(names)
    
    name = re.findall(pattern3, names)
    dates = re.findall(pattern4, names)
    viaf=re.findall(pattern5, names)
    #print  (dates)
    
    if name:
        name=name[0]
    else:
        name='brak'
    
    if dates:
        
        datesstr=re.sub('\)|\(','',dates[0])
        datesstr=datesstr.strip('.')
        
    else:
        datesstr=''
    if viaf:
        viaf=viaf[0]
    else:
        viaf='brak'
    #print(datesstr)
    name_date=name.strip('.')+' '+datesstr
    name_date_list.append(name_date)
    viafs_list.append(viaf)
    
df2 = pd.DataFrame (list(zip(original_names,name_date_list,viafs_list)), columns =['100','nazwisko data','viaf' ]) 
df2 = df2[df2.viaf != 'brak']
lista_viaf=df2.viaf.tolist()
nazwisko_data_lista=df2['nazwisko data'].tolist()
viaf_nazwa = dict(zip(lista_viaf,nazwisko_data_lista))
#viaf_nazwa_df=pd.DataFrame.from_dict(viaf_nazwa, orient='index')
#viaf_nazwa_df.to_excel("Przed_ujednolicanko_Books_BN.xlsx", sheet_name='Sheet_name_1') 

#%%
zle_viafy=[]
nowe_viaf_nazwy={}
for viaf in tqdm(viaf_nazwa.keys()):
    #print(viaf)
    pattern_daty=r'(([\d?]+-[\d?]+)|(\d+-)|(\d+\?-)|(\d+))(?=$|\.|\)| )'
    query='https://www.viaf.org/viaf/{}/viaf.json'.format('98953497')
    try:
        r = requests.get(query)
        r.encoding = 'utf-8'
        response = r.json()
        #viafy=response['viafID']
        warianty=response['mainHeadings']['data']
            
        
        
        
        
        wszystko=[]
        if type(warianty) is list:
            
            
            for wariant in warianty:
                if 'ORCID' in wariant:
                    pass
                nazwa=wariant['text']
                zrodla=wariant['sources']['s']
                if type(zrodla) is list:
                    liczba_zrodel=len(zrodla)
                else:
                    liczba_zrodel=1
                        
                #print(nazwa)
                #print(zrodla)
                daty = re.findall(pattern_daty, nazwa)
                #print(daty)
                
                if daty:
                    for index,grupa in enumerate(daty[0][1:]):
                        if grupa:
                            priorytet=index
                            break
                            
                else: 
                    priorytet=5
                    
                jeden_wariant=[nazwa,priorytet,liczba_zrodel] 
                wszystko.append(jeden_wariant)
            best_option=wszystko[0]
                
                
            for el in wszystko:
                
               if el[1]<best_option[1]:
                   
                   best_option=el
               elif el[1]==best_option[1]:
                   
                   if el[2]>best_option[2]:
                       best_option=el
                       
        else:
            best_option=[warianty['text']]
              
        nowe_viaf_nazwy[viaf]=best_option[0]
    
    except:
        zle_viafy.append(viaf)
excel=pd.DataFrame.from_dict(nowe_viaf_nazwy, orient='index')
excel.to_excel("ujednolicanko_Chapters_BN.xlsx", sheet_name='Sheet_name_1') 
df = pd.DataFrame(zle_viafy)
df.to_excel("ujednolicanko_blad_Chapters.xlsx", sheet_name='Sheet_name_1') 
        
                                   
                           
                       
                       
                       
                       
                            
                    
                    
                            
                            
                            
                
                
                
                
                
                
                
                
                
                
                
                
                

                

                
                

        



#print(json.dumps(response, indent=4))