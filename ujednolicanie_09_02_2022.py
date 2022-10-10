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

#%%
zle_viafy=[]
nowe_viaf_nazwy={}
for viaf in viaf_nazwa.keys():
    print(viaf)
    pattern_daty=r'(((\d+-\d+)|([\d?]+-[\d?]+)|(\d+-)|(\d+\?-)|(\d+))(?=$|\.| ))'
    query='https://www.viaf.org//viaf/search?query=local.viafID={}&maximumRecords=10&startRecord=1&httpAccept=application/json'.format(viaf)

    r = requests.get(query)
    r.encoding = 'utf-8'
    response = r.json()
    records=response['searchRetrieveResponse']['records']
    

    nazwa_dict={}
    lenght_list=[]
    dopasowania_lista=[]
    nowa_viaf_nazwa={}
    for record in records:
        viafy=record['record']['recordData']['viafID']
        warianty=(record['record']['recordData']['mainHeadings']['data'])
        nazwa_dict={}
        zrodla_list=[]
    
        if type(warianty) is list:
            
            for wariant in warianty:
                nazwa=wariant['text']
                zrodla=wariant['sources']['s']
                print(nazwa)
                print(zrodla)
                name = re.findall(pattern_daty, nazwa)
                print(name)
                if name:
                    if name[0][2]:
                        print(type(name[0][2]))
                        print(name)
                
                        if type(zrodla) is list:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1][0])
                            lenght_list.append(lenght)
                        else:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1])
                            lenght_list.append(lenght)
                        break
                    if name[0][3]:
                        if type(zrodla) is list:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1][0])
                            lenght_list.append(lenght)
                        else:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1])
                            lenght_list.append(lenght)
                        break
                    if name[0][4]:
                        if type(zrodla) is list:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1][0])
                            lenght_list.append(lenght)
                        else:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1])
                            lenght_list.append(lenght)
                        break
                    if name[0][5]:
                        if type(zrodla) is list:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1][0])
                            lenght_list.append(lenght)
                        else:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1])
                            lenght_list.append(lenght)
                        break
                    if name[0][6]:
                        if type(zrodla) is list:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1][0])
                            lenght_list.append(lenght)
                        else:
                            nazwa1=wariant['text']
                            nazwa_dict[nazwa1]=[]
                            
                            nazwa_dict[nazwa1].append(zrodla)
                            lenght=len(nazwa_dict[nazwa1])
                            lenght_list.append(lenght)
                        break
                    break
                else:
                    if type(zrodla) is list:
                        nazwa1=wariant['text']
                        nazwa_dict[nazwa1]=[]
                        
                        nazwa_dict[nazwa1].append(zrodla)
                        lenght=len(nazwa_dict[nazwa1][0])
                        lenght_list.append(lenght)
                    else:
                        nazwa1=wariant['text']
                        nazwa_dict[nazwa1]=[]
                        
                        nazwa_dict[nazwa1].append(zrodla)
                        lenght=len(nazwa_dict[nazwa1])
                        lenght_list.append(lenght)
        else:
            nazwa=wariant['text']
            zrodla=wariant['sources']['s']
            print(nazwa)
            print(zrodla)
    
            if type(zrodla) is list:
                nazwa1=wariant['text']
                nazwa_dict[nazwa1]=[]
                
                nazwa_dict[nazwa1].append(zrodla)
                lenght=len(nazwa_dict[nazwa1][0])
                lenght_list.append(lenght)
            else:
                nazwa1=wariant['text']
                nazwa_dict[nazwa1]=[]
                
                nazwa_dict[nazwa1].append(zrodla)
                lenght=len(nazwa_dict[nazwa1])
                lenght_list.append(lenght)            
    lista_dopasowan=[]
    max_index = [i for i, x in enumerate(lenght_list) if x == max(lenght_list)]
    for index in max_index:
        
        nasze_dopasowanie = list(nazwa_dict.keys())[index]   
        lista_dopasowan.append(nasze_dopasowanie)
    nowe_viaf_nazwy[viaf]=lista_dopasowan
#    except:
#        zle_viafy.append(viaf)
        
        
#priorytet dwie daty, potem jedna i potem źródła           
                
                

                
                

        



#print(json.dumps(response, indent=4))