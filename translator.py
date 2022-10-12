# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:43:17 2022

@author: darek
"""


import unicodedata as ud
import json
import pandas as pd
from definicje import *
import regex as re
import translators as ts
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from definicje import *

pat=r'(?<=\$a).*?(?=\$|$)'

genre = pd.read_excel ('C:/Users/dariu/650_czechy.xlsx', sheet_name=1)
desk_600=genre['brak eng'].to_list()
pl1=genre['pl1'].to_list()
setpl1 = list(dict.fromkeys(desk_600))
part1=setpl1[:2400]
part2=setpl1[2400:4800]
part3=setpl1[4800:7200]
part4=setpl1[7200:8600]
part5=setpl1[8600:11000]
part6=setpl1[11000:13000]
part7=setpl1[13000:15000]
part8=setpl1[15000:]

pl2=genre['pl2'].to_list()
counter=genre['counter'].to_list()
en=genre['en'].to_list()
LCSH=genre['LCSH'].to_list()
lista=list(zip(pl1,pl2,desk_600))
dictzip={z:[c,k,v,x,l] for c,k,v,z,x,l in list(zip(counter,pl1,pl2, desk_600,en,LCSH))}
results={}
for k,v in tqdm(dictzip.items()):
    results[k]=[]
    translated_en=ts.google(v[1], from_language='pl', to_language='en')
    results[k].append(translated_en)
    for e in v:
        results[k].append(e)
        print(k)
        #translated_en=ts.google(e, from_language='pl', to_language='en')
        
excel=pd.DataFrame.from_dict(results, orient='index')
excel.to_excel('PBL_650_google_translated.xlsx', sheet_name='655')         
      
#%%
def  translate_my_friend2 (k,v):
    
        results={}
        results[k]=[]
        translated_en=ts.google(v[1], from_language='pl', to_language='en')
        results[k].append(translated_en)
        for e in v:
            results[k].append(e)
        return results
with ThreadPoolExecutor() as executor:
    
    results=list(tqdm(executor.map(translate_my_friend2,dictzip.keys(),dictzip.values()),total=len(dictzip.items())))
    
#%%
def  translate_my_friend3 (k):
    
        results={}
        results[k]=[]
        translated_en=ts.google(k, from_language='cs', to_language='en')
        results[k].append(translated_en)

        return results
list_without_nan = [x for x in setpl1 if type(x) is not float]   
with ThreadPoolExecutor(1) as executor:
 
    results=list(tqdm(executor.map(translate_my_friend3,list_without_nan),total=len(list_without_nan)))

output={}
for li in results:
    for k,v in li.items():
        output[k]=v

excel=pd.DataFrame.from_dict(output, orient='index')
excel.to_excel('Czech_google_translate.xlsx')  

#%%split text by space and unique
genre = pd.read_excel ('C:/Users/dariu/650_czechy.xlsx', sheet_name=0)
desk_600=genre['tłumaczenie ze słownika (https://aleph.nkp.cz/F/)'].to_list()
wynik=[]
for d in desk_600:
    for c in d.split(' '):
        wynik.append(c)
listaaa=set([c for d in desk_600 for c in d.split(' ')])
excel=pd.DataFrame(listaaa)
excel.to_excel('czechunique650.xlsx', sheet_name='650')  
#%%
genre.to_html()
genre_list=genre.desk655.to_list()
translated_dict={'desk655':[], 'translated':[],'translated_en':[]}
for g in tqdm(genre_list):
    gen=re.findall(pat, g)
    translated=ts.google(gen[0], from_language='cs', to_language='pl')
    translated_en=ts.google(gen[0], from_language='cs', to_language='en')
    translated_dict['desk655'].append(g)
    translated_dict['translated'].append(translated)
    translated_dict['translated_en'].append(translated_en)
    
excel=pd.DataFrame.from_dict(translated_dict) #orient='index')
excel.to_excel('genre_czech_655_translated.xlsx', sheet_name='655')


def translate_my_friend (lista):
    translated_dict={'desk655':[], 'translated':[],'translated_en':[]}
    gen=re.findall(pat, lista)
    translated=ts.google(gen[0], from_language='cs', to_language='pl')
    translated_en=ts.google(gen[0], from_language='cs', to_language='en')
    translated_dict['desk655'].append(lista)
    translated_dict['translated'].append(translated)
    translated_dict['translated_en'].append(translated_en)
    return translated_dict

with ThreadPoolExecutor(5) as executor:
    
    results=list(tqdm(executor.map(translate_my_friend,genre_list),total=len(genre_list)))
 
    
excel=pd.DataFrame.from_dict(results[0]) #orient='index')

excel2= pd.DataFrame.from_records(results)
rezultat={'desk655':[], 'translated':[],'translated_en':[]}
for r in results:
    rezultat['desk655'].append(r['desk655'][0])  
    rezultat['translated'].append(r['translated'][0]) 
    rezultat['translated_en'].append(r['translated_en'][0]) 
excel=pd.DataFrame.from_dict(rezultat) #orient='index')
excel.to_excel('genre_chapters_czech_655_google_translated.xlsx', sheet_name='655')    
    
    

posts=[{'author':'darey', 'title':'co ten ŁKS','content':'co tam się wyrabia'},
{'author':'darey2', 'title':'co tren ŁKSu','content':'co tam się wyyyyrabia'}]
for post in posts:
    print(post.get('author'))


    
    