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
adres=urllib.parse.unquote('http://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2Fp3537&format=&format=application/json')
print(adres)
#number='p3537'
#patternYSO=r'(?<=\/yso\/).*?(?=\$|$)'





#proba
files=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/arto_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/fennica_2022-09-02.mrk"]

output={}
do_roboty=set()
for plik in files:

    
    dictrec=list_of_dict_from_file(plik)
    for rekord in tqdm(dictrec):
        
        for key, val in rekord.items():
            if key=='650':
                
                
                for v in val:
                    do_roboty.add(v)
                    if v in output:
                        output[v]+=1
                    else:
                        output[v]=1
                    
do_roboty2=list(do_roboty)
                    
def pole_650(lista):    
    cale={}       

       # print(v)
    pattern_a_marc=r'(?<=\$a).*?(?=\$|$)'
    patternYSO=r'(?<=\/yso\/).*?(?=\$|$)'
    field_a=re.findall(pattern_a_marc, lista)
    YSO=re.findall(patternYSO, lista)
    if YSO and field_a:
        number=YSO[0]           
    
        response = requests.get(url=f'https://finto.fi/yso/en/page/{number}')
       
        bs=BeautifulSoup(response.content)
        engVal=bs.title.string
        #print(engVal)
        enval2=bs.find("span", {'id':"pref-label"})
        if enval2:
            enval2=enval2.text
            if lista not in cale:
            
                cale[lista]=[1,field_a[0],enval2]
            else:
                cale[lista][0]+=1
        else:
                                        
            if lista not in cale:
            
                cale[lista]=[1,field_a[0],'brak']
            else:
                cale[lista][0]+=1
    elif field_a:
            if lista not in cale:
            
                cale[lista]=[1,field_a[0],'brak']
            else:
                cale[lista][0]+=1
    else:
        if lista not in cale:
        
            cale[lista]=[1,'brak','brak']
        else:
            cale[lista][0]+=1
    return cale
                            
                            
                        

                       
with ThreadPoolExecutor() as executor:
    
    results=list(tqdm(executor.map(pole_650,do_roboty2),total=len(do_roboty)))
do_excel={}                              
for r in results:
    for k,v in r.items():
        do_excel[k]=v

for key in do_excel.keys():
    for k in output.keys():
        if key==k:
            do_excel[key][0]=output[key]
        
excel=pd.DataFrame.from_dict(do_excel, orient='index')
excel.to_excel('650_finlandia.xlsx', sheet_name='format1')                      
                        
                       
                        
                       
                        
                       
                        
                       
                        
                       
#%%                        
                       
                        graph=response['graph']
                        for gr in graph:
                            #print(g)
                            if 'prefLabel' in gr:
                                for g in gr['prefLabel']:
                                    if g['lang']=='en':
                                        print(g['value'])
                                        
                                        
                                        
lista=[{"lang":"sme","value":"girjeárvvoštallamat"},{"lang":"en","value":"book reviews"},{"lang":"fi","value":"kirja-arvostelut"}]
"kirja-arvostelut" in lista