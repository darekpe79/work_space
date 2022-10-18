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
from time import sleep
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
    link=r'(?<=\$0).*?(?=\$|$)'
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
files=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles0_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles1_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles2_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles3_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles4_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_books_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_chapters_2022-09-02.mrk"]

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

results={}
for genre_field in tqdm(do_roboty2):
    sleep(1)
    pattern_a_marc=r'(?<=\$a).*?(?=\$|$)'
    pat=r'(?<=\$7[a-zA-Z]{2}).*?(?=\$|$)'
    ident=re.findall(pat, genre_field)
    field_a=re.findall(pattern_a_marc, genre_field)
    
        
        
    if ident and field_a:
        goodnum=''.join(filter(str.isdigit, ident[0]))
        print(goodnum)
        pad_string =goodnum.zfill(9)
        print(pad_string)
        URL=r'https://aleph.nkp.cz/F/?func=direct&doc_number={}&local_base=AUT'.format(pad_string)
        page=requests.get(URL)
        bs=BeautifulSoup(page.content, features="lxml")
        #genre_eng=bs.find("td", text="Angl. ekvivalent").find_next_sibling("td").text
        table=bs.find("td", text="Angl. ekvivalent")
        if table:
            genre_eng=table.find_next_sibling("td").text
            print(genre_eng)
            results[genre_field]=[1,field_a[0], genre_eng]
        else:
            results[genre_field]=[1,field_a[0], 'brak eng']
    elif ident:
        goodnum=''.join(filter(str.isdigit, ident[0]))
        print(goodnum)
        pad_string =goodnum.zfill(9)
        print(pad_string)
        URL=r'https://aleph.nkp.cz/F/?func=direct&doc_number={}&local_base=AUT'.format(pad_string)
        page=requests.get(URL)
        bs=BeautifulSoup(page.content, features="lxml")
        #genre_eng=bs.find("td", text="Angl. ekvivalent").find_next_sibling("td").text
        table=bs.find("td", text="Angl. ekvivalent")
        if table:
            genre_eng=table.find_next_sibling("td").text
            print(genre_eng)
            results[genre_field]=[1,'brak_a', genre_eng]
        else:
            results[genre_field]=[1,'brak_a', 'brak eng']
    elif field_a:
        results[genre_field]=[1,field_a[0], 'brak eng']
    else:
        results[genre_field]=[1,'brak_a', 'brak eng']
        
    
for key in results.keys():
    for k in output.keys():
        if key==k:
            results[key][0]=output[key]          
            
excel=pd.DataFrame.from_dict(results, orient='index')
excel.to_excel('650_czechy.xlsx', sheet_name='format1')                
#%%     
pole650 = pd.read_excel ("C:/Users/dariu/650_czechy.xlsx", sheet_name=1)
dict_650 = dict(zip(pole650['field'].to_list(),pole650['czech'].to_list()))
path='C:/Users/dariu/Desktop/praca/wetransfer_ucla0110-mrk_2022-08-10_1019/150.txt'
records = []
with open(path, 'r', encoding = 'utf-8') as f:
    record=[]
    for line in f.readlines():
        line=line[10:]
        print(line)
        if line == '\n':
            pass
        elif line.startswith('LDR') and record: 
            records.append(record)
            record = []
            record.append(line)
        else:
            record.append(line)
    records.append(record)  


poID={}
ponazwie={}
identyfikator=[]
idem=set()
for rec in tqdm(records):
    #print(rec)
    pat=r'(?<=\$7).*?(?=\$|$)'
    pattern_a_marc=r'(?<=\$a).*?(?=\$|$)'
    line001=''
    key_dict_650=''
    for line in rec:
        
        
        if line.startswith('001'):
            #print(line)
            line2=line[8:].strip()
            for key in dict_650.keys():
                ident=re.findall(pat, key)
                if ident:
                    ident=ident[0].strip()
                    idem.add(ident)
                    
                    if ident==line2:
                        line001=key
                        print(ident)
                    
                    
        if line001:
            if line.startswith('75007'):
                pole_a=re.findall(pattern_a_marc, line)
                poID[line001]=pole_a[0]
        
        
        else:
            if line.startswith('150'):
                for key in dict_650.keys():
                
                    field_a_dict=re.findall(pattern_a_marc, key)
                
                    if field_a_dict:
                        field_a_dict=field_a_dict[0].strip()
                        pole_a150=re.findall(pattern_a_marc, line)
                        if pole_a150:
                            pole_a150=pole_a150[0].strip()
                            #print(pole_a150)
                            if pole_a150==field_a_dict:
                                print(pole_a150)
                                key_dict_650=key
            if key_dict_650:
                if line.startswith('75007'):
                    pole_a=re.findall(pattern_a_marc, line)
                    ponazwie[key_dict_650]=pole_a[0]
                
                                
                    

                


                                              
                         
excel=pd.DataFrame.from_dict(ponazwie, orient='index')
excel.to_excel('650_czechy_po_nazwie.xlsx', sheet_name='format1') 
excel=pd.DataFrame.from_dict(poID, orient='index')
excel.to_excel('650_czechy_po_ID.xlsx', sheet_name='format1') 

    
    
    while True:
            line = f.readline()
            if not line:
                break
            print(line.strip())
            
idx = ['a', 'b', 'c', 'd']
l_1 = [1, 2, 3, 4]
l_2 = [5, 6, 7, 8]

keys = ['mkt_o', 'mkt_c'] 
new_dict = {k: dict(zip(keys, v)) for k, v in zip(idx, zip(l_1, l_2))}
   