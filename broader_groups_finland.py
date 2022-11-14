# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:09:06 2022

@author: dariu
"""

import json
from tqdm import tqdm
import requests
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
mojeposplitowane=pd.read_excel ('C:/Users/User/Desktop/650__do_pracy_wszystko (1).xlsx',sheet_name='Fin_650')

list6501=mojeposplitowane['links'].to_list()
list_without_nan = [x for x in list6501 if type(x) is not float] 

output={} 
outputT={}                 
for adress in tqdm(list_without_nan):
    output[adress]=[]
    outputT[adress]=[]
    number=adress.split('/')[-1].strip()
    
    response = requests.get(url='https://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2F{}&format=application/ld%2Bjson'.format(number) )                     
    try:
        data = response.json() ['graph']
        next_adress=[]
        for d in data:
            
            if 'narrower' in d:
                if isinstance(d['narrower'], dict) and d['narrower']['uri']==adress :
                    
                    uri=d['uri']
                    next_adress.append(uri)
                    prefLabel=d['prefLabel']
                    if not any(d['lang'] == 'en' for d in prefLabel):
                        en_value=[v['value'] for v in prefLabel if v['lang']=='fi']
    
                    else:
                        en_value=[v['value'] for v in prefLabel if v['lang']=='en']
                    
                    output[adress]=[(uri,en_value[0])]
                    outputT[adress]=[en_value[0]]
        while len(next_adress)>0:
            next_adress1=next_adress[0]
            #print(proba1)  
            number=next_adress1.split('/')[-1].strip()
            
            response = requests.get(url='https://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2F{}&format=application/ld%2Bjson'.format(number) )                     
            data = response.json() ['graph']  
            next_adress=[]
            for d in data:
                
                if 'narrower' in d:
                    if isinstance(d['narrower'], dict) and d['narrower']['uri']==next_adress1:
                        
                        uri=d['uri']
                        
                        next_adress.append(uri)
                        prefLabel=d['prefLabel']
                        if not any(d['lang'] == 'en' for d in prefLabel):
                            en_value=[v['value'] for v in prefLabel if v['lang']=='fi']
    
                        else:
                            en_value=[v['value'] for v in prefLabel if v['lang']=='en']
                
                        output[adress].append((uri,en_value[0]))
                        outputT[adress].append(en_value[0])
    except:
        outputT[adress]=['brak_wartosci']
with open ('json_broader_fin.json', 'w', encoding='utf-8') as file:
    json.dump(output,file,ensure_ascii=False)  
    
# CIEKAWOSTKA!!!!!!!!!  
    
keys_file = open("D:/Nowa_praca/broader concepts_FIN/json_broader_fin.json")
keys = keys_file.read().encode('utf-8')
keys_json = json.loads(keys)   
first2pairs = {k: keys_json[k] for k in list(keys_json)[:1]}
output={}
output2={}


for key,val in tqdm(first2pairs.items()):
    if val:
        for v in val:
            #print(v)
       
            
       
            
            output2[v[0]]={}
            
            number=v[0].split('/')[-1].strip()
            #print(number)
            response = requests.get(url='https://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2F{}&format=application/ld%2Bjson'.format(number) )                     
            
            data = response.json() ['graph']
            
            for d in data:
                if 'skos:member' in d:
                    
                    #print(d['skos:member']['uri'])
                    if d['skos:member']['uri']==v[0]:
                        #print(d['skos:member'])
                        
                        
                        prefLabel=d['prefLabel']
                        if not any(d['lang'] == 'en' for d in prefLabel):
                            en_value=[v['value'] for v in prefLabel if v['lang']=='fi']
    
                        else:
                            en_value=[v['value'] for v in prefLabel if v['lang']=='en']
                        print('jeden:  ',en_value) 
                        #print(v[1])
                        if v[1] not in output2[v[0]]:
                            output2[v[0]][v[1]]=en_value

                            
                        if v[1] not in output:
                            output[v[1]]=en_value
                        else:
                            output[v[1]].append(en_value[0])
                            
excel=pd.DataFrame.from_dict(output, orient='index')
excel.to_excel('broader_groups.xlsx', sheet_name='broader') 



output3={}

for key,val in tqdm(keys_json.items()):
    if val:
        for v in val:
            #print(v)
            
            
            output3[v[0]]=v[1]  
            
            
            
mojeposplitowane=pd.read_excel ('D:/Nowa_praca/broader concepts_FIN/koncepty650doSPRbelongstogroups.xlsx',sheet_name='Arkusz1')
dictionary = dict(zip(mojeposplitowane['links'].to_list(), mojeposplitowane['dictionary_translate'].to_list()))

dobry={}
for key,val in tqdm(dictionary.items()):
        dobry[key]={}
        number=key.split('/')[-1].strip()
        #print(number)
        response = requests.get(url='https://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2F{}&format=application/ld%2Bjson'.format(number) )                     
        
        data = response.json() ['graph']
        
        for d in data:
            if 'skos:member' in d:
                
                #print(d['skos:member']['uri'])
                if type(d['skos:member']) is list:
                    if any(d['uri']== key for d in d['skos:member']):
                        prefLabel=d['prefLabel']
                        if not any(d['lang'] == 'en' for d in prefLabel):
                            en_value=[v['value'] for v in prefLabel if v['lang']=='fi']

                        else:
                            en_value=[v['value'] for v in prefLabel if v['lang']=='en']
                        #print('jeden:  ',en_value) 
                        if val not in dobry[key]:
                            dobry[key][val]=en_value
                        else:
                            dobry[key][val].append(en_value[0])
                    
                elif d['skos:member']['uri']==key:
                    #print(d['skos:member'])
                    
                    
                    prefLabel=d['prefLabel']
                    if not any(d['lang'] == 'en' for d in prefLabel):
                        en_value=[v['value'] for v in prefLabel if v['lang']=='fi']

                    else:
                        en_value=[v['value'] for v in prefLabel if v['lang']=='en']
                    #print('jeden:  ',en_value) 
                    if val not in dobry[key]:
                        dobry[key][val]=en_value
                    else:
                        dobry[key][val].append(en_value[0])
with open ('json_main_groups_belongs.json', 'w', encoding='utf-8') as file:
    json.dump(dobry,file,ensure_ascii=False)  
                        
keys_file = open('D:/Nowa_praca/broader concepts_FIN/json_main_groups_belongs.json')
keys = keys_file.read().encode('utf-8')
keys_json = json.loads(keys)  
out={}

for key, val in keys_json.items():
    out[key]=[]
    if val:
        for k,v in val.items():
            #out[key]=[k]+v
            out[key]=[k,v]

            

out2={}
listofall=[]
for key, val in out.items():
    out2[key]={}
    if val:
        
            #out[key]=[k]+v
        out2[key][val[0]]=val[1]
        listofall.extend(val[1])
# ILE GRUP STATSY:
stat={}
for li in listofall:
    if li in stat:
        stat[li]+=1
    else:
        stat[li]=1
        
excel=pd.DataFrame.from_dict(stat, orient='index')

excel.to_excel('group_stats.xlsx', sheet_name='groups')     
with open ('listoffallgrouops.json', 'w', encoding='utf-8') as file:
    json.dump(listofall,file,ensure_ascii=False)             
# ILE GRUP STATSY:
with open ('json_all_groups_belongs2.json', 'w', encoding='utf-8') as file:
    json.dump(out,file,ensure_ascii=False)  
with open('D:/Nowa_praca/broader concepts_FIN/json_all_groups_belongs.json', encoding='utf-8') as f:
    statdata = json.load(f)                        

                
    
excel=pd.DataFrame.from_dict(out, orient='index')

excel.to_excel('mall_groups_belongs.xlsx', sheet_name='broader')
mojeposplitowane=pd.read_excel ('D:/Nowa_praca/broader concepts_FIN/do_liczenia_grup_all.xlsx',sheet_name='broaders+nasze')
diciton=mojeposplitowane.to_dict('records')
probny={}
for l in diciton:
    print(l)
    for x in range(1,6):
        if l[x] not in probny:
            probny[l[x]]=l['count']
        else:
            probny[l[x]]+=l['count']
excel=pd.DataFrame.from_dict(probny, orient='index')

excel.to_excel('groups_all.xlsx', sheet_name='group')            
dicto={}
dicto['book']=[100,'lala']
dicto['cool']=[200,'lala']
licz={}
for key, val in dicto.items():
    print(val)
    if val[1] not in licz:
        licz[val[1]]=val[0]
    else:
        licz[val[1]]+=val[0]
