# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:42:18 2023

@author: dariu
"""

import pywikibot
from pywikibot import pagegenerators
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

#wyciaganie po czeskim ID z wikidaty


field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/650__do_pracy_wszystko.xlsx', sheet_name='czech2',dtype=str)
listy=dict(zip(field650['desk_650'].to_list(),field650['tonew650_national'].to_list()))
dictionary=field650.to_dict('records')
idpattern=r"\$7(.+?)(?:\$|$)"
apattern=r"\$a(.+?)(?:\$|$)"
czeid={}
for rec in dictionary:
    desk_650=rec['desk_650']
    matchid = re.search(idpattern, desk_650)
    matchsub=re.search(apattern, desk_650)
    if matchid:
        
        czeid[desk_650]=[matchsub.group(1),matchid.group(1)]
        
        
for key,val in tqdm(czeid.items()):
    id_num=val[1]

    sleep(0.2)
    
    
    id_proba="ph115157"
    query = f"""
        SELECT ?item ?itemLabel ?locId ?propertyP950 ?propertyP2347 ?propertyP691
        WHERE 
        {{
          ?item wdt:P691 "{id_num}" . 
          OPTIONAL {{ ?item wdt:P244 ?locId . }}
          OPTIONAL {{ ?item wdt:P950 ?propertyP950 . }}
          OPTIONAL {{ ?item wdt:P2347 ?propertyP2347 . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
        }}
    """
    
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/sparql-results+json",
    }
    
    params = {
        "query": query,
    }
    url='https://query.wikidata.org/sparql'
    response = requests.get(url, headers=headers, params=params)
    data = response.json()

# Process the results
    item_uri_list=[]
    item_label_list=[]
    loc_list=[]
    sp_list=[]
    fi_list=[]
    #cz_list=[]
    
    for result in data["results"]["bindings"]:
        print(result)
        
        item_uri = result["item"]["value"]
        item_uri_list.append(item_uri)
        item_label = result.get("itemLabel", {}).get("value", "N/A")
        item_label_list.append(item_label)
        loc_id=result.get('locId',{}).get("value", "N/A")
        loc_list.append(loc_id)
        print(loc_id)
        sp_id=result.get('propertyP950',{}).get("value", "N/A")
        sp_list.append(sp_id)
        fi_id=result.get('propertyP2347',{}).get("value", "N/A")
        fi_list.append(fi_id)
        #cz_id=result.get('propertyP691',{}).get("value", "N/A")
        print(f"Item URI: {item_uri}")
        print(f"Item Label: {item_label}")
        
    if item_uri_list:
        unique(item_uri_list)
        czeid[key].append(", ".join(item_uri_list))
        unique(item_label_list)
            
        czeid[key].append(", ".join(item_label_list))
        unique(loc_list)
        czeid[key].append(", ".join(loc_list))
        unique(sp_list)
        czeid[key].append(", ".join(sp_list))
        unique(fi_list)
        czeid[key].append(", ".join(fi_list))
            


czech_df=pd.DataFrame.from_dict(czeid, orient='index')
czech_df.to_excel("czech_WIKI_sparql.xlsx")  

#%% wyciaganie po finskim id z wikidaty
field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/650__do_pracy_wszystko.xlsx', sheet_name='fin2',dtype=str)

dictionary=field650.to_dict('records')
idpattern=r"\$0(.+?)(?:\$|$)"
apattern=r"\$a(.+?)(?:\$|$)"
finid={}
for rec in dictionary:
    desk_650=rec['desk_650']
    matchid = re.search(idpattern, desk_650)
    matchsub=re.search(apattern, desk_650)
    if matchid:
        
        finid[desk_650]=[matchsub.group(1),matchid.group(1)]
        
        
for key,val in tqdm(finid.items()):
    id_num=val[1].split('/')[-1][1:]

    sleep(0.1)
    
    #item = '10123'

    query = f"""
        SELECT ?item ?itemLabel ?locId ?propertyP950 ?propertyP2347 ?propertyP691
        WHERE 
        {{
          ?item wdt:P2347 "{id_num}" . 
          OPTIONAL {{ ?item wdt:P244 ?locId . }}
          OPTIONAL {{ ?item wdt:P950 ?propertyP950 . }}
          OPTIONAL {{ ?item wdt:P691 ?propertyP691 . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
        }}
    """
    url='https://query.wikidata.org/sparql'
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/sparql-results+json",
    }
    
    params = {
        "query": query,
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()

# Process the results
    item_uri_list=[]
    item_label_list=[]
    loc_list=[]
    sp_list=[]
    cz_list=[]
    #cz_list=[]
    
    for result in data["results"]["bindings"]:
        print(result)
        
        item_uri = result["item"]["value"]
        item_uri_list.append(item_uri)
        item_label = result.get("itemLabel", {}).get("value", "N/A")
        item_label_list.append(item_label)
        loc_id=result.get('locId',{}).get("value", "N/A")
        loc_list.append(loc_id)
        print(loc_id)
        sp_id=result.get('propertyP950',{}).get("value", "N/A")
        sp_list.append(sp_id)
        cz_id=result.get('propertyP691',{}).get("value", "N/A")
        cz_list.append(cz_id)
        #cz_id=result.get('propertyP691',{}).get("value", "N/A")
        print(f"Item URI: {item_uri}")
        print(f"Item Label: {item_label}")
        
    if item_uri_list:
        unique(item_uri_list)
        finid[key].append(", ".join(item_uri_list))
        unique(item_label_list)
            
        finid[key].append(", ".join(item_label_list))
        unique(loc_list)
        finid[key].append(", ".join(loc_list))
        unique(sp_list)
        finid[key].append(", ".join(sp_list))
        unique(cz_list)
        finid[key].append(", ".join(cz_list))
fin_df=pd.DataFrame.from_dict(finid, orient='index')
fin_df.to_excel("fin_WIKI_sparql.xlsx") 
           
#%%spanish_id_loc_for spanish items   
field650=pd.read_excel('C:/Users/dariu/12062023_odpytka_slownik_hiszpanski_loc_wiki_ver1.0.xlsx', sheet_name='Sheet1',dtype=str)

dictionary=field650.to_dict('records')


espid={}

for value in dictionary:
    key=value['field 650']
    espid[key]=[]
    
    if value['loc_id']!="N/D":
        loc_to_split=value['loc_id']
        locId=loc_to_split.split('/')[-1]
        
        
        espid[key].append(value['loc_id'])
    


        #locId='n95011545'   
       #n95011545

        query = f"""
            SELECT ?item ?itemLabel ?locId ?propertyP950 ?propertyP2347 ?propertyP691
            WHERE 
            {{
              ?item wdt:P244 "{locId}" .
              OPTIONAL {{ ?item wdt:P950 ?propertyP950 . }}
              OPTIONAL {{ ?item wdt:P2347 ?propertyP2347 . }}
              OPTIONAL {{ ?item wdt:P691 ?propertyP691 . }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" . }}
            }}
        """
        url='https://query.wikidata.org/sparql'
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/sparql-results+json",
        }
        
        params = {
            "query": query,
        }
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        item_uri_list=[]
        item_label_list=[]
        fin_list=[]
        sp_list=[]
        cz_list=[]
        for result in data["results"]["bindings"]:
             print(result)
             
             item_uri = result["item"]["value"]
             item_uri_list.append(item_uri)
             item_label = result.get("itemLabel", {}).get("value", "N/A")
             item_label_list.append(item_label)
             fin_id=result.get('propertyP2347',{}).get("value", "N/A")
             fin_list.append(fin_id)
             
             sp_id=result.get('propertyP950',{}).get("value", "N/A")
             sp_list.append(sp_id)
             cz_id=result.get('propertyP691',{}).get("value", "N/A")
             cz_list.append(cz_id)
             #cz_id=result.get('propertyP691',{}).get("value", "N/A")
             print(f"Item URI: {item_uri}")
             print(f"Item Label: {item_label}")
             
        if item_uri_list:
            unique(item_uri_list)
            espid[key].append(", ".join(item_uri_list))
            unique(item_label_list)
                
            espid[key].append(", ".join(item_label_list))
            unique(fin_list)
            espid[key].append(", ".join(fin_list))
            unique(sp_list)
            espid[key].append(", ".join(sp_list))
            unique(cz_list)
            espid[key].append(", ".join(cz_list))
            
fin_df=pd.DataFrame.from_dict(espid, orient='index')
fin_df.to_excel("esp_WIKI_sparql.xlsx") 