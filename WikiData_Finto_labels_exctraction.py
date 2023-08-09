# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:53:45 2023

@author: dariu
"""

#%% Fin sparql extraction
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
from ast import literal_eval


field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/18072023fin_narrower_broader_uri.xlsx', sheet_name='Sheet1',dtype=str)
lista_uri=field650['broader'].tolist()
lista_main=field650['main'].tolist()
dictionary=dict(zip(lista_main,lista_uri))
id_num = '4668'
final={}
for key, value in dictionary.items():
    
    final[key]={'broader':[]}
    
    listy=literal_eval(value)
    for elem in listy:
        print(elem)
        id_num=elem.split('/')[-1][1:]




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
        labels=[]
        item_uri_list=[]
        
        for result in data["results"]["bindings"]:
            print(result)
            
            item_uri = result["item"]["value"]
            labels.append(item_uri)
            item_label = result.get("itemLabel", {}).get("value", "N/A")

            labels.append(item_label)
        unique(labels)
        final[key]['broader'].append((elem,labels))
        
lista_uri=field650['narrower'].tolist()
lista_main=field650['main'].tolist()
dictionary=dict(zip(lista_main,lista_uri))
for key, value in dictionary.items():
    
    final[key].update({'narrower':[]})
    
    listy=literal_eval(value)
    for elem in listy:
        print(elem)
        id_num=elem.split('/')[-1][1:]




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
        labels=[]
        item_uri_list=[]
        
        for result in data["results"]["bindings"]:
            print(result)
            
            item_uri = result["item"]["value"]
            labels.append(item_uri)
            item_label = result.get("itemLabel", {}).get("value", "N/A")

            labels.append(item_label)
        unique(labels)
        final[key]['narrower'].append((elem,labels))
fin_df=pd.DataFrame.from_dict(final, orient='index')    
fin_df.to_excel("fin_broader_narrower_labels.xlsx") 
with open("fin_broader_narrower_labels.json","w", encoding='utf-8') as jsonfile:
        json.dump(final,jsonfile,ensure_ascii=False)
with open('fin_broader_narrower_labels.json', encoding='utf-8') as user_file:
  parsed_json = json.load(user_file)    
#%%Czech broader/narrower labels
field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/26072023cze_narrower_broader_id.xlsx', sheet_name='Sheet1',dtype=str)
lista_uri=field650['broader'].tolist()
lista_main=field650['main'].tolist()
dictionary=dict(zip(lista_main,lista_uri))

final={}
for key, value in dictionary.items():
    
    final[key]={'broader':[]}
    
    listy=literal_eval(value)
    for elem in listy:
       # print(elem)

    #id_num="ph115157"
        id_num=elem
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
    
           
        labels=[]
        item_uri_list=[]
        
        for result in data["results"]["bindings"]:
            #print(result)
            
            item_uri = result["item"]["value"]
            labels.append(item_uri)
            item_label = result.get("itemLabel", {}).get("value", "N/A")
    
            labels.append(item_label)
        unique(labels)
        final[key]['broader'].append((elem,labels))

lista_uri=field650['narrower'].tolist()
lista_main=field650['main'].tolist()
dictionary=dict(zip(lista_main,lista_uri))
for key, value in dictionary.items():
    
    final[key].update({'narrower':[]})
    
    listy=literal_eval(value)
    for elem in listy:
        print(elem)
        id_num=elem




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
        labels=[]
        item_uri_list=[]
        
        for result in data["results"]["bindings"]:
            print(result)
            
            item_uri = result["item"]["value"]
            labels.append(item_uri)
            item_label = result.get("itemLabel", {}).get("value", "N/A")

            labels.append(item_label)
        unique(labels)
        final[key]['narrower'].append((elem,labels))
fin_df=pd.DataFrame.from_dict(final, orient='index')    
fin_df.to_excel("cze_broader_narrower_labels.xlsx") 
with open("cze_broader_narrower_labels.json","w", encoding='utf-8') as jsonfile:
        json.dump(final,jsonfile,ensure_ascii=False)
#%%SP BROADER NARROWER

field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/01082023esp_narrower_broader_id.xlsx', sheet_name='Sheet1',dtype=str)
lista_uri=field650['broader'].tolist()
lista_main=field650['main'].tolist()
dictionary=dict(zip(lista_main,lista_uri))

final={}
for key, value in dictionary.items():
    
    final[key]={'broader':[]}
    
    listy=literal_eval(value)
    for elem in listy:
        print(elem)

    #id_num="ph115157"
        id_num=elem
        query = f"""
            SELECT ?item ?itemLabel ?locId ?propertyP950 ?propertyP2347 ?propertyP691
            WHERE 
            {{
              ?item wdt:P950 "{id_num}" . 
              OPTIONAL {{ ?item wdt:P244 ?locId . }}
              
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
    
           
        labels=[]
        item_uri_list=[]
        
        for result in data["results"]["bindings"]:
            print(key, result)
            
            item_uri = result["item"]["value"]
            labels.append(item_uri)
            item_label = result.get("itemLabel", {}).get("value", "N/A")
    
            labels.append(item_label)
        unique(labels)
        final[key]['broader'].append((elem,labels))

lista_uri=field650['narrower'].tolist()
lista_main=field650['main'].tolist()
dictionary=dict(zip(lista_main,lista_uri))
for key, value in dictionary.items():
    
    final[key].update({'narrower':[]})
    
    listy=literal_eval(value)
    for elem in listy:
        print(elem)
        id_num=elem




        query = f"""
            SELECT ?item ?itemLabel ?locId ?propertyP950 ?propertyP2347 ?propertyP691
            WHERE 
            {{
              ?item wdt:P950 "{id_num}" . 
              OPTIONAL {{ ?item wdt:P244 ?locId . }}
              
              OPTIONAL {{ ?item wdt:P2347 ?propertyP2347 . }}
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
        labels=[]
        item_uri_list=[]
        
        for result in data["results"]["bindings"]:
            print(result)
            
            item_uri = result["item"]["value"]
            labels.append(item_uri)
            item_label = result.get("itemLabel", {}).get("value", "N/A")

            labels.append(item_label)
        unique(labels)
        final[key]['narrower'].append((elem,labels))
fin_df=pd.DataFrame.from_dict(final, orient='index')    
fin_df.to_excel("01082023sp_broader_narrower_labels.xlsx") 
with open("01082023sp_broader_narrower_labels.json","w", encoding='utf-8') as jsonfile:
        json.dump(final,jsonfile,ensure_ascii=False)








#%% API PHP Extract labels- brakujące pl 
def get_wikidata_label(entity_id, language_code='en'):
    url = f'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbgetentities',
        'format': 'json',
        'ids': entity_id,
        'props': 'labels',
        'languages': language_code,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'entities' in data and entity_id in data['entities']:
        entity_data = data['entities'][entity_id]
        if 'labels' in entity_data and language_code in entity_data['labels']:
            return entity_data['labels'][language_code]['value']

    return None
    
field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/27062023_matched_fi_cze_sp_pl_ver_1.xlsx', sheet_name='Arkusz1',dtype=str)
lista_uri=field650['wiki_id'].tolist()  
final_dict={}
for uri in lista_uri:
    id_num=uri.split('/')[-1].strip("]'")
    label = get_wikidata_label(id_num)
    final_dict[uri]=label
    
fin_df=pd.DataFrame.from_dict(final_dict, orient='index')    
fin_df.to_excel("brakujacelabele.xlsx") 



# Replace 'Q798134' with the entity ID you want to retrieve the English label for
entity_id = 'Q798134'
label = get_wikidata_label(entity_id)

if label:
    print(f"The English label for {entity_id} is: {label}")
else:
    print(f"Entity ID '{entity_id}' not found or does not have an English label.")
    
#%%   

genre = pd.read_excel (r"D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/01082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze,fi).xlsx", sheet_name='Arkusz1')

ids=[]
for things in genre.fi_id.to_list():
    things=literal_eval(things) 
    if things:
        for thing in things:
            ids.append(thing) 
fin_dict={}     

      
for uri in tqdm(ids):
    #fin_dict[uri]=[]
      
    endpoint = "https://finto.fi/rest/v1/yso/data"
    params = {
        "uri": uri,
        "format": "application/ld+json"
    }
    
    response = requests.get(endpoint, params=params)

    data = response.json()

    
    
    # Extract close match from Wikidata
    graph_list = data["graph"]
    extracted_list=[]
    for graph in graph_list:
        if graph['uri']==uri:
            #print(graph)
            
            for g in graph['prefLabel']:
                if g['lang']=='en':
                    print(g['value'])
                    fin_dict[uri]=g['value']
excel=pd.DataFrame.from_dict(fin_dict, orient='index')
excel.to_excel('07082023_elb_en_fi_uzupelnienie.xlsx', sheet_name='new')                     