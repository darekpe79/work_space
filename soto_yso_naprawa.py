# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:26:09 2023

@author: dariu
"""

import requests
import pandas as pd
import re
from tqdm import tqdm
from definicje import *
# DANE OD STRONY SLOWNIKOW plik wiki_sparql- od strony wikidata
#finowie
field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/07062023finowie_650_zebrane_do_maczowania.xlsx', sheet_name='Sheet1',dtype=str)

dictionary=field650.to_dict('records')
patterna2=r"\$a(.+?)\$|\$a(.+$)"
pattern1=r"\$a(.+?)(?:\$|$)"
patterna = r"\$a([^$]*)"
pattern = r"\$0(.+)"

pattern = r"\$0(.+)"

#extract wrong soto
new_dict_url={}
for elem in dictionary:
    desk_650=elem['field_650']
    
    matchurl = re.search(pattern, desk_650)
    matcha=re.search(patterna, desk_650)

    if matchurl:
        url = matchurl.group(1)
        if url.startswith('http://www.yso.fi/onto/soto'):
            
            print("URL:", url)
            sub_a=matcha.group(1)
            new_dict_url[desk_650]=[url, sub_a]


#extract yso for wrong soto from soto dict  
fin_dict1={}     
for key, value in tqdm(new_dict_url.items()):
    print(value) 
    uri=value[0]
    fin_dict1[uri]=[key,value[1]]
      


    #uri="http://www.yso.fi/onto/yso/p10123"
    #fin_dict1[uri]=[]
        
    endpoint = "https://finto.fi/rest/v1/soto/data"
    params = {
        "uri": uri,
        "format": "application/ld+json"
    }
    
    response = requests.get(endpoint, params=params)
    try:
        data = response.json()
    except:
        continue
    
    
    # Extract close match from Wikidata
    graph_list = data["graph"]
    extracted_list=[]
    for graph in graph_list:
        
        if graph['uri']==uri:
            
            close_matches=graph.get('exactMatch')
            if close_matches:
                if type(close_matches) is list:
                    
                    for match in close_matches:
                        
                            
                            extracted_list.append(match['uri'])
                        # else:
                        #     fin_dict1[uri].insert(0,'brak')
                else:
                    extracted_list.append(close_matches['uri'])
                     
           
                            
  
    if extracted_list:
        if len(extracted_list)>1:
            extracted_list.sort()
            
            for uris in extracted_list:
                fin_dict1[uri].append(uris)
        else:
            if extracted_list[0].startswith("http://www.wikidata.org"):
                fin_dict1[uri].append('brak')
                fin_dict1[uri].append(extracted_list[0])
            else:
                fin_dict1[uri].append(extracted_list[0])
               
              
#extract wiki and loc from yso dict                 
fin_dict={}     
for key, value in tqdm(fin_dict1.items()):
   #print(value)
    if len(value)>2:
        print(value[2])
        uri=value[2]
        fin_dict[key]=[value[1],uri]
      


    #uri="http://www.yso.fi/onto/yso/p10123"
    #fin_dict[uri]=[]
        
        endpoint = "https://finto.fi/rest/v1/yso/data"
        params = {
            "uri": uri,
            "format": "application/ld+json"
        }
        
        response = requests.get(endpoint, params=params)
        try:
            data = response.json()
        except:
            continue
        
        
        # Extract close match from Wikidata
        graph_list = data["graph"]
        extracted_list=[]
        for graph in graph_list:
            
            if graph['uri']==uri:
                
                close_matches=graph.get('closeMatch')
                if close_matches:
                    if type(close_matches) is list:
                        
                        for match in close_matches:
                            if match['uri'].startswith("http://www.wikidata.org"):
                                
                                extracted_list.append(match['uri'])
                            # else:
                            #     fin_dict[uri].insert(0,'brak')
                                
                            if match['uri'].startswith("http://id.loc.gov"):
                                extracted_list.append(match['uri'])
                            # else:
                            #     fin_dict[uri].insert(1,'brak')
                                
                          
                    else:
                        if close_matches['uri'].startswith("http://www.wikidata.org"):
                            print(close_matches['uri'])
                            extracted_list.append(close_matches['uri'])
    
                            
                        if close_matches['uri'].startswith("http://id.loc.gov"):
                            extracted_list.append(close_matches['uri'])
        if extracted_list:
            if len(extracted_list)>1:
                extracted_list.sort()
                
                for uris in extracted_list:
                    fin_dict[key].append(uris)
            else:
                if extracted_list[0].startswith("http://www.wikidata.org"):
                    fin_dict[key].append('brak')
                    fin_dict[key].append(extracted_list[0])
                else:
                    fin_dict[key].append(extracted_list[0])
                    fin_dict[key].append('brak')
                
#wiki label wyciagniecie wiki sparql:
finid={}
for key,val in tqdm(fin_dict.items()):
    finid[key]=[]
    id_num=val[1].split('/')[-1][1:]

    #sleep(0.1)
    
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
                
czech_df=pd.DataFrame.from_dict(finid, orient='index')
czech_df.to_excel("soto-yso-wiki-loc_SPARQL.xlsx")                 
               