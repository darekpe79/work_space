# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:18:19 2023

@author: dariu
"""

import requests
import pandas as pd
import re
field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/650__do_pracy_wszystko.xlsx', sheet_name='fin2',dtype=str)
listy=dict(zip(field650['desk_650'].to_list(),field650['tonew650_national'].to_list()))
dictionary=field650.to_dict('records')
patterna2=r"\$a(.+?)\$|\$a(.+$)"
pattern1=r"\$a(.+?)(?:\$|$)"
patterna = r"\$a([^$]*)"
pattern = r"\$0(.+)"

pattern = r"\$0(.+)"
new_dict_url={}
for elem in dictionary:
    desk_650=elem['desk_650']
    
    matchurl = re.search(pattern, desk_650)
    matcha=re.search(patterna, desk_650)

    if matchurl:
        url = matchurl.group(1)
        print("URL:", url)
        sub_a=matcha.group(1)
        new_dict_url[desk_650]=[url, sub_a]
fin_dict      
for key, value in new_dict_url.items():
    print(value) 
    uri=value[0]      


    uri="http://www.yso.fi/onto/yso/p10123"
        
    endpoint = "https://finto.fi/rest/v1/yso/data"
    params = {
        "uri": uri,
        "format": "application/ld+json"
    }
    
    response = requests.get(endpoint, params=params)
    data = response.json()
    response.url
    
    # Extract close match from Wikidata
    graph_list = data["graph"]
    for graph in graph_list:
        
        if graph['uri']==uri:
            
            close_matches=graph.get('closeMatch')
            if close_matches:
                if type(close_matches) is list:
                    for match in close_matches:
                        if match['uri'].startswith("http://www.wikidata.org"):
                            print(match['uri'])
                        if match['uri'].startswith("http://id.loc.gov"):
                            print(match['uri'])
                      
                else:
                    if close_matches['uri'].startswith("http://www.wikidata.org"):
                        print(close_matches['uri'])
                    if close_matches['uri'].startswith("http://id.loc.gov"):
                        
                    

print("Close match from Wikidata:", close_match)

print("Close match from Wikidata:", close_match)