# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:50:40 2023

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
from ast import literal_eval
from pathlib import Path

with open('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/01082023sp_broader_narrower_labels.json', encoding='utf-8') as user_file:
  parsed_json = json.load(user_file)  
  
field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/08082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze)_FINAL.xlsx', sheet_name='Sheet1',dtype=str)
dictionary_dataframe=field650.to_dict('records')
labels=field650['wiki_id'].tolist() 

labels_list=[]
for label in labels:
    for lab in literal_eval(label):
        lab=lab.split(',')
        for l in lab:
        
            if type(l) is not float:
                labels_list.append(l)
#label-broader, yso-narrower
matching={}
for key, val in parsed_json.items():
    for broader_labels in val['broader']:
        
            try:
                #broader_lower=broader_labels[1][1].lower()
                if broader_labels[1][0] in labels_list:
                    if key not in matching:
                        matching[key]={'broader_id_wiki':[]}
                        matching[key]['broader_id_wiki']=[broader_labels[1][0]]
                    else:
                        matching[key]['broader_id_wiki'].append(broader_labels[1][0])
            except:
                pass

fin_df=pd.DataFrame.from_dict(matching, orient='index')
fin_df.to_excel("label-broader_ph-narrower.xlsx")
#label-narrower, yso broader



matchingnarrower={}
for key, val in parsed_json.items():
    for narrower_labels in val['narrower']:
        try:
            #broader_lower=broader_labels[1][1].lower()
            if narrower_labels[1][0] in labels_list:
                if key not in matchingnarrower:
                    matchingnarrower[key]={'narrower_id_wiki':[]}
                    matchingnarrower[key]['narrower_id_wiki']=[narrower_labels[1][0]]
                else:
                    matchingnarrower[key]['narrower_id_wiki'].append(narrower_labels[1][0])
        except:
            pass
fin_df=pd.DataFrame.from_dict(matchingnarrower, orient='index')
fin_df.to_excel("label-narrower_yso-broader.xlsx")
        
        
           

 #   jeśli broader się gdzie pojawi to jego narrower jest narrowerem dla labela
  #  jeli narrower się pojawi to jego broader jest broaderem dla labela 
  
#%% PART2 do broaderow i narrowerow ze slownika finto doszly dane (wiki_id) z innych slownikow teraz zrobię z nich jeden słownik którego podstawą będzie broader label
#usunę duplikaty i wrzucę do głównego słownika


#do yso dodaje label i id_label:
labels_fi_id=dict(zip(field650['libri_id'].tolist(), field650['fi_id'].tolist()))
dictionary_dataframe=field650.to_dict('records')
    
        
#label-narrower_yso-broader:
counter=0   
for key, val in labels_fi_id.items():
    for v in literal_eval(val):
        for keys, values in matchingnarrower.items():
            if v=='http://www.yso.fi/onto/yso/p1775':
                print(key)
  
            if keys==v:
                print(key)
                matchingnarrower[keys]['broader_id']=key
                for elements in dictionary_dataframe:
                    if elements['libri_id']==key:
                        matchingnarrower[keys]['broader_wiki_label']=elements['en_wiki_label']
                        matchingnarrower[keys]['broader_wiki_id']=literal_eval(elements['wiki_id'])
                        
                
        
                    

    
    
fin_df=pd.DataFrame.from_dict(matchingnarrower, orient='index')
fin_df.to_excel("label-narrower_yso-broader(withlabel_id).xlsx")

#label-broader_yso-narrower               
for key, val in labels_fi_id.items():
    for v in literal_eval(val):
        for keys, values in matching.items():
            if keys==v:
                print(key)
                matching[keys]['narrower_id']=key
                for elements in dictionary_dataframe:
                    if elements['libri_id']==key:
                        matching[keys]['narrower_wiki_label']=elements['en_wiki_label']
                        matching[keys]['narrower_wiki_id']=literal_eval(elements['wiki_id'])
                        
                        

fin_df=pd.DataFrame.from_dict(matching, orient='index')
fin_df.to_excel("label-broader_yso-narrower(wthlabel_id).xlsx")      

#z label-broader_yso_narrower robię słownik- likwiduje wielosc broader w kolumnach i robie slownik
#broader= i lista narrower- analogicznie do matchingnarrower (to jest gotowe) potem polacze narrower i wyrzuce duplikaty wszystko po wikidacie
broader={}          
          
for key, val in matching.items():
    for v in val['broader_id_wiki']:
        if v not in broader:
            
            
            
            broader[v]=val['narrower_wiki_id']
        else:
            broader[v].extend(val['narrower_wiki_id'])
            
          
         
for key, val in matchingnarrower.items():
    for v in val['broader_wiki_id']:
        if v not in broader:
            
            
            
            broader[v]=val['narrower_id_wiki']
        else:
            broader[v].extend(val['narrower_id_wiki'])
#po zsumowaniu dwoch podejsc (do broaderow narrowery z labeli i na odwrót w jeden slownik) unique na narrowerach
unique_broader_narrower={}
for key, val in broader.items():
    unique(val)
    unique_broader_narrower[key]=val
    
#klucz jako nasz id klucz broader wartosci narrower
broader_elb_id={}
for x in dictionary_dataframe:
    wiki_id=literal_eval(x['wiki_id'])
    
    
    for ids in wiki_id:
        
        v=broader.get(ids)
        if v:
            
            el_id=[]
            for element in v:
                for el in dictionary_dataframe:
                    if element in literal_eval(el['wiki_id']):
                      el_id.append(el["libri_id"])
            broader_elb_id[x['libri_id']]=(ids,v,el_id)         
            break          
            
                        
                        
              
            #     if element in wiki_id:
            #         elb_id.append(x['libri_id'])
            # broader_elb_id[x['libri_id']]=(*broader_elb_id[x['libri_id']],elb_id)
                    
                    

fin_df=pd.DataFrame.from_dict(broader_elb_id, orient='index')   
fin_df.to_excel("key-broader_val-narrower.xlsx")   
#klucz narrower wartosc broader
narrower_elb_id={}
for k,v in broader_elb_id.items():
      for index,elb_id in enumerate(v[2]):
          if elb_id not in narrower_elb_id:
              narrower_elb_id[elb_id]={}
              narrower_elb_id[elb_id]['broader']=[]
              narrower_elb_id[elb_id]['broader']=[k]
          else:
              narrower_elb_id[elb_id]['broader'].append(k)
              
fin_df2=pd.DataFrame.from_dict(narrower_elb_id, orient='index')  
fin_df2.to_excel("key-narrower_val-broader.xlsx")   
#%% TO SAMO CO W PART2 tylko dla czechów
labels_fi_id=dict(zip(field650['libri_id'].tolist(), field650['cze_id'].tolist()))
dictionary_dataframe=field650.to_dict('records')
broader_narrower_excel=dict(zip(field650['libri_id'].tolist(), field650['narrower'].tolist()))
narrower_broader_excel=dict(zip(field650['libri_id'].tolist(), field650['broader'].tolist()))        
#label-narrower_yso-broader:
counter=0   
for key, val in labels_fi_id.items():
    for v in literal_eval(val):
        for keys, values in matchingnarrower.items():
            if v=='http://www.yso.fi/onto/yso/p1775':
                print(key)
  
            if keys==v:
                print(key)
                matchingnarrower[keys]['broader_id']=key
                for elements in dictionary_dataframe:
                    if elements['libri_id']==key:
                        matchingnarrower[keys]['broader_wiki_label']=elements['elb_concept']
                        matchingnarrower[keys]['broader_wiki_id']=literal_eval(elements['wiki_id'])
                        
                
        
                    

    
    
fin_df=pd.DataFrame.from_dict(matchingnarrower, orient='index')
fin_df.to_excel("label-narrower_yso-broader(withlabel_id).xlsx")

#label-broader_yso-narrower               
for key, val in labels_fi_id.items():
    for v in literal_eval(val):
        for keys, values in matching.items():
            if keys==v:
                print(key)
                matching[keys]['narrower_id']=key
                for elements in dictionary_dataframe:
                    if elements['libri_id']==key:
                        matching[keys]['narrower_wiki_label']=elements['elb_concept']
                        matching[keys]['narrower_wiki_id']=literal_eval(elements['wiki_id'])
                        
                        

fin_df=pd.DataFrame.from_dict(matching, orient='index')
fin_df.to_excel("label-broader_yso-narrower(wthlabel_id).xlsx")      

#z label-broader_yso_narrower robię słownik- likwiduje wielosc broader w kolumnach i robie slownik
#broader= i lista narrower- analogicznie do matchingnarrower (to jest gotowe) potem polacze narrower i wyrzuce duplikaty wszystko po wikidacie
broader={}          
          
for key, val in matching.items():
    for v in val['broader_id_wiki']:
        if v not in broader:
            
            
            
            broader[v]=val['narrower_wiki_id']
        else:
            broader[v].extend(val['narrower_wiki_id'])
            
          
         
for key, val in matchingnarrower.items():
    for v in val['broader_wiki_id']:
        if v not in broader:
            
            
            
            broader[v]=val['narrower_id_wiki']
        else:
            broader[v].extend(val['narrower_id_wiki'])
#po zsumowaniu dwoch podejsc (do broaderow narrowery z labeli i na odwrót w jeden slownik) unique na narrowerach
unique_broader_narrower={}
for key, val in broader.items():
    unique(val)
    unique_broader_narrower[key]=val
    
#klucz jako nasz id klucz broader wartosci narrower
broader_elb_id={}
for x in tqdm(dictionary_dataframe):
    wiki_id=literal_eval(x['wiki_id'])
    
    
    for ids in wiki_id:
        
        v=broader.get(ids)
        if v:
            
            el_id=[]
            for element in v:
                for el in dictionary_dataframe:
                    if element in literal_eval(el['wiki_id']):
                      el_id.append(el["libri_id"])
            broader_elb_id[x['libri_id']]=(ids,v,el_id)         
            break          
            
                        
                        
              
            #     if element in wiki_id:
            #         elb_id.append(x['libri_id'])
            # broader_elb_id[x['libri_id']]=(*broader_elb_id[x['libri_id']],elb_id)
                    
                    

fin_df=pd.DataFrame.from_dict(broader_elb_id, orient='index')   
fin_df.to_excel("key-broader_val-narrower.xlsx")   
#klucz narrower wartosc broader
narrower_elb_id={}
for k,v in broader_elb_id.items():
      for index,elb_id in enumerate(v[2]):
          if elb_id not in narrower_elb_id:
              narrower_elb_id[elb_id]={}
              narrower_elb_id[elb_id]['broader']=[]
              narrower_elb_id[elb_id]['broader']=[k]
          else:
              narrower_elb_id[elb_id]['broader'].append(k)
              
fin_df2=pd.DataFrame.from_dict(narrower_elb_id, orient='index')  
fin_df2.to_excel("key-narrower_val-broader.xlsx")   
# dodanie istniejących narrow i broader do nowych i unique
#klucz narower wartość broader
narrower_fi_cze={}
for narrow_excel, broader_excel in narrower_broader_excel.items():
    if type(broader_excel) is not float:
        excel_broader=literal_eval(broader_excel)
        narrower_fi_cze[narrow_excel]=excel_broader
for key,val in narrower_elb_id.items():
    if key not in narrower_fi_cze:
        narrower_fi_cze[key]=val['broader']
    else:
       value=val['broader']+narrower_fi_cze[key]
       unique(value)                 
       narrower_fi_cze[key]=value
       
df = pd.DataFrame(list(narrower_fi_cze.items()), columns=['narrower', 'broader'])
fin_df2=pd.DataFrame.from_dict(narrower_fi_cze, orient='index')  
df.to_excel("key-narrower_val-broader_cze_fin.xlsx")   
#klucz broader                
broader_fi_cze={}
for broader_excel, narrower_excel in broader_narrower_excel.items():
    if type(narrower_excel) is not float:
        excel_narrower=literal_eval(narrower_excel)
        broader_fi_cze[broader_excel]=excel_narrower
for key,val in broader_elb_id.items():
    if key not in broader_fi_cze:
        broader_fi_cze[key]=val[2]
    else:
       print(key)
       value=val[2]+broader_fi_cze[key]
       unique(value)                 
       broader_fi_cze[key]=value             
df = pd.DataFrame(list(broader_fi_cze.items()), columns=['broader', 'narrower'])               
fin_df2=pd.DataFrame.from_dict(broader_fi_cze, orient='index')  
df.to_excel("key-broader_val-narrower_cze_fin.xlsx")           


dictionary={'lolo':[1,2,3],'polo':[4,5,6],'kolo':[7,9,0]} 

dictionary2={'ronnie':[1,2,3,4],'polo':[4,5,6]} 
new={}
for key, val in dictionary.items():
    new[key]=val
    for key1, val1 in dictionary2.items():
        print(key,key1)
        if key1 not in new:
            new[key1]=val1
        else:
            print("key111::::", key1)

#klucz broader wartosć narrower/broader poprawa dwóch tabelek:
    
key_broader_val_narrower=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/28072023key-narrower_val-broader_cze_fin_ver_final.xlsx', sheet_name='Sheet1',dtype=str)
dict_from_df = key_broader_val_narrower.set_index(key_broader_val_narrower.columns[0]).T.to_dict('list')
new_dict = {key: {'broader': [i for i in value if pd.notnull(i)]} for key, value in dict_from_df.items()}
fin_df=pd.DataFrame.from_dict(new_dict, orient='index')

fin_df.to_excel("key-narrower_val-broader_cze_fin_ver2.xlsx")  

#%% PART 2 dla sp         
labels_fi_id=dict(zip(field650['libri_id'].tolist(), field650['esp_id'].tolist()))
dictionary_dataframe=field650.to_dict('records')
broader_narrower_excel=dict(zip(field650['libri_id'].tolist(), field650['narrower'].tolist()))
narrower_broader_excel=dict(zip(field650['libri_id'].tolist(), field650['broader'].tolist()))        
#label-narrower_yso-broader:
counter=0   
for key, val in labels_fi_id.items():
    for v in literal_eval(val):
        for keys, values in matchingnarrower.items():
            if v=='http://www.yso.fi/onto/yso/p1775':
                print(key)
  
            if keys==v:
                print(key)
                matchingnarrower[keys]['broader_id']=key
                for elements in dictionary_dataframe:
                    if elements['libri_id']==key:
                        matchingnarrower[keys]['broader_wiki_label']=elements['elb_concept']
                        matchingnarrower[keys]['broader_wiki_id']=literal_eval(elements['wiki_id'])
                        
                
        
                    

    
    
fin_df=pd.DataFrame.from_dict(matchingnarrower, orient='index')
fin_df.to_excel("label-narrower_yso-broader(withlabel_id).xlsx")

#label-broader_yso-narrower               
for key, val in labels_fi_id.items():
    for v in literal_eval(val):
        for keys, values in matching.items():
            if keys==v:
                print(key)
                matching[keys]['narrower_id']=key
                for elements in dictionary_dataframe:
                    if elements['libri_id']==key:
                        matching[keys]['narrower_wiki_label']=elements['elb_concept']
                        matching[keys]['narrower_wiki_id']=literal_eval(elements['wiki_id'])
                        
                        

fin_df=pd.DataFrame.from_dict(matching, orient='index')
fin_df.to_excel("label-broader_yso-narrower(wthlabel_id).xlsx")      

#z label-broader_yso_narrower robię słownik- likwiduje wielosc broader w kolumnach i robie slownik
#broader= i lista narrower- analogicznie do matchingnarrower (to jest gotowe) potem polacze narrower i wyrzuce duplikaty wszystko po wikidacie
broader={}          
          
for key, val in matching.items():
    for v in val['broader_id_wiki']:
        if v not in broader:
            
            
            
            broader[v]=val['narrower_wiki_id']
        else:
            broader[v].extend(val['narrower_wiki_id'])
            
          
         
for key, val in matchingnarrower.items():
    for v in val['broader_wiki_id']:
        if v not in broader:
            
            
            
            broader[v]=val['narrower_id_wiki']
        else:
            broader[v].extend(val['narrower_id_wiki'])
#po zsumowaniu dwoch podejsc (do broaderow narrowery z labeli i na odwrót w jeden slownik) unique na narrowerach
unique_broader_narrower={}
for key, val in broader.items():
    unique(val)
    unique_broader_narrower[key]=val
    
#klucz jako nasz id klucz broader wartosci narrower
broader_elb_id={}
for x in dictionary_dataframe:
    wiki_id=literal_eval(x['wiki_id'])
    
    
    for ids in wiki_id:
        
        v=broader.get(ids)
        if v:
            
            el_id=[]
            for element in v:
                for el in dictionary_dataframe:
                    if element in literal_eval(el['wiki_id']):
                      el_id.append(el["libri_id"])
            broader_elb_id[x['libri_id']]=(ids,v,el_id)         
            break          
            
                        
                        
              
            #     if element in wiki_id:
            #         elb_id.append(x['libri_id'])
            # broader_elb_id[x['libri_id']]=(*broader_elb_id[x['libri_id']],elb_id)
                    
                    

fin_df=pd.DataFrame.from_dict(broader_elb_id, orient='index')   
fin_df.to_excel("key-broader_val-narrower.xlsx")   
#klucz narrower wartosc broader
narrower_elb_id={}
for k,v in broader_elb_id.items():
      for index,elb_id in enumerate(v[2]):
          if elb_id not in narrower_elb_id:
              narrower_elb_id[elb_id]={}
              narrower_elb_id[elb_id]['broader']=[]
              narrower_elb_id[elb_id]['broader']=[k]
          else:
              narrower_elb_id[elb_id]['broader'].append(k)
              
fin_df2=pd.DataFrame.from_dict(narrower_elb_id, orient='index')  
fin_df2.to_excel("key-narrower_val-broader.xlsx")   
# dodanie istniejących narrow i broader do nowych i unique
#klucz narower wartość broader
narrower_fi_cze={}
for narrow_excel, broader_excel in narrower_broader_excel.items():
    if type(broader_excel) is not float:
        excel_broader=literal_eval(broader_excel)
        narrower_fi_cze[narrow_excel]=excel_broader
for key,val in narrower_elb_id.items():
    if key not in narrower_fi_cze:
        narrower_fi_cze[key]=val['broader']
    else:
       value=val['broader']+narrower_fi_cze[key]
       unique(value)                 
       narrower_fi_cze[key]=value
       
df = pd.DataFrame(list(narrower_fi_cze.items()), columns=['narrower', 'broader'])
  
df.to_excel("01082023key-narrower_val-broader_cze_fin_sp.xlsx")   
#klucz broader                
broader_fi_cze={}
for broader_excel, narrower_excel in broader_narrower_excel.items():
    if type(narrower_excel) is not float:
        excel_narrower=literal_eval(narrower_excel)
        broader_fi_cze[broader_excel]=excel_narrower
for key,val in broader_elb_id.items():
    if key not in broader_fi_cze:
        broader_fi_cze[key]=val[2]
    else:
       value=val[2]+broader_fi_cze[key]
       unique(value)                 
       broader_fi_cze[key]=value             
                
df = pd.DataFrame(list(broader_fi_cze.items()), columns=['broader', 'narrower'])
  
df.to_excel("01082023key-broader_val-narrower_cze_fin_sp.xlsx")                  
#%% dla pomocy zamiast id całe koncepty
field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/08082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze,esp)_FINAL.xlsx', sheet_name='Sheet1',dtype=str)
dictionary_dataframe=field650.to_dict('records')  
concept={}
for data in tqdm(dictionary_dataframe):
    if type (data['broader']) is not float:
        
        list_broader=literal_eval(data['broader'])
        concept[data['libri_id']]={'broader':[]}
        for broader in list_broader:
            
            for data1 in dictionary_dataframe:
                
                
                if broader==data1['libri_id']:
                    print(data['elb_concept'])
                    print(data1['elb_concept'])
                    concept[data['libri_id']]['broader'].append(data1['elb_concept'])
            
fin_df=pd.DataFrame.from_dict(concept, orient='index')   
fin_df.to_excel("key-narrower_val-broader.xlsx")                
            
            
        