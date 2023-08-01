# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:18:19 2023

@author: dariu
"""

import requests
import pandas as pd
import re
from tqdm import tqdm
from ast import literal_eval
# DANE OD STRONY SLOWNIKOW plik wiki_sparql- od strony wikidata
#finowie
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
fin_dict={}     
for key, value in tqdm(new_dict_url.items()):
    print(value) 
    uri=value[0]
    fin_dict[uri]=[key,value[1]]
      


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
                fin_dict[uri].append(uris)
        else:
            if extracted_list[0].startswith("http://www.wikidata.org"):
                fin_dict[uri].append('brak')
                fin_dict[uri].append(extracted_list[0])
            else:
                fin_dict[uri].append(extracted_list[0])
                fin_dict[uri].append('brak')
                
                
                
                
                    
fin_df=pd.DataFrame.from_dict(fin_dict, orient='index')
fin_df.to_excel("finowie650_loc_wiki.xlsx")  
                       
#%% Czesch extractor
# first from file
import xml.etree.ElementTree as ET

# Load the MARCXML file
tree = ET.parse('D:/Nowa_praca/czech authority/aut_ph.xml')
root = tree.getroot()

# Define the namespace
namespace = {'marc': 'http://www.loc.gov/MARC21/slim'}

# Extract values from the MARCXML file
for record in root.findall('marc:record', namespace):
    controlfield = record.find('marc:controlfield[@tag="001"]', namespace).text
    leader = record.find('marc:leader', namespace).text
    
    datafields = record.findall('marc:datafield', namespace)
    for datafield in datafields:
        tag = datafield.get('tag')
        subfields = datafield.findall('marc:subfield', namespace)
        for subfield in subfields:
            code = subfield.get('code')
            value = subfield.text
            
            # Print the extracted values
            print(f'Tag: {tag}, Subfield Code: {code}, Value: {value}')

    print('---')  # Separate records with a line   
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
from pymarc import MARCReader,JSONReader
from tqdm import tqdm
from pymarc import Record, Field, Subfield



# html = '''
# <!-- filename: full-999-body-media -->
# <tr style="display:none" valign="top">
#   <td class=td1 id=bold width=15% nowrap>Další informace</td>
#   <td class=td1><A HREF='javascript:open_window("https://aleph.nkp.cz/F/7DVAU6AYKANNGTTC2E1JNXE7UXPCA4JI2AY81BRTH7ENGCIUN8-00502?func=service&doc_library=AUT10&doc_number=000119513&line_number=0001&func_code=WEB-FULL&service_type=MEDIA");'><img src="https://aleph.nkp.cz/exlibris/aleph/u22_1/alephe/www_f_cze/icon/f-tn-link.jpg" border=0 title="Typ souboru: url" alt=" ">Wikipedie (Divadlo)</a>&nbsp;</td>
# </tr>

# <!-- filename: full-999-body-media -->
# <tr style="display:none" valign="top">
#   <td class=td1 id=bold width=15% nowrap></td>
#   <td class=td1><A HREF='javascript:open_window("https://aleph.nkp.cz/F/7DVAU6AYKANNGTTC2E1JNXE7UXPCA4JI2AY81BRTH7ENGCIUN8-00503?func=service&doc_library=AUT10&doc_number=000119513&line_number=0002&func_code=WEB-FULL&service_type=MEDIA");'><img src="https://aleph.nkp.cz/exlibris/aleph/u22_1/alephe/www_f_cze/icon/f-tn-link.jpg" border=0 title="Typ souboru: url" alt=" ">Wikidata</a>&nbsp;</td>
# </tr>
# '''

#now from page
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
    sleep(5)
    URL=fr'https://aleph.nkp.cz/F/?func=find-c&local_base=aut&ccl_term=ica={id_num}'
    
    header={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
               'AppleWebKit/537.36 (KHTML, like Gecko) '\
               'Chrome/75.0.3770.80 Safari/537.36'}
    page=requests.get(URL, headers=header)
    #page=requests.get(URL)
                   
    soup = BeautifulSoup(page.content, 'html.parser')
    
    td_tags = soup.find_all('td', class_='td1')
    print(td_tags)
    
    # for td_tag in td_tags:
        
    #     if 'Wikidata' in td_tag.text:
    #         link = td_tag.find('a')['href']
    #         print(link)
    #         break
    for td_tag in td_tags:
        if 'Wikidata' in td_tag.text:
            link = re.search(r'"(https?://.*?)"', td_tag.find('a')['href']).group(1)
            #print(link)
            


            header={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
                       'AppleWebKit/537.36 (KHTML, like Gecko) '\
                       'Chrome/75.0.3770.80 Safari/537.36'}
            sleep(5)
            page=requests.get(link, headers=header)
           # page=requests.get(link)
                           
            soup = BeautifulSoup(page.content, 'html.parser')
            html_str = str(soup)
            
            
            match = re.search(r'onload=\'window.location.replace\("(.*?)"\)', html_str)
            if match:
                link = match.group(1)
                czeid[key].append(link)
                print(link)
            break
fin_df=pd.DataFrame.from_dict(czeid, orient='index')
fin_df.to_excel("cze_slownik_YSO_loc_wiki.xlsx") 
#%% Spain extractor
genre = pd.read_excel ('D:/Nowa_praca/Espana/650,655 staystyki_english_etc/words_650_stats.xlsx', sheet_name='Arkusz1')
list650=genre['field_650'].to_list()
list6501=[]
for l in tqdm(list650):
    list6501.append(compose_data(l))
s=set(list650)
my_marc_files = ["D:/Nowa_praca/Espana/wzorcówki/TITULO.mrc",
"D:/Nowa_praca/Espana/wzorcówki/CONGRESO.mrc",
"D:/Nowa_praca/Espana/wzorcówki/ENTIDAD.mrc",
"D:/Nowa_praca/Espana/wzorcówki/GENEROFORMA.mrc",
"D:/Nowa_praca/Espana/wzorcówki/GEOGRAFICO.mrc",
"D:/Nowa_praca/Espana/wzorcówki/GEOGRAFICO_ESP.mrc",
"D:/Nowa_praca/Espana/wzorcówki/MATERIA.mrc",
"D:/Nowa_praca/Espana/wzorcówki/MATERIASUB.mrc",
"D:/Nowa_praca/Espana/wzorcówki/PERSONA.mrc",
"D:/Nowa_praca/Espana/wzorcówki/PERSONATIT.mrc",
"D:/Nowa_praca/Espana/wzorcówki/SUBDIV.mrc"]
dictonario={}
for my_marc_file in my_marc_files:
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
           # print(record['001'].value())
          # print(record)
            my = record.get_fields('150','130','180','185','100','151','155','110','111')
            found_list=[]
            for field in my:
                
                check_a=field.get_subfields('a')
                check_x=field.get_subfields('x')
                check_v=field.get_subfields('v')
                check_k=field.get_subfields('k')
                
                if check_a:
                    for check in check_a:
                        if check in list6501:
                            found_list.append(check)
                            
                            
                elif check_x:
                    for check in check_x:
                        if check in list6501:
                            found_list.append(check)
                        
                elif check_v:
                    for check in check_v:
                        if check in list6501:
                            found_list.append(check)
                elif check_k:
                    for check in check_v:
                        if check in list6501:
                            found_list.append(check)
                   
            if found_list:
                loc_wiki_list=[] 
                field_024 = record.get_fields('024')
                
                for field in field_024:
                    sub_a=field.get_subfields('a')
                    for sub in sub_a:
                        if sub.startswith('http://id.loc.gov'):
                            loc_wiki_list.append(sub)
                           # print(sub)
                        if sub.startswith('https://www.wikidata.org'):
                            #print(sub)
                            loc_wiki_list.append(sub)
                        if sub.startswith('http://viaf'):
                           # print(sub)
                            loc_wiki_list.append(sub)
                if loc_wiki_list:
                    loc_wiki_list.sort()
                    
                    if found_list[0] in dictonario:
                        dictonario[found_list[0]][4]+=1
                        if dictonario[found_list[0]][1].startswith('http://id.loc.gov'):
                            pass
                        else:
                            for uris in loc_wiki_list:
                                
                                if uris.startswith('http://id.loc.gov'):
                                    dictonario[found_list[0]][1]=uris
                        if dictonario[found_list[0]][2].startswith('https://www.wikidata.org'):
                            pass
                        else:
                            for uris in loc_wiki_list:
                                 
                                if uris.startswith('https://www.wikidata.org'):
                                    print(uris)
                                    dictonario[found_list[0]][2]=uris 
                        if dictonario[found_list[0]][3].startswith('http://viaf'):
                            pass
                        else:
                            for uris in loc_wiki_list:
                                 
                                if uris.startswith('http://viaf'):
                                    dictonario[found_list[0]][3]=uris
                    else:
                        #dictonario[found_list[0]]=[record['001'].value(),'N/D','N/D','N/D',1]
                      #  
                            dictonario[found_list[0]]=[record['001'].value(),'N/D','N/D','N/D',1]
                            for uris in loc_wiki_list:
                                
                                if uris.startswith('http://id.loc.gov'):
                                    dictonario[found_list[0]][1]=uris 
                                if uris.startswith('https://www.wikidata.org'):
                                    print(uris)
                                    dictonario[found_list[0]][2]=uris 
                                if uris.startswith('http://viaf'):
                                    dictonario[found_list[0]][3]=uris
                else:
                    if found_list[0] in dictonario:
                        continue
                    else:
                        dictonario[found_list[0]]=[record['001'].value(),'N/D','N/D','N/D',1]
fin_df=pd.DataFrame.from_dict(dictonario, orient='index')
fin_df.to_excel("sp_slowniki_loc_wiki.xlsx")                     
omited=[]
for sth in list650:
    if sth not in dictonario:
        omited.append(sth)
        
        
        

                      
listaaa=['a','f','c']
listaaa.sort()
for a in listaaa:
    if a=='c':
        print(listaaa)
    if a=='a':
        print(listaaa)
slownik={'brawo':['lla','klaps']}

slownik['brawo'][0]='krak'               


#%%
#BROADER NARROWER EXTRACTIONS fin

import requests



def get_json_ld_data(url):
    endpoint = "https://finto.fi/rest/v1/yso/data"
    params = {
        "uri": url,
        "format": "application/ld+json"
    }

    try:
        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            json_ld_data = response.json()
            return json_ld_data
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return None

url='http://www.yso.fi/onto/yso/p712'
datas=get_json_ld_data(url)



import json

def extract_concepts(json_ld_data,uris):
    try:
        concepts = json_ld_data.get("graph", [])
    
        
    
        concepts_results={}
        for concept in concepts:
            uri = concept.get("uri")
            
            #print(uri2)
            
            broader = concept.get("broader", [])
            narrower = concept.get("narrower", [])
            pref_labels = concept.get("prefLabel", [])
            if uri:
                if uri==uris: 
                    concepts_results[uri]=[]
                    print(uri)
                    if isinstance(broader, dict):  # Single object
                        broader_uris={'broader':[broader.get("uri")]}
                        
                        concepts_results[uri].append(broader_uris)
                    elif isinstance(broader, list):  # List of objects
                        broader_uris = [broad.get("uri") for broad in broader]
                        concepts_results[uri].append({'broader':broader_uris})
                    if isinstance(narrower, dict):  # Single object
                        narrower_uris={'narrower':[narrower.get("uri")]}
                        concepts_results[uri].append(narrower_uris)
                    elif isinstance(narrower, list):  # List of objects
                        narrower_uris = [narrow.get("uri") for narrow in narrower]
                    
                        concepts_results[uri].append({'narrower':narrower_uris})
    except:
        return None
    return concepts_results

field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/27062023_matched_fi_cze_sp_pl_ver_1.xlsx', sheet_name='Sheet1',dtype=str)
lista_uri=set(field650['fi_id'].tolist())
#concepts = datas.get("graph", [])
#extract_concepts(datas)


new=[]
for uriss in lista_uri:
    uriss=uriss.strip("[]'")
    if len(uriss)>1:
        if uriss.startswith('http://www.yso.fi/onto/soto'):
            uriss=uriss.split(',')[1].strip(" []'")
        else:
            uriss_list=uriss.split(',')
            for uriss in uriss_list:
                uriss=uriss.strip(" []'")
            
        #print(uriss)
        datas=get_json_ld_data(uriss)
        new.append(extract_concepts(datas,uriss))
new_dict={}    
new_list=[]
for values in new:
    if values is not None:
        new_list.append(values)
        for key,value in values.items():
            new_dict[key]={"broader":[],"narrower":[]}
            new_dict[key]["broader"].extend(value[0]['broader'])
            new_dict[key]["narrower"].extend(value[1]['narrower'])
    


        
fin_df=pd.DataFrame(new_list)       
fin_df=pd.DataFrame.from_dict(new_dict, orient='index')
fin_df.to_excel("18072023fin_narrower_broader_uri.xlsx")  


#%%czech broader narrower 
import xml.etree.ElementTree as ET

# Load the MARCXML file
tree = ET.parse('D:/Nowa_praca/czech authority/aut_ph.xml')
root = tree.getroot()

# Define the namespace
namespace = {'marc': 'http://www.loc.gov/MARC21/slim'}

# Extract values from the MARCXML file all values
for record in root.findall('marc:record', namespace):
    controlfield = record.find('marc:controlfield[@tag="001"]', namespace).text
    leader = record.find('marc:leader', namespace).text
    
    datafields = record.findall('marc:datafield', namespace)
    for datafield in datafields:
        tag = datafield.get('tag')
        subfields = datafield.findall('marc:subfield', namespace)
        for subfield in subfields:
            code = subfield.get('code')
            value = subfield.text
            
            
            # Print the extracted values
            print(f'Tag: {tag}, Subfield Code: {code}, Value: {value}')

    print('---')  # Separate records with a line    
ns = {'marc': 'http://www.loc.gov/MARC21/slim'}

#extract only broader narrower, code::
field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/26072023_matched_fi_cze_sp_pl_(broader_narrower-yso).xlsx', sheet_name='Sheet1',dtype=str)
lista_uri=set(field650['cze_id'].tolist())    
lista_id=[]
for idis in lista_uri:
    for ids in literal_eval(idis):
        print(ids)
        lista_id.append(ids)
    
tree = ET.parse('D:/Nowa_praca/czech authority/aut_ph.xml')
root = tree.getroot()
    # Find all records
records = root.findall('.//marc:record', ns)   
#lista_id=['ph124475']
slownik_broader_narrower={}
for ids in lista_id:
    slownik_broader_narrower[ids]={"narrower":[], 'broader':[]}
    for record in records:
        controlfield = record.find('marc:controlfield[@tag="001"]', ns)
        if controlfield is not None and controlfield.text == ids:
            # Extract the 'afirmace' and 'přerámování (psychologie)' terms
            datafields = record.findall(".//marc:datafield[@tag='550']", ns)
            for datafield in datafields:
                w_subfield = datafield.find("marc:subfield[@code='w']", ns)
                a_subfield = datafield.find("marc:subfield[@code='a']", ns)
                subfield_7=datafield.find("marc:subfield[@code='7']", ns)
                #narrower h
                if w_subfield is not None and w_subfield.text == 'h' and subfield_7 is not None:
                    
                    slownik_broader_narrower[ids]["narrower"].append(subfield_7.text)
    
                #broader g
                if w_subfield is not None and w_subfield.text == 'g' and a_subfield is not None:
                    slownik_broader_narrower[ids]["broader"].append(subfield_7.text)

fin_df=pd.DataFrame.from_dict(slownik_broader_narrower, orient='index')
fin_df.to_excel("26072023cze_narrower_broader_id.xlsx")  
#%%fin broader narrower 
# =555  \\$wg broader,   =555  \\$wh narrower

field650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/31072023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze).xlsx', sheet_name='Sheet1',dtype=str)
lista_uri=set(field650['esp_id'].tolist())
dictionary=field650.to_dict('records')

lista_id=[]
for idis in lista_uri:
    for ids in literal_eval(idis):
        if ids:
            lista_id.append(ids)
            print(ids)
my_marc_files = ["D:/Nowa_praca/Espana/wzorcówki/TITULO.mrc",
"D:/Nowa_praca/Espana/wzorcówki/CONGRESO.mrc",
"D:/Nowa_praca/Espana/wzorcówki/ENTIDAD.mrc",
"D:/Nowa_praca/Espana/wzorcówki/GENEROFORMA.mrc",
"D:/Nowa_praca/Espana/wzorcówki/GEOGRAFICO.mrc",
"D:/Nowa_praca/Espana/wzorcówki/GEOGRAFICO_ESP.mrc",
"D:/Nowa_praca/Espana/wzorcówki/MATERIA.mrc",
"D:/Nowa_praca/Espana/wzorcówki/MATERIASUB.mrc",
"D:/Nowa_praca/Espana/wzorcówki/PERSONA.mrc",
"D:/Nowa_praca/Espana/wzorcówki/PERSONATIT.mrc",
"D:/Nowa_praca/Espana/wzorcówki/SUBDIV.mrc"]            
from pymarc import MARCReader

sp_broader_narrower={}
for file in my_marc_files:
    with open(file, 'rb') as fh:
        reader = MARCReader(fh)
        for record in tqdm(reader):
            field_001 = record['001'].data
            if field_001 in lista_id:
               
                
                
                field_555 = record.get_fields('555', '550')
                if field_555:
                    broader=[]
                    narrower=[]
                    
                    for field in field_555:
                        subfield_w = field.get_subfields('w')
                        subfield_a = field.get_subfields('a')
                   
                        for value in subfield_w:
                            
                            if value == 'g':
                                print("Broader terms:", subfield_a)
                                broader.extend(subfield_a)
                                
                            elif value == 'h':
                                print("Narrower terms:", subfield_a)
                                narrower.extend(subfield_a)
                                
                    if broader or narrower:
                        sp_broader_narrower[field_001]={"narrower":[], 'broader':[]}
                        
                        sp_broader_narrower[field_001]['broader'].extend(broader)
                        sp_broader_narrower[field_001]['narrower'].extend(narrower)
fin_df=pd.DataFrame.from_dict(sp_broader_narrower, orient='index')
fin_df.to_excel("31072023esp_narrower_broader_id.xlsx")   
#check for sp id

name_id={}
for my_marc_file in my_marc_files:
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
           # print(record['001'].value())
          # print(record)
            my = record.get_fields('150','130','180','185','100','151','155','110','111')
            found_list=[]
            for field in my:
                
                check_a=field.get_subfields('a')
                if check_a:
                    name_id[check_a[0]]= record['001'].data
                    
                    # field_024 = record.get_fields('024')
                    
                    # for field in field_024:
                    #     sub_a=field.get_subfields('a')
                    #     for sub in sub_a:
                    #         if sub.startswith('https://www.wikidata.org'):
                    #             name_id[check_a[0]]=sub
                    #             break
                    #         elif sub.startswith('http://id.loc.gov'):
                                
                    #             name_id[check_a[0]]=sub
                                
                
                

for key, val in sp_broader_narrower.items():
    narrower_list=[]
    broader_list=[]
    for narrower in val['narrower']:
        if narrower in name_id:
            narrower_list.append(name_id[narrower])
    for broader in val['broader']:
        if broader in name_id:
            broader_list.append(name_id[broader])
    
    
    sp_broader_narrower[key]['narrower']=narrower_list
    sp_broader_narrower[key]['broader']=broader_list
    
                         
fin_df=pd.DataFrame.from_dict(sp_broader_narrower, orient='index')
fin_df.to_excel("01082023esp_narrower_broader_id.xlsx")        



                        



