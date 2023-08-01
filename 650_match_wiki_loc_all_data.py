# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:22:05 2023

@author: dariu
"""

import pandas as pd
import re
from tqdm import tqdm
from definicje import *
from ast import literal_eval
# DANE OD STRONY SLOWNIKOW plik wiki_sparql- od strony wikidata
#finowie
fin650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/07062023finowie_650_zebrane_do_maczowania_ver2.xlsx', sheet_name='Sheet1',dtype=str)
cze650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/14062023czesi__650_zebrane_do_maczowania_ver.2.xlsx', sheet_name='Sheet1',dtype=str)
esp650=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/14062023hiszpanie_650_zebrane_do_maczowania_ver.2.xlsx', sheet_name='Sheet1',dtype=str)
# set_finowie=set(fin650.Wiki.to_list())
# set_czesi=set(cze650.Wiki.to_list())
# set_esp=set(esp650.Wiki.to_list())
# set_finowie_loc=set(fin650.LOC_id.to_list())
# set_czesi_loc=set(cze650.LOC_id.to_list())
# set_esp_loc=set(esp650.LOC_id.to_list())
fin_dictionary=fin650.to_dict('records')
cze_dictionary=cze650.to_dict('records')
esp_dictionary=esp650.to_dict('records')
#new approach
matched={}
dowykluczenia_cze={}
counter=0
for elements in tqdm(fin_dictionary):
    found_match = False  # Flag to track if a match is found
    
    # counter+=1
    # if counter==1000:
    #     break
    matched[elements['field_650']]={'en_wiki_label': elements['en_label'],'wiki_id':[], 'loc_id':[],'fin_a':elements['subfield_a'],'field_a_fin':elements['field_650'] }
    
    for key, val in elements.items():
        
        if key=='Wiki':
            if type(val)==float:
                
                continue
            finlist=val.split(',')
            for fin_elem in finlist:
                if fin_elem.startswith('http://id.loc.gov'):
                    fin_elem=fin_elem.split(r'/')[-1]
                    matched[elements['field_650']]['loc_id'].append(fin_elem)
                    
                    
                    continue
                
                if fin_elem=="N/A" or fin_elem=='0'or fin_elem=='brak':
                    continue
                else:
                
                    matched[elements['field_650']]['wiki_id'].append(fin_elem)
                    
        
        if key=='LOC_id':
            if type(val)==float:
                continue
            
            #print(sets)
            Loclist=val.split(',')
            for loc in Loclist:
                if loc=="N/A" or loc=='0'or loc=='brak':
                    continue
                if loc.startswith('http://www.wikidata.or'):
                    pass
                if loc.startswith('http'):
                    
                    loc=loc.split(r'/')[-1]
                    
                    matched[elements['field_650']]['loc_id'].append(loc)
                
                else:
                    
                    matched[elements['field_650']]['loc_id'].append(loc)
                    
    for elements2 in cze_dictionary:
        #found_match = False
        
        
        for key, val in elements2.items():
            
            if key == 'Wiki':
                if type(val) == float:
                    continue
               # print('key: ',key,'value: ', val)
                finlist = val.split(',')
                for fin_elem in finlist:
                    if fin_elem.startswith('http://id.loc.gov'):
                        fin_elem = fin_elem.split(r'/')[-1]
                        if fin_elem in matched[elements['field_650']]['loc_id']:
                            dowykluczenia_cze[elements2['field_650']]='znalezione'
                            counter+=1
                            matched[elements['field_650']]['field_650_cze'] = elements2['field_650']
                            matched[elements['field_650']]['cze_a'] = elements2['subfield_a']
                            matched[elements['field_650']]['en_wiki_label']=elements2['en_label']
                            found_match = True  # Match found, set flag to True
                            break
                    if fin_elem == "N/A" or fin_elem == '0' or fin_elem == 'brak':
                        continue
                    else:
                        if fin_elem in matched[elements['field_650']]['wiki_id']:
                            counter+=1
                            dowykluczenia_cze[elements2['field_650']]='znalezione'
                            #print('first ', fin_elem)
                            matched[elements['field_650']]['field_650_cze'] = elements2['field_650']
                            #matched[elements['field_650']]['en_wiki_label']=elements2['en_label']
                            #print('first ',matched)
                            matched[elements['field_650']]['cze_a'] = elements2['subfield_a']
                            if type(elements2['LOC_id'])!= float:
                                if elements2['LOC_id'] not in matched[elements['field_650']]['loc_id']:
                                    matched[elements['field_650']]['loc_id'].append(elements2['LOC_id'])
                            found_match = True  # Match found, set flag to True
                            break
                        
            if key=='LOC_id':
                if type(val)==float:
                    continue
                
                #print(sets)
                Loclist=val.split(',')
                for loc in Loclist:
                    if loc=="N/A" or loc=='0'or loc=='brak':
                        continue
                    if loc.startswith('http://www.wikidata.or'):
                        print (loc)
                    if loc.startswith('http://id.loc.gov'):
                        
                        loc=loc.split(r'/')[-1]
                        if loc in matched[elements['field_650']]['loc_id']:
                            print(loc)
                            matched[elements['field_650']]['field_650_cze'] = elements2['field_650']
                            matched[elements['field_650']]['cze_a'] = elements2['subfield_a']
                            dowykluczenia_cze[elements2['field_650']]='znalezione'
                            found_match = True  # Match found, set flag to True
                            break
                    else:
                        if loc in matched[elements['field_650']]['loc_id']:
                            print(loc)
                            matched[elements['field_650']]['field_650_cze'] = elements2['field_650']
                            matched[elements['field_650']]['cze_a'] = elements2['subfield_a']
                            dowykluczenia_cze[elements2['field_650']]='znalezione'
                            #matched[elements['field_650']]['en_wiki_label']=elements2['en_label']
                            if type(elements2['Wiki'])!= float:
                                if elements2['Wiki'] not in matched[elements['field_650']]['wiki_id']:
                                    matched[elements['field_650']]['wiki_id'].append(elements2['Wiki'])
                                    matched[elements['field_650']]['en_wiki_label']=elements2['en_label']
                            dowykluczenia_cze[elements2['field_650']]='znalezione'
                            found_match = True 
                            break
                        
        
                
        if found_match:
            #print(elements2)
            #cze_dictionary.remove(elements2)
            break  # Match found, exit the loop

    # Check the flag, if no match is found, set values to indicate no match
        if not found_match:
            matched[elements['field_650']]['field_650_cze'] = 'N/D'
            matched[elements['field_650']]['cze_a'] = 'N/D'
            #print(elements2['field_650'])
          #  matched[elements2['field_650']]={'loc_id':elements2['LOC_id'] , 'wiki_id':elements2['Wiki'],'cze_a':elements2['subfield_a'] }
            
        #     break  # Match found, exit the loop
           

counter=0  
    
for elem in cze_dictionary:

    
    if elem['field_650'] in  dowykluczenia_cze:
        continue
    else:
        
        matched[elem['field_650']]={'en_wiki_label': elem['en_label'],'loc_id':[] , 'wiki_id':[],'cze_a':elem['subfield_a'],'field_650_cze':elem['field_650'],'fin_a':'N/D','field_a_fin':'N/D' }
        
        
        
        for key, val in elem.items():
            
            if key=='Wiki':
                if type(val)==float:
                    
                    continue
                finlist=val.split(',')
                if len(finlist)>1:
                    
                    print(len(finlist))
                for fin_elem in finlist:
                    if fin_elem.startswith('http://id.loc.gov'):
                        fin_elem=fin_elem.split(r'/')[-1]
                        matched[elem['field_650']]['loc_id'].append(fin_elem)
                        
                        
                        continue
                    
                    if fin_elem=="N/A" or fin_elem=='0'or fin_elem=='brak':
                        continue
                    else:
                    
                        matched[elem['field_650']]['wiki_id'].append(fin_elem)
                        
            
            if key=='LOC_id':
                if type(val)==float:
                    continue
                
                #print(sets)
                Loclist=val.split(',')
                for loc in Loclist:
                    if loc=="N/A" or loc=='0'or loc=='brak':
                        continue
                    if loc.startswith('http://www.wikidata.or'):
                        pass
                    if loc.startswith('http'):
                        
                        loc=loc.split(r'/')[-1]
                        
                        matched[elem['field_650']]['loc_id'].append(loc)
                    
                    else:
                        
                        matched[elem['field_650']]['loc_id'].append(loc)
      
     

     
esp_do_wykluczenia={}
for key, value in matched.items():
    
    found_match=False

    
    wiki_id=value.get('wiki_id')
   # print(wiki_id)
    loc_id=value.get('loc_id') 
    ids=wiki_id+loc_id
   # print(ids)
    if ids:
        
        for esp_elem in esp_dictionary:
            
            
            wiki_esp=esp_elem.get('Wiki')
            if isinstance(wiki_esp, float):
                continue
            if wiki_esp:
               wiki_esp=wiki_esp.split(',')
               
               for wiki_esp_elem in wiki_esp:
                   print(wiki_esp_elem)
                   if wiki_esp_elem in ids:
                       
                       print(wiki_esp_elem)
                       matched[key]['field_650_esp'] = esp_elem['subfield_a']
                       matched[key]['en_wiki_label'] = esp_elem['en_wiki_label']
                       esp_do_wykluczenia[esp_elem['subfield_a']]='znalezione'
                       if esp_elem['LOC_id'] not in ids:
                           if type (esp_elem['LOC_id'])!=float:
                               matched[elements['field_650']]['loc_id'].append(esp_elem['LOC_id'])
                               
                               
                           
                       found_match=True
                       break

            loc_id_esp=esp_elem.get('LOC_id')  

            if isinstance(loc_id_esp, float):
                continue
            if loc_id_esp:
                loc_esp=loc_id_esp.split(',')
                for loc_esp_elem in loc_esp:
                    if loc_esp_elem.startswith('http'):
                        
                        loc_esp_elem=loc_esp_elem.split(r'/')[-1]
                    if loc_esp_elem in ids:
                        matched[key]['field_650_esp'] = esp_elem['subfield_a']
                        matched[key]['en_wiki_label'] = esp_elem['en_wiki_label']
                        
                        esp_do_wykluczenia[esp_elem['subfield_a']]='znalezione'
                        if esp_elem['Wiki'] not in ids:
                            if type (esp_elem['Wiki'])!=float:
                                matched[elements['field_650']]['wiki_id'].append(esp_elem['Wiki'])
                        found_match=True
                        break
                        
            
                
        
        
        
        
            if found_match:
               #print(elements2)
               #cze_dictionary.remove(elements2)
               break  # Match found, exit the loop            
    if not found_match:
        matched[key]['field_650_esp'] = 'N/D'
                
for elem in esp_dictionary:

    
    if elem['subfield_a'] in  esp_do_wykluczenia:
        continue
    else:
        
        matched[elem['subfield_a']]={'loc_id':[] ,'en_wiki_label': elem['en_wiki_label'], 'wiki_id':[],'cze_a':'N/D','field_650_cze':'N/D','fin_a':'N/D','field_a_fin':'N/D','field_650_esp':elem['subfield_a'] }
        
        
        
        
        for key, val in elem.items():
            
            if key=='Wiki':
                if type(val)==float:
                    
                    continue
                finlist=val.split(',')
                if len(finlist)>1:
                    
                    print(len(finlist))
                for fin_elem in finlist:
                    if fin_elem.startswith('http://id.loc.gov'):
                        fin_elem=fin_elem.split(r'/')[-1]
                        matched[elem['subfield_a']]['loc_id'].append(fin_elem)
                        
                        
                        continue
                    
                    if fin_elem=="N/A" or fin_elem=='0'or fin_elem=='brak':
                        continue
                    else:
                    
                        matched[elem['subfield_a']]['wiki_id'].append(fin_elem)
                        
            
            if key=='LOC_id':
                if type(val)==float:
                    continue
                
                #print(sets)
                Loclist=val.split(',')
                for loc in Loclist:
                    if loc=="N/A" or loc=='0'or loc=='brak':
                        continue
                    if loc.startswith('http://www.wikidata.or'):
                        pass
                    if loc.startswith('http'):
                        
                        loc=loc.split(r'/')[-1]
                        
                        matched[elem['subfield_a']]['loc_id'].append(loc)
                    
                    else:
                        
                        matched[elem['subfield_a']]['loc_id'].append(loc)
                        
from rdflib import Graph, plugin
from rdflib.serializer import Serializer
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
from rdflib import Dataset
from rdflib import URIRef
from rdflib import Literal
#new approach
from rdflib import Graph, Namespace

# Load the TTL data into an RDF graph
graph = Graph()
graph.parse("D:/Nowa_praca/lit_bn_skos_27_04_23.ttl", format='ttl')



# Define the SKOS namespace
skos = Namespace("http://www.w3.org/2004/02/skos/core#")

# Find all concepts with their close matches and preferred labels
concepts = graph.subjects(predicate=skos.prefLabel, object=None)
labels_close={}
count=0
for concept in concepts:
    concept_str=str(concept)
    labels_close[concept_str]={'Preferred Label:':'',"Close Match:":[] }
    count+=1

    print(concept)
    preferred_label = graph.value(subject=concept, predicate=skos.prefLabel)
    if preferred_label:
        print("Preferred Label:", preferred_label)
        labels_close[concept_str]['Preferred Label:']=str(preferred_label)

        close_matches = graph.objects(subject=concept, predicate=skos.closeMatch)

        for match in close_matches:
            labels_close[concept_str]['Close Match:'].append(str(match))  
dowywaleniapl={}
for key_val in matched:
    flag=False
    # print(matched[key_val]['wiki_id'])  
    # print(matched[key_val]['loc_id']) 
    ids=matched[key_val]['loc_id']+matched[key_val]['wiki_id']
    for labels in labels_close:
        closematches=labels_close.get(labels).get('Close Match:')
        for match in closematches:
            if match.startswith('http:/www.wikidata.org'):
                match=match.replace('http:/www.wikidata.org','http://www.wikidata.org')
                if match in ids:
                    dowywaleniapl[labels]='jest'
                    print(closematches)
                    matched[key_val]['pl_concept']=labels
                    flag=True
                    
                    break
            if match.startswith('http:/id.loc.gov'):
                match=match.split(r'/')[-1]
                if match in ids:
                    matched[key_val]['pl_concept']=labels
                    dowywaleniapl[labels]='jest'
                    flag=True
                    print(match)
                    break
               #print(match)
                
        if flag:
            break
    if not flag:
        matched[key_val]['pl_concept']='N/D'
            
    
    
                    


for labels in labels_close:
    if labels in dowywaleniapl:
        print(labels)
    else:
        matched[labels]={'pl_concept':labels,'loc_id':[] ,'en_wiki_label': 'N/D', 'wiki_id':[],'cze_a':'N/D','field_650_cze':'N/D','fin_a':'N/D','field_a_fin':'N/D','field_650_esp':'N/D' }
        closematches=labels_close.get(labels).get('Close Match:')
        for match in closematches:
            
            if match.startswith('http:/www.wikidata.org'):
                match=match.replace('http:/www.wikidata.org','http://www.wikidata.org')

                matched[labels]['wiki_id'].append(match)
                    
                    
                 
            if match.startswith('http:/id.loc.gov'):
                match=match.split(r'/')[-1]
                
                matched[labels]['loc_id'].append(match)
fin_df=pd.DataFrame.from_dict(matched, orient='index')
fin_df.to_excel("fi_cze_sp_pl6.xlsx")  
#matching all to wiki one table
all_data=pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/fi_cze_sp_pl6.xlsx', sheet_name='Sheet1',dtype=str).fillna('')
all_dictionary=all_data.to_dict('records')
all_dictionary=[e for e in all_dictionary if literal_eval(e['wiki_id'])]


for keyval in all_dictionary:
    keyval['wiki_id']=literal_eval(keyval['wiki_id'])
    keyval['loc_id']=literal_eval(keyval['loc_id'])
    keyval['cze_a']=[keyval['cze_a']]
    keyval['cze_id']=keyval['cze_id'].split(',')
    keyval['en_wiki_label']=[keyval['en_wiki_label']]
    keyval['esp_id']=keyval['esp_id'].split(',')
    keyval['fi_id']=keyval['fi_id'].split(',')
    del keyval['field650']
    del keyval['field_650_cze']
    keyval['field_a_esp']=[keyval['field_a_esp']]
    del keyval['field_a_fin']
    keyval['fin_a']=[keyval['fin_a']]
    keyval['pl_concept']=[keyval['pl_concept']]
    
    for k, v in keyval.items():
        if v==['']:
            keyval[k]=[]
    
    
    
    
    
for idx, elem in tqdm(enumerate(all_dictionary)):
    if elem:
        for idx2, e in enumerate(all_dictionary[idx+1:]):
            if e:
                matched=False
                for wiki_id in elem['wiki_id']:
                    if wiki_id in e['wiki_id']:
                        matched=True
                        break
                if matched:
                    for key, val in elem.items():
                        elem[key].extend(e[key])
                    all_dictionary[idx+1+idx2] = 0

all_data=[]
for x in all_dictionary:
    if x==0:
        continue
    else:
        all_data.append(x)
       
for x in all_data:
    for k, v in x.items():
        unique(x[k])        
fin_df=pd.DataFrame(all_data)
fin_df.to_excel("27062023_matched_fi_cze_sp_pl_ver_1.xlsx")         
a = ['b', 'a', 'b', 'a', 'c']
unique(a)
for elem in a:
    if elem=='b' or elem=='c' or elem=='a':
        a.remove(elem)
    
        
            
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                        