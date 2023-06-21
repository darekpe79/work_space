# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:22:05 2023

@author: dariu
"""

import pandas as pd
import re
from tqdm import tqdm
from definicje import *
# DANE OD STRONY SLOWNIKOW plik wiki_sparql- od strony wikidata
#finowie
fin650=pd.read_excel('C:/Users/dariu/07062023finowie_650_zebrane_do_maczowania.xlsx', sheet_name='Sheet1',dtype=str)
cze650=pd.read_excel('C:/Users/dariu/14062023czesi__650_zebrane_do_maczowania.xlsx', sheet_name='Sheet1',dtype=str)
esp650=pd.read_excel('C:/Users/dariu/14062023hiszpanie_650_zebrane_do_maczowania.xlsx', sheet_name='Sheet1',dtype=str)
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
                            #break
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
                            #break
                        
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
    loc_id=value.get('loc_id') 
    ids=wiki_id+loc_id
    if ids:
        
        for esp_elem in esp_dictionary:
            
            wiki_esp=esp_elem.get('Wiki')
            if isinstance(wiki_esp, float):
                continue
            if wiki_esp:
               
               for wiki_esp_elem in wiki_esp:
                   if wiki_esp_elem in ids:
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
                        
                        
fin_df=pd.DataFrame.from_dict(matched, orient='index')
fin_df.to_excel("fi_cze_sp3.xlsx")   