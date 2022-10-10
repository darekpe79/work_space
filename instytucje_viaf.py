import gspread as gs
from gspread_dataframe import get_as_dataframe

import pandas as pd
import requests
from tqdm import tqdm
import json
import simplejson
from difflib import get_close_matches, SequenceMatcher
import unicodedata

#%%
sheet_key = '19eY6Pnk58QgkaTrBrpHTSlrbYHyi8vZTSpIvxG9fGLk'

gc = gs.oauth()
sheet = gc.open_by_key(sheet_key)
worksheet = sheet.get_worksheet(0)
df = get_as_dataframe(worksheet, evaluate_formulas=True).fillna('').astype(str)

institutions_dict = {}

for index, row in df.iterrows():
    institutions_dict[row['ID_INSTYTUCJI']] = {'name': row['NAZWA_INSTYTUCJI_UJEDNOLICONA']}

#%%
new_institutions_dict = {}

for key, value in tqdm(institutions_dict.items()):
    # key = '319'
    # value = institutions_dict[key]
    new_institutions_dict[key] = value
    url = 'https://www.viaf.org/viaf/search?query=local.corporateNames%20all%20%22{}%22&sortKeys=holdingscount&maximumRecords=10&startRecord=1&httpAccept=application/json'.format(value['name'].replace("\"", "").replace("'", ""))
    
    try:
        response = requests.get(url)
        new_institutions_dict[key]['resp_rec_num'] = int(response.json()['searchRetrieveResponse']['numberOfRecords'])
        if int(response.json()['searchRetrieveResponse']['numberOfRecords']) > 0:
            if isinstance(response.json()['searchRetrieveResponse']['records'], list):
                new_institutions_dict[key]['records'] = []
                for elem in response.json()['searchRetrieveResponse']['records']:
                    viaf = elem['record']['recordData']['viafID']
                    headings = elem['record']['recordData']['mainHeadings']['data']
                    new_institutions_dict[key]['records'].append({'viaf': viaf, 'headings': headings})
            else:
                viaf = response.json()['searchRetrieveResponse']['records']['record']['recordData']['viafID']
                headings = response.json()['searchRetrieveResponse']['records']['record']['recordData']['mainHeadings']['data']
                new_institutions_dict[key]['records'].append({'viaf': viaf, 'headings': headings})
        else: new_institutions_dict[key]['records'] = []
    except (ConnectionError, TimeoutError): 
        new_institutions_dict[key]['records'] = 'error'
        new_institutions_dict[key]['resp_rec_num'] = 'error'
    
    # if int(response.json()['searchRetrieveResponse']['numberOfRecords']) > 10:
    #     for i in range(11, int(response.json()['searchRetrieveResponse']['numberOfRecords']), 10):
    #         print(i)
    
    
with open('institution_dict.json', 'w', encoding='utf-8') as jfile:
    json.dump(new_institutions_dict, jfile, ensure_ascii=False, indent=4)
    
#%%  
with open('institution_dict.json', 'r', encoding='utf-8') as jfile:
    new_institutions_dict = json.load(jfile)

matches_dict = {}
others_list = []

for key, value in tqdm(new_institutions_dict.items()):
    # key = '1'
    # value = new_institutions_dict[key]
    
    if value['resp_rec_num'] != 0 and value['resp_rec_num'] != 'error':
        matches = {}     
        for elem in value['records']:
            flags = 0
            viaf = 'https://www.viaf.org/viaf/' + elem['viaf']
            wiki = ''
            if isinstance(elem['headings'], list):
                headings = [e['text'] for e in elem['headings']]
                if not wiki:
                    for head in elem['headings']:
                        if isinstance(head['sources']['sid'], list):
                            for e in head['sources']['sid']:
                                if e.startswith('WKP'):
                                    wiki = e
                                
                        else:
                            if head['sources']['sid'].startswith('WKP'):
                                wiki = head['sources']['sid']
                # flags
                for e in elem['headings']:
                    if isinstance(e['sources']['s'], list):
                        flags += len(e['sources']['s'])
                    else: flags += 1  
                    
            else: 
                headings = [elem['headings']['text']]
                if not wiki:
                    if isinstance(elem['headings']['sources']['sid'], list):
                        for e in elem['headings']['sources']['sid']:
                            if e.startswith('WKP'):
                                wiki = e
                            
                    else:
                        if elem['headings']['sources']['sid'].startswith('WKP'):
                            wiki = elem['headings']['sources']['sid']
                # flags
                if isinstance(elem['headings']['sources']['s'], list):
                    flags += len(elem['headings']['sources']['s'])
                else: flags += 1      
            
            best_heading = get_close_matches(value['name'], headings, cutoff=0.0, n=1)
            
            # https://www.wikidata.org/wiki/Q11712953
            # wiki = [e for e in elem['headings']['sources']['sid'] if e.startswith('WKP') else '']
            if best_heading:
                if wiki: wiki = 'https://www.wikidata.org/wiki/' + wiki[4:]
                matches[viaf] = best_heading + [SequenceMatcher(None, value['name'], best_heading[0]).ratio(), wiki, flags]
        
        if matches:
            matches_dict[key] = {'id': key,
                              'name': value['name'],
                              'matches': matches}
        else: others_list.append([key, value['name']])
        
    else: others_list.append([key, value['name']])
    
    
to_df = []
for key, value in matches_dict.items():
    for viaf, match in value['matches'].items():
        to_df.append([key, value['name'], viaf, match[0], match[1], match[2], match[3]])

df2 = pd.DataFrame(to_df, columns=['id', 'pbl_institution', 'viaf', 'name', 'ratio', 'wikidata', 'flags'])
df_others = df2[df2['ratio'] < 0.5]
df2 =  df2[df2['ratio'] >= 0.5]

df_others = df_others[['id', 'pbl_institution']].append(pd.DataFrame(others_list, columns=['id', 'pbl_institution'])).drop_duplicates()
df_others = df_others[~df_others['id'].isin(df2['id'])]

df2.to_excel('institutions_viaf_ratio.xlsx', index=False)
df_others.to_excel('institutions_others.xlsx', index=False)

#%%

sheet_key = '19eY6Pnk58QgkaTrBrpHTSlrbYHyi8vZTSpIvxG9fGLk'

gc = gs.oauth()
sheet = gc.open_by_key(sheet_key)
worksheet = sheet.get_worksheet(4)
df = get_as_dataframe(worksheet, evaluate_formulas=True).fillna('').astype(str)
viaf_list = list(set(df['viaf_url']))

output_dict = {}
for viaf in tqdm(viaf_list):
    viaf = viaf.strip('/')
    url = viaf + '/viaf.json'
    try:
        response = requests.get(url)
        if 'x500s' in response.json():
            if isinstance(response.json()['mainHeadings']['data'], list):
                main_name = response.json()['mainHeadings']['data'][0]['text']
            else: main_name = response.json()['mainHeadings']['data']['text']
            
            if isinstance(response.json()['x500s']['x500'], list):
                connected_viafs = []
                for elem in response.json()['x500s']['x500']:
                    if '@viafLink' in elem:
                        identifier = elem['@viafLink']
                    else: identifier = 'brak'
                    
                    if 'subfield' in elem['datafield']:
                        if isinstance(elem['datafield']['subfield'], list):
                            name = [e['#text'] for e in elem['datafield']['subfield'] if e['@code'] == 'a']
                            if name: name = name[0]
                            else: name = 'brak' 
                        else: 
                            if elem['datafield']['subfield']['@code'] == 'a':
                                name = elem['datafield']['subfield']['#text']
                            else: name = 'brak'
                    else: name = 'brak'
                    temp = [identifier, name]
                    connected_viafs.append(temp)
            else: 
                if '@viafLink' in response.json()['x500s']['x500']:
                    identifier = response.json()['x500s']['x500']['@viafLink']
                else: identifier = 'brak'
                
                if 'subfield' in response.json()['x500s']['x500']['datafield']:
                    if isinstance(response.json()['x500s']['x500']['datafield']['subfield'], list):
                        name = [e['#text'] for e in response.json()['x500s']['x500']['datafield']['subfield'] if e['@code'] == 'a']
                        if name: name = name[0]
                        else: name = 'brak' 
                    else: 
                        if response.json()['x500s']['x500']['datafield']['subfield']['@code'] == 'a':
                            name = response.json()['x500s']['x500']['datafield']['subfield']['#text']
                        else: name = 'brak'
                else: name = 'brak'
                connected_viafs = [[identifier, name]]
            output_dict[viaf] = {'name': main_name, 'connected_viafs': connected_viafs}
        else: output_dict[viaf] = 'bez powiazanych'          
    except (ConnectionError, TimeoutError): 
        output_dict[viaf] = 'connection error'
    except simplejson.decoder.JSONDecodeError:
        output_dict[viaf] = 'json decode error'

with open('powiazane_instytucje.json', 'w', encoding='utf-8') as jfile:
    json.dump(output_dict, jfile, ensure_ascii=False, indent=4)

#%%

with open('powiazane_instytucje.json', 'r', encoding='utf-8') as jfile:
    powiazane_inst_dict = json.load(jfile)

to_df_list = []
for key, value in powiazane_inst_dict.items():
    if not isinstance(value, dict):
        to_df_list.append([value, key, '', '', ''])
    else:
        for elem in value['connected_viafs']:
            if elem[0] == 'brak':
                viaf = 'brak'
            else:
                viaf = 'https://www.viaf.org/viaf/' + elem[0]
            name = unicodedata.normalize( 'NFC', elem[1])
            to_df_list.append(['ok', key, value['name'], viaf, name])
            
df = pd.DataFrame(to_df_list, columns=['status', 'viaf', 'name', 'related_viaf', 'related_name'])
df = df.drop_duplicates()
df.to_excel('related_institutions.xlsx', index=False)

#%%

df = pd.read_excel('related_institutions.xlsx')
df = df[df['status'] == 'ok']
bez_viaf = list(set(df[df['related_viaf'] == 'brak']['related_name']))

viaf_output = {}
for name in tqdm(bez_viaf):
    viaf_output[name] = {}
    url = 'https://www.viaf.org/viaf/search?query=local.corporateNames%20all%20%22{}%22&sortKeys=holdingscount&maximumRecords=10&startRecord=1&httpAccept=application/json'.format(name.replace("\"", "").replace("'", ""))
    try:
        response = requests.get(url)
        if int(response.json()['searchRetrieveResponse']['numberOfRecords']) > 0:
            if isinstance(response.json()['searchRetrieveResponse']['records'], list):
                viaf_output[name]['records'] = []
                for elem in response.json()['searchRetrieveResponse']['records']:
                    viaf = elem['record']['recordData']['viafID']
                    headings = elem['record']['recordData']['mainHeadings']['data']
                    viaf_output[name]['records'].append({'viaf': viaf, 'headings': headings})
            else:
                viaf = response.json()['searchRetrieveResponse']['records']['record']['recordData']['viafID']
                headings = response.json()['searchRetrieveResponse']['records']['record']['recordData']['mainHeadings']['data']
                viaf_output[name]['records'].append({'viaf': viaf, 'headings': headings})
        else: viaf_output[name]['records'] = []
    except (ConnectionError, TimeoutError): 
        viaf_output[name]['records'] = 'error'
        viaf_output[name]['resp_rec_num'] = 'error'
    
with open('braki_viaf_resp.json', 'w', encoding='utf-8') as jfile:
    json.dump(viaf_output, jfile, ensure_ascii=False, indent=4)

#%%

with open('braki_viaf_resp.json', 'r', encoding='utf-8') as jfile:
    viaf_output = json.load(jfile)

output_dict = {}
for key, value in tqdm(viaf_output.items()):
    output_dict[key] = []
    key_alnum = ''.join([e for e in key if e.isalnum()]).lower()
    for rec in value['records']:
        rec_num_sources = 0
        viaf = rec['viaf']
        # rec_name_alnum = ''.join([e for e in rec['name'] if e.isalnum()]).lower()
        if isinstance(rec['headings'], dict):
            best_name = rec['headings']['text']
            if isinstance(rec['headings']['sources']['s'], list):
                rec_num_sources = len(rec['headings']['sources']['s'])     
            else:
                rec_num_sources = 1
        else: 
            best_name =''
            temp_similarity = 0
            for elem in rec['headings']:
                elem_alnum = ''.join([e for e in elem['text'] if e.isalnum()]).lower()
                elem_similarity = SequenceMatcher(None, key_alnum, elem_alnum).ratio()
                if elem_similarity > temp_similarity:
                    temp_similarity = elem_similarity
                    best_name = elem['text']
                if isinstance(elem['sources']['s'], list):
                    rec_num_sources += len(elem['sources']['s'])
                else:
                    rec_num_sources += 1
        best_name_alnum = ''.join([e for e in best_name if e.isalnum()]).lower()
        similarity = SequenceMatcher(None, key_alnum, best_name_alnum).ratio()

        output_dict[key].append([viaf, best_name, rec_num_sources, similarity])


for key, value in tqdm(output_dict.items()):        
    temp = []  
    for elem in value:
        if elem[3] > 0.85:
            temp.append(elem)
    output_dict[key] = temp


final = {}
for key, value in tqdm(output_dict.items()):     
    best = []
    temp_source = 0
    temp_sim = 0
    for elem in value:
        if elem[2] > temp_source:
            best = elem
            temp_source = elem[2]
            temp_sim = elem[3]
        elif elem[2] == temp_source:
            if elem[3] > temp_sim:
                best = elem
                temp_source = elem[2]
                temp_sim = elem[3]
    final[key] = best
        
for key, value in final.items():
    if value:
        value[0] = 'https://www.viaf.org/viaf/' + value[0]

for index, row in df.iterrows():
    if row['related_viaf'] == 'brak':
        if final[row['related_name']]:
            row['related_viaf'] = final[row['related_name']][0]

df = df.drop(columns=['related_name', 'status'])
df = df[df['related_viaf'] != 'brak']
df = df.drop_duplicates()

#%%
from concurrent.futures import ThreadPoolExecutor

def get_viaf_name(viaf_url):
    url = viaf_url + '/viaf.json'
    r = requests.get(url)
    try:
        if r.json().get('mainHeadings'):
            if isinstance(r.json()['mainHeadings']['data'], list):
                name = r.json()['mainHeadings']['data'][0]['text']
            else:
                name = r.json()['mainHeadings']['data']['text']
            viaf_names_resp[viaf_url] = name
        elif r.json().get('redirect'):
            new_viaf = r.json()['redirect']['directto']
            new_url = 'https://www.viaf.org/viaf/' + new_viaf
            viaf_names_resp[viaf_url] = new_url
            get_viaf_name(new_url)
    except KeyboardInterrupt as exc:
        raise exc
    except:
        raise print(url)

viaf_names_resp = {}
viaf_url_set = set(df['related_viaf'])

with ThreadPoolExecutor(max_workers=50) as excecutor:
    list(tqdm(excecutor.map(get_viaf_name, viaf_url_set)))

#%%
headers_list = ['viaf', 'name', 'related_viaf', 'related_name']
df = df.reindex(columns=headers_list).fillna('')

for index, row in df.iterrows():
    if viaf_names_resp[row['related_viaf']].startswith('http'):
        row['related_name'] = viaf_names_resp[viaf_names_resp[row['related_viaf']]]
    else:
        row['related_name'] = viaf_names_resp[row['related_viaf']]
        
df = df.drop_duplicates()
df.to_excel('institutions_final.xlsx', index=False)



