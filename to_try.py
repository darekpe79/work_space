# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:47:54 2023

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
# Find matches
matched = {}

for elements in fin_dictionary:
    wiki_values = elements.get('Wiki', '')
    loc_values = elements.get('LOC_id', '')

    if isinstance(wiki_values, str) and isinstance(loc_values, str):
        wiki_matches = set(wiki_values.split(',')) & cze_wiki_ids
        loc_matches = set(loc_values.split(',')) & cze_loc_ids

        if wiki_matches or loc_matches:
            matched[elements['field_650']] = {
                'wiki_id': list(wiki_matches),
                'loc_id': list(loc_matches),
                'fin_a': elements['subfield_a']
            }

            # Search for matches in cze_dictionary
            for cze_elem in cze_dictionary:
                cze_wiki_values = cze_elem.get('Wiki', '')
                cze_loc_values = cze_elem.get('LOC_id', '')

                if (
                    isinstance(cze_wiki_values, str)
                    and isinstance(cze_loc_values, str)
                    and set(wiki_values.split(',')) <= set(cze_wiki_values.split(','))
                    and set(loc_values.split(',')) <= set(cze_loc_values.split(','))
                ):
                    matched[elements['field_650']]['field_650_cze'] = cze_elem['field_650']
                    matched[elements['field_650']]['cze_a'] = cze_elem['subfield_a']
                    #matched[elements['field_650']]['cze_other_value'] = cze_elem['other_value']
                    # Add more values from cze_elem as needed
matched = {}
unmatched = {}
counter = 0

for elements in tqdm(fin_dictionary):
    found_match = False  # Flag to track if a match is found

    matched[elements['field_650']] = {'wiki_id': [], 'loc_id': [], 'fin_a': elements['subfield_a']}

    for key, val in elements.items():
        if key == 'Wiki':
            if type(val) == float:
                continue
            finlist = val.split(',')
            for fin_elem in finlist:
                if fin_elem.startswith('http://id.loc.gov'):
                    fin_elem = fin_elem.split(r'/')[-1]
                    matched[elements['field_650']]['loc_id'].append(fin_elem)
                    continue
                if fin_elem == "N/A" or fin_elem == '0' or fin_elem == 'brak':
                    continue
                else:
                    matched[elements['field_650']]['wiki_id'].append(fin_elem)
        
        if key == 'LOC_id':
            if type(val) == float:
                continue
            Loclist = val.split(',')
            for loc in Loclist:
                if loc == "N/A" or loc == '0' or loc == 'brak':
                    continue
                if loc.startswith('http://www.wikidata.or'):
                    pass
                if loc.startswith('http'):
                    loc = loc.split(r'/')[-1]
                    matched[elements['field_650']]['loc_id'].append(loc)
                else:
                    matched[elements['field_650']]['loc_id'].append(loc)

    for elements2 in cze_dictionary:
        for key, val in elements2.items():
            if key == 'Wiki':
                if type(val) == float:
                    continue
                finlist = val.split(',')
                for fin_elem in finlist:
                    if fin_elem.startswith('http://id.loc.gov'):
                        fin_elem = fin_elem.split(r'/')[-1]
                        if fin_elem in matched[elements['field_650']]['loc_id']:
                            matched[elements['field_650']]['field_650_cze'] = elements2['field_650']
                            matched[elements['field_650']]['cze_a'] = elements2['subfield_a']
                            found_match = True
        
        if found_match:
            break  # Match found, exit the loop

    # Check the flag, if no match is found, set values to indicate no match
    if not found_match:
        matched[elements['field_650']]['field_650_cze'] = 'N/D'
        matched[elements['field_650']]['cze_a'] = 'N/D'

# Find unmatched elements
for elements in fin_dictionary:
    if elements['field_650'] not in matched:
        unmatched[elements['field_650']] = {
            'wiki_id': [],
            'loc_id': [],
            'fin_a': elements['subfield_a'],
            'field_650_cze': 'N/D',
            'cze_a': 'N/D'
        }

