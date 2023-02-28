# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:51:00 2023

@author: dariu
"""
from pymarc import MARCReader
from tqdm import tqdm
import requests
import json
from pymarc import MARCReader
from pymarc import parse_json_to_array
from pymarc import TextWriter
from pymarc import XMLWriter
from pymarc import JSONWriter
from io import BytesIO
import warnings
from pymarc import MARCReader
from pymarc import Record, Field 
import pandas as pd
from copy import deepcopy
from definicje import *

my_marc_files = ["D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/arto_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/pbl_articles_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/bn_articles_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/bn_books_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/bn_chapters_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles0_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles1_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles2_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles3_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles4_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_books_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_chapters_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/fennica_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/pbl_books_21-02-2023.mrc"]



zviaf={}
bezviaf={}
for my_marc_file in tqdm(my_marc_files):
    #writer = TextWriter(open(my_marc_file.replace('.mrc','.mrk'),'wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):
            print(record)
            
        
            try:
                
                my_500s = record.get_fields('100','700','600')
                for my in my_500s:
                    #print(my.get_subfields('a', 'd', '1'))
                    x= my.get_subfields('1')
                    d=my.get_subfields('d')
                    if x and d:
                        
                        #print (my['a']+'   '+ my['1'])
                        if my['a']+'  '+my['d'] in zviaf:
                            zviaf[my['a']+'  '+my['d']][3]+=1
                        else:
                            zviaf[my['a']+'  '+my['d']]=[my['a'],my['1'],my['d'],1]
                    elif x:
                        
                        #print (my['a']+'   '+ my['1'])
                        if my['a'] in zviaf:
                            zviaf[my['a']][3]+=1
                        else:
                            zviaf[my['a']]=[my['a'],my['1'],'',1]
                    elif d:
                        if my['a']+'  '+my['d'] in bezviaf:
                            bezviaf[my['a']+'  '+my['d']][2]+=1
                        else:
                            bezviaf[my['a']+'  '+my['d']]=[my['a'],my['d'],1]
                        
                            
                    else:
                        if my['a'] in bezviaf:
                            bezviaf[my['a']][2]+=1
                        else:
                            bezviaf[my['a']]=[my['a'],'',1]
            except:
                continue
viaf_nazwa_df=pd.DataFrame.from_dict(zviaf, orient='index') 
bez_viaf_nazwa_df=pd.DataFrame.from_dict(bezviaf, orient='index')
viaf_nazwa_df.to_excel("wszystko_viaf_data)23.02.2023.xlsx", sheet_name='Sheet_name_1')
bez_viaf_nazwa_df.to_excel("wszystko_bez_VIAF_data23.02.2023.xlsx")