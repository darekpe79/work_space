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

my_marc_files2 =["C:/Users/dariu/libri_marc_bn_articles_2023-3-13.mrc",
"C:/Users/dariu/libri_marc_bn_books_2023-03-14.mrc"]

zviaf={}
bezviaf={}
field_260_264=set()

#count=0
for my_marc_file in tqdm(my_marc_files2):
    #writer = TextWriter(open(my_marc_file.replace('.mrc','.mrk'),'wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):
            #count+=1
            #print(count)
           
            #print(record)
            try:
                
                my_200s = record.get_fields('001')
                #print(my_200s)
                for my in my_200s:
                    field_260_264.add(my.value())
                    
                    #print(my.subfields)
                    
                    #print(my)
                    #print(my.get_subfields('a', 'd', '1'))
                    # subfields_b= my.get_subfields('a')
                    # for b in subfields_b:
                    #     field_260_264.add(b)
                    
            
            except:
                pass
            
field_260_264_all=set()
count=0
for my_marc_file in tqdm(my_marc_files):
    #writer = TextWriter(open(my_marc_file.replace('.mrc','.mrk'),'wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):
            count+=1
            #print(count)
           
            #print(record)
            try:
                
                my_200s = record.get_fields('001')
               # print(my_200s)
                for my in my_200s:
                #    print(my.subfields)
                    field_260_264_all.add(my.value())
                    #print(my)
                    #print(my.get_subfields('a', 'd', '1'))
                    # subfields_b= my.get_subfields('a')
                    # for b in subfields_b:
                    #     field_260_264_all.add(b)
                    
            
            except:
                pass            
            
            
a = {1,2,3}
b = {3,4,5}
differences = field_260_264.difference(field_260_264_all)
#'Papierówna-Chudoń, Zofia.' in field_260_264
field_260_264_all_list = list(field_260_264_all)
field_260_264_list=list(field_260_264)
differences_list=list(differences)
df_field_260_264_all_list = pd.DataFrame(field_260_264_all_list)
df_field_260_264_list = pd.DataFrame(field_260_264_list)
df_differences_list=pd.DataFrame(differences_list)


writer = pd.ExcelWriter("dane_110_610_710.xlsx", engine = 'xlsxwriter')
df_field_260_264_all_list.to_excel(writer, sheet_name = 'all')
df_field_260_264_list.to_excel(writer, sheet_name = 'new')
df_differences_list.to_excel(writer, sheet_name = 'differences')
writer.close()        
            

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