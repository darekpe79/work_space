# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:56:09 2022

@author: darek
"""

from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests

dataframe_viafy_ujednolicone=pd.read_excel(r"F:\Nowa_praca\fennica\10.03.2022.fennica_arto_ujednolicanie-po_ujednolicaniuMARK_EXCEL_Ujednolicone_osoby\ujednolicanko_arto_fennica.xlsx", sheet_name=0) 
pole100_lista=dataframe_viafy_ujednolicone['viaf'].tolist()
ujednolicona_nazwa=dataframe_viafy_ujednolicone['ujednolicone'].tolist()
dict_pole100_viaf = dict(zip(pole100_lista,ujednolicona_nazwa))
path=r"F:\Nowa_praca\06.05.2022,Wszyscy ujednolicone czasopisma i ludzieMARC_EXCEL\fennica.mrk"
lista=mark_to_list(path)
dictrec=list_of_dict_from_list_of_lists(lista)
path2=path.split('\\')
#pattern_daty_stary=r'\(?[\d\?.]{2,5}-(\(?[\d\?]{3,5}\)?| \))?'
pattern_daty=r'\(?[\(?\d\? ]{2,5}[-–.](\(?[\d\?]{3,5}\)?| \))?'
pattern_daty_marc=r'(?<=\$d).*?(?=\$|$)'
pattern_a_marc=r'(?<=\$a).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'


switch=False
tylkoViaf=[]
#proba=[]
counter=0
cale=[]
for rekord in tqdm(dictrec):
    
    for key, val in rekord.items():
        if key=='700' or key=='600' or key=='100':
            
             val2=val.split('❦')
             listavalue=[]
             for value in val2:
                 if 'viaf.org' in value:
                     id_viaf = re.findall(pattern5, value)
                 
                     if id_viaf[0] in dict_pole100_viaf:
                         
                         podpole_a_marc=re.findall(pattern_a_marc, value)
                         podpole_d_marc=re.findall(pattern_daty_marc, value)
                         #print(podpole_d_marc)
                         #print(podpole_a_marc)
                         #print(rekord)
                         switch=True
                         #print(value)
     
                         nazwa_i_data_ujednolicona=dict_pole100_viaf[id_viaf[0]]
                         #print(nazwa_i_data_ujednolicona)
                         data_ujednolicona=re.search(pattern_daty,nazwa_i_data_ujednolicona)
                         
                         
                         if podpole_d_marc and data_ujednolicona and podpole_a_marc:
                            
                            data_ujednolicona_strip=(data_ujednolicona.group()).strip('() ')
                            a_ujednolicone=nazwa_i_data_ujednolicona.replace(data_ujednolicona.group(),'')
                            value=value.replace(podpole_a_marc[0],a_ujednolicone.strip(', .'))
                            if data_ujednolicona_strip.endswith(('-','–',' -','.')):
                                data_ujednolicona_strip=data_ujednolicona_strip.strip(' -–.')
                                value=value.replace(podpole_d_marc[0],'('+data_ujednolicona_strip+'- )')
                            else: 
                                value=value.replace(podpole_d_marc[0],'('+data_ujednolicona_strip+')')   
                         #print(value)
                         elif data_ujednolicona and not podpole_d_marc:
                            data_ujednolicona_strip=(data_ujednolicona.group()).strip('() ')
                            a_ujednolicone=nazwa_i_data_ujednolicona.replace(data_ujednolicona.group(),'')
                            value=value.replace(podpole_a_marc[0],a_ujednolicone.strip(', .'))
                            value_split=value.split('$')
                            print(value_split)
                            if data_ujednolicona_strip.endswith(('-','–',' -','.')):
                                data_ujednolicona_strip=data_ujednolicona_strip.strip(' -–.')
                            
                                value=value_split[:2]+['d'+'('+data_ujednolicona_strip+'- )']+value_split[2:]
                                
                            
                            else:
                                value=value_split[:2]+['d'+'('+data_ujednolicona_strip+')']+value_split[2:]
                                
                            value='$'.join(value)
                         elif podpole_d_marc and not data_ujednolicona:
                            value=value.replace(podpole_a_marc[0],nazwa_i_data_ujednolicona.strip(', .'))
                            value_split=value.split('$')
                            del value_split[2]
                            value='$'.join(value_split)
                         else:
                            
                            value=value.replace(podpole_a_marc[0],nazwa_i_data_ujednolicona.strip(', .'))
                            


                         podpole_d_marc=re.findall(pattern_daty_marc, value)
                         podpole_a_marc=re.findall(pattern_a_marc, value)
                         if podpole_d_marc and podpole_a_marc:

                             split_order=value.split('$')
                             podpole_a_marc='a'+re.findall(pattern_a_marc, value)[0]
                             podpole_d_marc='d'+re.findall(pattern_daty_marc, value)[0]
                             index_a = split_order.index(podpole_a_marc)
                             index_d=split_order.index(podpole_d_marc)
                             if index_a>index_d:
                                 split_order=split_order[:index_a+1]+[podpole_d_marc]+split_order[index_a+1:]
                                 #print(split_order)
                                 del split_order[index_d]
                                 
                                 
                             value='$'.join(split_order)
                                 
                                 
                             
                             
    
                                    
                            

                            

                            

                         #value=value+'$1http://viaf.org/viaf/'+str(viafy)
                         ###proba.append(rekord)
                         
                 listavalue.append(value)
                         
                         #print(poleviaf)
             rekord[key]='❦'.join(listavalue)
                
                     
    cale.append(rekord)
    if switch==True:
        tylkoViaf.append(rekord)
        switch=False
to_file(path2[-1], cale)
to_file(r'ujedntylk.mrk', tylkoViaf) 
from ftfy import fix_text
x=fix_text("Cervantes Saavedra, Miguel de 1547-1616")
y=fix_text(x)


