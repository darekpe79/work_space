from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
'''
dataframe_viafy=pd.read_excel(r"C:\Users\darek\ujednolicanie\PO_UjednolicoankoWSzystko_BN_zapisane_na_Chapters_BN.xlsx") 
listaViafs=dataframe_viafy['viafspr'].tolist()
lista_names=dataframe_viafy['name'].tolist()
imie_viaf = dict(zip(listaViafs, lista_names))       
df_wszyscy_autorzy = pd.read_excel(r"C:\Users\darek\ujednolicanie\Do_UJEDNLOICANKO_dopracy_dobre_WSZYSTKO_CO_MA_VIAF_BN.xlsx")
df_wszyscy_autorzy['ujednolicona_nazwa'] = df_wszyscy_autorzy['viaf'].map(imie_viaf)
df_wszyscy_autorzy.to_excel(r"C:\Users\darek\ujednolicanie\ujednolicony_BN.xlsx", sheet_name='Sheet_name_1') 
'''



dataframe_viafy_ujednolicone=pd.read_excel(r"C:\Users\darek\ujednolicanie\ujednolicony_BN.xlsx") 






pole100_lista=pole100_viaF['100'].tolist()
#proba

  

viaf_lista=pole100_viaF['viaf'].tolist()
dict_pole100_viaf = dict(zip(pole100_lista,viaf_lista))
##dla jednego pliku
path=r"F:\\Nowa_praca\do_podmianki100600proba.mrk"
path2=path.split('\\')

lista=mark_to_list(path)
dictrec=list_of_dict_from_list_of_lists(lista)
switch=False
tylkoViaf=[]
#proba=[]
counter=0
cale=[]
for rekord in dictrec:
    
    for key, val in rekord.items():
        if key=='700':
            
             val2=val.split('❦')
             listavalue=[]
             for value in val2:
                 if 'viaf.org' not in value:
                 #print(value)
                     if value in dict_pole100_viaf:
                         #print(rekord)
                         switch=True
                         #print(value)
     
                         viafy=dict_pole100_viaf[value]
                         value=value+'$1http://viaf.org/viaf/'+str(viafy)
                         #proba.append(rekord)
                         
                 listavalue.append(value)
                         
                         #print(poleviaf)
             rekord[key]='❦'.join(listavalue)
                
                     
    cale.append(rekord)
    if switch==True:
        tylkoViaf.append(rekord)
        switch=False
to_file(path2[-1], cale)
to_file(path2[-1]+'2TylkoViaf.mrk', tylkoViaf)
