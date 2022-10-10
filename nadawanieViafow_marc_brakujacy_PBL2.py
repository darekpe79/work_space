import json
from tqdm import tqdm
import requests
from definicje import *
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
from pprint import pprint
import pprint
from time import time


#%%
pole100_viaF = pd.read_excel (r"C:\Users\darek\pierwszy1000_po_weryf_zVIAF.xlsx", sheet_name=0)





dict_pole100_viaf = dict(zip(pole100_viaF['propozycja'].tolist(),pole100_viaF['viaf'].tolist()))
viaf_imie = dict(zip(pole100_viaF['viaf'].tolist(),pole100_viaF['ujednolicone'].tolist()))

#proba
files=["F:/Nowa_praca/18.05.2022 najnowsze Marki/PBL_books.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/arto.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/BN_articles.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/BN_books.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/BN_chapters.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/cz_articles0.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/cz_articles1.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/cz_articles2.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/cz_articles3.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/cz_articles4.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/cz_books.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/fennica.mrk",
"F:/Nowa_praca/18.05.2022 najnowsze Marki/PBL_articles.mrk"]


for plik in files:

    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    


##dla jednego pliku

    path2=plik.split('/')
    pattern4=r'(?<=\$7).*?(?=\$|$)'
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
                     if 'viaf.org' not in value:
                         
                         podpole_a_marc=re.findall(pattern_a_marc, value)
                         if podpole_a_marc:
                             if podpole_a_marc[0].strip() in dict_pole100_viaf:
                             
                                 #print(rekord)
                                 switch=True
                                 #print(value)
             
                                 viafy=dict_pole100_viaf[podpole_a_marc[0].strip()]
                                 value=value+'$1http://viaf.org/viaf/'+str(viafy)
                                 #proba.append(rekord)
                                 #podpole_a_marc=re.findall(pattern_a_marc, value)
                                 podpole_d_marc=re.findall(pattern_daty_marc, value)
                                 #print(podpole_d_marc)
                                 #print(podpole_a_marc)
                                 #print(rekord)
                                 switch=True
                                 #print(value)
             
                                 nazwa_i_data_ujednolicona=viaf_imie[viafy]
                                 #print(nazwa_i_data_ujednolicona)
                                 data_ujednolicona=re.search(pattern_daty,nazwa_i_data_ujednolicona)
                                 
                                 
                                 if podpole_d_marc and not data_ujednolicona and not podpole_a_marc:
                                      podpole_d_marc='d'+re.findall(pattern_daty_marc, value)[0]
                                      #value=value.replace(podpole_a_marc[0],nazwa_i_data_ujednolicona.strip(', .'))
                                      value_split=value.split('$')
                                      #print(value)
                                      value_split.remove(podpole_d_marc)
                                      value_split.insert(-1,'a'+nazwa_i_data_ujednolicona.strip(', .') )
                                      value='$'.join(value_split)
                                   
                                   
                                 elif podpole_d_marc and data_ujednolicona:
                                      
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
                                      #print(value_split)
                                      if data_ujednolicona_strip.endswith(('-','–',' -','.')):
                                          data_ujednolicona_strip=data_ujednolicona_strip.strip(' -–.')
                                      
                                          value=value_split[:2]+['d'+'('+data_ujednolicona_strip+'- )']+value_split[2:]
                                          
                                      
                                      else:
                                          value=value_split[:2]+['d'+'('+data_ujednolicona_strip+')']+value_split[2:]
                                          
                                      value='$'.join(value)
                                 elif podpole_d_marc and not data_ujednolicona:
                                      podpole_d_marc='d'+re.findall(pattern_daty_marc, value)[0]
                                      value=value.replace(podpole_a_marc[0],nazwa_i_data_ujednolicona.strip(', .'))
                                      value_split=value.split('$')
                                      value_split.remove(podpole_d_marc)
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
         
                         
                             
                     listavalue.append(value)
                             
                             #print(poleviaf)
                 rekord[key]='❦'.join(listavalue)
                    
                         
        cale.append(rekord)
        if switch==True:
            tylkoViaf.append(rekord)
            switch=False
    to_file(path2[-1], cale)
    to_file(path2[-1]+'2TylkoViaf.mrk', tylkoViaf) 

##dla wielu plików
switch=False
tylkoViaf=[]
proba=[]
counter=0
for filename in tqdm(os.listdir(r'F:\Nowa_praca\fennica\odsiane_z_viaf_100_700_fennica')):
                           if filename.endswith(".mrk"):
                               
                               path=os.path.join(r"F:\Nowa_praca\fennica\odsiane_z_viaf_100_700_fennica", filename)
                               #print(path)
                               file=filename.split('.')
                               file=file[0]+str(counter)+'zViaf100_700_600.mrk'
                               counter+=1
                               #print(file)
                               lista=mark_to_list(path)
                               dictrec=list_of_dict_from_list_of_lists(lista)

                               cale=[]
                               
                               for rekord in dictrec:
                                   for key, val in rekord.items():
                                       if key=='100':
                                           
                                            val2=val.split('❦')
                                            listavalue=[]
                                            for value in val2:
                                                if value in dict_pole100_viaf:
                                                    switch=True
                                                    #print(value)
 
                                                    viafy=dict_pole100_viaf[value]
                                                    value=value+'$0(VIAF)'+viafy
                                                    proba.append(rekord)
                                                listavalue.append(value)
                                                
                                                    #print(poleviaf)
                                            rekord[key]='❦'.join(listavalue)
                                            #tylkoViaf.append(rekord)
                                   cale.append(rekord)
                                   if switch==True:
                                       tylkoViaf.append(rekord)
                                       switch=False

    
                                    
                                      
                               to_file(file, cale)                              
to_file('tylko_z_Viaf_afennica600_nazwisko_data_i_pojedyncze_najnowsze.mrk', tylkoViaf)
