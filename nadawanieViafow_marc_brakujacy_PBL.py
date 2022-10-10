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
pole100_viaF = pd.read_excel (r"F:\Nowa_praca\libri\Iteracja 2021-07\05.04.2022_ujednolicone_czasopisma_ludzie\caly_PBL.xlsx", sheet_name=1)
viaf_lista=pole100_viaF['viaf'].tolist()


pole100_lista=pole100_viaF['imie2'].tolist()
ujedolicone_lista=pole100_viaF['ujednolicone'].tolist()
dict_pole100_viaf = dict(zip(pole100_lista,viaf_lista))
viaf_imie = dict(zip(viaf_lista,ujedolicone_lista))
#proba
files=["F:/Nowa_praca/libri/Iteracja 2021-07/05.04.2022_ujednolicone_czasopisma_ludzie/PBL_books.mrk",
"F:/Nowa_praca/libri/Iteracja 2021-07/05.04.2022_ujednolicone_czasopisma_ludzie/PBL_articles.mrk"]
pattern4=r'(?<=\$7).*?(?=\$|$)'

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
                                 
                                 
                                 if podpole_d_marc and data_ujednolicona:
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
                                    value=value.replace(podpole_a_marc[0],nazwa_i_data_ujednolicona.strip(', .'))
                                    value_split=value.split('$')
                                    del value_split[2]
                                    value='$'.join(value_split)
                                 else:
                                    
                                    value=value.replace(podpole_a_marc[0],nazwa_i_data_ujednolicona.strip(', .'))
                     else:
                         id_viaf = re.findall(pattern5, value)
                     
                         if id_viaf[0] in viaf_imie:
                             podpole_a_marc=re.findall(pattern_a_marc, value)
                             podpole_d_marc=re.findall(pattern_daty_marc, value)
                             #print(podpole_d_marc)
                             #print(podpole_a_marc)
                             #print(rekord)
                             switch=True
                             #print(value)
         
                             nazwa_i_data_ujednolicona=viaf_imie[id_viaf[0]]
                             #print(nazwa_i_data_ujednolicona)
                             data_ujednolicona=re.search(pattern_daty,nazwa_i_data_ujednolicona)
                             
                             
                             if podpole_d_marc and data_ujednolicona:
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
                                value=value.replace(podpole_a_marc[0],nazwa_i_data_ujednolicona.strip(', .'))
                                value_split=value.split('$')
                                del value_split[2]
                                value='$'.join(value_split)
                             else:
                                
                                value=value.replace(podpole_a_marc[0],nazwa_i_data_ujednolicona.strip(', .'))
                         
                             
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
