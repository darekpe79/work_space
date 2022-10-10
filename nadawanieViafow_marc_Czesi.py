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
pole100_viaF = pd.read_excel (r"C:\Users\darek\ujednolicanko_Czesi_calosc.xlsx")

pole100_lista=pole100_viaF['id_czech'].tolist()
#proba

    

viaf_lista=pole100_viaF['viaf'].tolist()
dict_pole100_viaf = dict(zip(pole100_lista,viaf_lista))
##dla jednego pliku
path=r"F:\Nowa_praca\czesi_viaf.mrk"
path2=path.split('\\')
pattern4=r'(?<=\$7).*?(?=\$|$)'
lista=mark_to_list(path)
dictrec=list_of_dict_from_list_of_lists(lista)
switch=False
tylkoViaf=[]
#proba=[]
counter=0
cale=[]
for rekord in dictrec:
    
    for key, val in rekord.items():
        if key=='700' or key=='600' or key=='100':
            
             val2=val.split('❦')
             listavalue=[]
             for value in val2:
                 if '$7' in value:
                     id_cz = re.findall(pattern4, value)
                     if id_cz[0] in dict_pole100_viaf:
                     
                         #print(rekord)
                         switch=True
                         #print(value)
     
                         viafy=dict_pole100_viaf[id_cz[0]]
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
