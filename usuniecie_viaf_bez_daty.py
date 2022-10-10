# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:35:07 2022

@author: darek
"""

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

#proba

patternViaf='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'    


path=r"F:\Nowa_praca\libri\Iteracja 2021-07\oviafowane02.02.2022BN_PBL\pbl_marc_books2021-8-4_good995_ALL_good_VIAF_700.mrk"
path2=path.split('\\')


lista=mark_to_list(path)
dictrec=list_of_dict_from_list_of_lists(lista)
switch=False
tylkoViaf=[]
#proba=[]
counter=0
cale=[]
for rekord in tqdm(dictrec):
    
    for key, val in rekord.items():
        if key=='700':
            
             val2=val.split('❦')
             listavalue=[]
             for value in val2:
                 if '$d' not in value and 'viaf.org' in value:
                     #print(value)
                     #viaf = re.findall(patternViaf, value)
                     #print(viaf)
                     #value=value.replace('$1http://viaf.org/viaf/'+viaf[0],'')
                     value=re.sub('\$1http.+', '', value)


                         #proba.append(rekord)
                         
                 listavalue.append(value)
                         
                         #print(poleviaf)
             rekord[key]='❦'.join(listavalue)
                
                     
    cale.append(rekord)

      
to_file(path2[-1]+'_korekta-usuniety_viaf_bezdat700.mrk', cale)
#to_file(path2[-1]+'2TylkoViaf.mrk', tylkoViaf) 





