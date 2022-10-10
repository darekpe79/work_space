# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:48:03 2021

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
translator = Translator()

#%% Proba na jednym pliku
plik=r"F:\Nowa_praca\fennica\odsiane_z_viaf_100_700_600fennica\msplit00000010odsiane10zViaf10zViaf100_700_600.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)

        
        

strings=json.dumps(dictrec)
string_replaced=strings.replace("$0(VIAF)", "$1http://viaf.org/viaf/")

mydict=json.loads(string_replaced)        
to_file('my_new_marcFENNICA_ISSN_ALL10_good_VIAF.mrk',mydict)

## inne podescie ten sam efekt: 
rekordy=[]
for listy in dictrec:
    nowe={}
    for k,v in listy.items():
        new=v.replace("$0(VIAF)", "$1http://viaf.org/viaf/")
        
        nowe[k]=new
    rekordy.append(nowe)
   
