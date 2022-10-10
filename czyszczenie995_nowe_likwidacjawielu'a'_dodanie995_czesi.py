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


from itertools import zip_longest
import regex as re
import requests


#%% Proba na jednym pliku
plik=r"F:\Nowa_praca\pliki_nikodem_4.04.2022\wetransfer_czech_part1_2022-04-04_1128\cz_articles3.mrk"
path2=plik.split('\\')
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)

        
            
        


pattern=r'(?<=\\\$).*?(?=$)'
string="\\\$aČeská Literární Bibliografie"
## inne podescie ten sam efekt: 
rekordy=[]
for listy in tqdm(dictrec):
    listy.setdefault('995', '\\\$aČeská Literární Bibliografie')
    
to_file(path2[-1],dictrec)
