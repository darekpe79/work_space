# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:42:51 2022

@author: dariu
"""

import json
from tqdm import tqdm
import requests
from definicje import *
import pandas as pd
import os
from tqdm import tqdm
#from googletrans import Translator
from itertools import zip_longest
import regex as re
from pprint import pprint
import pprint
from time import time

def checker(field_list, check_list):
    if field_list:
        for elem in check_list:
            if any([elem in field for field in field_list]):
                return True
    else:
        return False
    


pliki=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/arto_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/fennica_2022-09-02.mrk"]
output={}
for ks in pliki:
    ksiazki1=list_of_dict_from_file(ks)
    
    for rekord in tqdm(ksiazki1):
        for key, value in rekord.items():
            if key=='610' or key=='611' or key=='647' or key=='648' or key=='651'or key=='600':
                print(key, value)
                if key not in output:
                    output[key]=1
                else:
                    output[key]+=1
excel=pd.DataFrame.from_dict(output, orient='index')

excel.to_excel('finstats6xx.xlsx', sheet_name='6xxstats')
        
       rekord.get('610')
       rekord.get('611')
       rekord.get('647', '648', '651')
       

to_file2('chiny.mrk', output)






    field650=rekord.get('650')
    field651=rekord.get('651')
    field655=rekord.get('655')