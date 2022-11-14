# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:47:38 2022

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
    


ksiazki=["C:/Users/dariu/Desktop/ksiazki_bn/msplit00000004.mrk",
"C:/Users/dariu/Desktop/ksiazki_bn/msplit00000000.mrk",
"C:/Users/dariu/Desktop/ksiazki_bn/msplit00000001.mrk",
"C:/Users/dariu/Desktop/ksiazki_bn/msplit00000002.mrk",
"C:/Users/dariu/Desktop/ksiazki_bn/msplit00000003.mrk"]
output=[]
for ks in ksiazki:
    ksiazki1=list_of_dict_from_file(ks)
    
    for rekord in tqdm(ksiazki1):
        
        if checker(rekord.get('655'), ['Literatura podróżnicza','Reportaż', 'Relacja z podróży']):
            if checker(rekord.get('650'), ['Chiny']) or checker(rekord.get('651'), ['Chiny']):
                output.append(rekord)
        

to_file2('chiny.mrk', output)






    field650=rekord.get('650')
    field651=rekord.get('651')
    field655=rekord.get('655')
    if field655:
        for searchdesc in ['Powieść']:
            if any ([searchdesc in field for field in field655]):
                if field650 or field651:
                if any    [x for x in part8 if type(x) is not float] 
            
ksiazki1=list_of_dict_from_file('C:/Users/dariu/Desktop/ksiazki_bn/chiny.mrk')
rekordy=[]
for rekord in tqdm(ksiazki1):
    # patterna=r'(?<=\$a).*?(?=\$|$)'
    # field008=rekord.get('008')
    # print(field008[0][35:38])

    # if field041:
    #     if field041[0].startswith('0'):
    #         print(field041)
    #         podpole_a_marc=re.findall(patterna, field041[0])
    #         if podpole_a_marc[0]=='pol':
                
                field260=rekord.get('260')
                patternc=r'(?<=\$c).*?(?=\$|$)'
                podpole_c_marc=re.findall(patternc, field260[0])
                if podpole_c_marc:
                    if podpole_c_marc[0].startswith('2'):
                        rekordy.append(rekord)
            
            
to_file2('chinyod2000.mrk', rekordy)          
        
    
     
    if checker(rekord.get('655'), ['Literatura podróżnicza','Reportaż', 'Relacja z podróży']):
         if checker(rekord.get('650'), ['Chiny']) or checker(rekord.get('651'), ['Chiny']):
             output.append(rekord)    
    
    
    
    for key, val in rekord.items():
        if key=='650' or key=='651:
            
            for v in val:
                if 'Chiny' in v:
                    