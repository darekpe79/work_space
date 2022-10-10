# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:15:57 2022

@author: dariu
"""

from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
#from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
import xlsxwriter
import datetime

major_gen=[r'\\$iMajor genre$aLyrical poetry$leng',
r'\\$iMajor genre$aFiction$leng',r'\\$iMajor genre$aOther$leng',r'\\$iMajor genre$aDrama$leng']
literature_str=r'\\$iMajor genre$aLiterature$leng'
secondary_str=r'\\$iMajor genre$aSecondary literature$leng'
def list_of_dict_from_list_of_lists2 (records):
    recs2table = []
    for record in records:
        rec_dict = {}
        for field in record:
            if field[1:4] in rec_dict.keys():
                rec_dict[field[1:4]].append(field[6:].strip())
            else:
                rec_dict[field[1:4]] = [field[6:].strip()]
        recs2table.append(rec_dict)
    return recs2table

def to_file2 (file_name, list_of_dict_records):
    ''' list of dict records to file mrk'''
    file1 = open(file_name, "w", encoding='utf-8') 
    for record in list_of_dict_records:
        
        dictionary=sortdict(record)
        dictionary2=compose_data(dictionary)
        
        for key, value in dictionary2.items():
            for field in value:
                line='='+key+'  '+field+'\n'
                file1.writelines(line)
        file1.writelines('\n')
    
    file1.close()

pattern3=r'(?<=\$a).*?(?=\$|$)' 



for plik in ["D:/Nowa_praca/marki_29_07_2022/arto.mrk",
"D:/Nowa_praca/marki_29_07_2022/fennica.mrk"]:
    path=plik.split('/')[-1].split('.')[-2]
    
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists2(lista)
    #to_file2('new.mrk',dictrec)
    rekordy=[]
    for rekord in tqdm(dictrec):
        
        

        

        nowe={}

        for key, val in rekord.items():
            nowe[key]=val
            
            if key=='381':
                
                
                    
                    
                    #print(val_381)
                    
                if secondary_str in val:
                    print(val)
                    if len(val)>1:
                        val.remove(secondary_str)
                    else:
                        del nowe['381']
                    if '380' in nowe:
                        nowe['380'].append(secondary_str)
                    else:
                        nowe['380']=[secondary_str]
                        
                for el in major_gen:
                    if el in val:
                        if '380' in nowe:
                            nowe['380'].append(literature_str)
                        else:
                            nowe['380']=[literature_str]
                        
                            

                    
            
            
            
            
           
        
        
            
        rekordy.append(nowe)   
    date_object = datetime.date.today()        
    to_file2 (path+'_'+str(date_object)+'.mrk', rekordy)  