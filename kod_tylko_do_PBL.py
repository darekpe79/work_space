# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:27:17 2022

@author: darek
"""

#%%     PBL

from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
#from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
from time import time
from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()
from concurrent.futures import ThreadPoolExecutor
import threading
import xlsxwriter
                       
probny={'liryka':['\\\\$aLyrical poetry$leng','\\\\$aLiryka$lpol','\\\\$aLyrická poezie$lcze','\\\\$aRunot$lfin'],
        'epika':['\\\\$aFiction$leng','\\\\$aEpika$lpol', '\\\\$aProza$lcze','\\\\$aKertomakirjallisuus$lfin'],
        'dramat':['\\\\$aDrama$leng','\\\\$aDramat$lpol', '\\\\$aDrama$lcze','\\\\$aNäytelmät$lfin'],
        'inne':['\\\\$aOther$leng','\\\\$aInne$lpol','\\\\$aJiný$lcze','\\\\$aMuu$lfin'],
        'secondary':['\\\\$aSecondary literature$leng','\\\\$aLiteratura przedmiotu$lpol','\\\\$aSekundární literatura$lcze','\\\\$aToissijainen kirjallisuus$lfin']
        }


genre = pd.read_excel (r"C:\Users\darek\pbl655_380_zrobione.xlsx")

col655=genre[655].tolist()
nowe655=genre['zrobione'].tolist()
nowe380=genre[380].tolist()
    
    
nowe3801=[]
for n in nowe380:
    
    nowe3801.append(n.strip('|'))
    
    
genredict={}
for index,stare in enumerate(col655):
    genredict[stare]=[]
    genredict[stare].append(nowe655[index])
    genredict[stare].append(nowe3801[index])
    
    


        
        



nieobrobione=[]
output=[]
rekordy=[]
for plik in [r"F:\Nowa_praca\marki 29.06.2022\PBL_articles.mrk"]:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    
    
    for rekord in tqdm(dictrec):
        new_value_list=[]
        new655=[]

        
        
        nowe={}
        for key, val in rekord.items():
            nowe[key]=val
            

            
            if key=='655':
                

                v=val.split('❦')
                
                for value in v:
                    if len(value)>=1:
                        

                        
                        
                        if value in genredict:
                            print(v)
                            v.remove(value)
                            nowe['655']=v
                            field_380=genredict[value][1].split('|')
                            
                            for f in field_380:
                                new_value_list.extend(probny[f])
                            field_655=genredict[value][0].split('|')
                            for f655 in field_655:
                                new655.append('\\4$a'+f655)
                                #\4$a
                        
      

                            
        if new_value_list:
            if '380' in nowe:

                list_set=unique(new_value_list+[nowe['380']])
                 
                nowe['380']='❦'.join(list_set)
            else:
                list_set = unique(new_value_list)
                                                  
                nowe['380']='❦'.join(list_set)
            
            if nowe['655']:    
                nowe['655']='❦'.join(nowe['655']+new655)
            else:
                nowe['655']='❦'.join(new655)
            

            
                
        rekordy.append(nowe)   

to_file ('PBL_articles.mrk', rekordy)  