# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:10:22 2022

@author: dariu
"""
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm

from itertools import zip_longest
import regex as re
import requests
from time import time
#from alphabet_detector import AlphabetDetector
#ad = AlphabetDetector()
from concurrent.futures import ThreadPoolExecutor
import threading
paths=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/arto_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/fennica_2022-09-02.mrk"]
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$b).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
zViaf={}
bezviaf={}
viafs_list=[]

allnames={}
for plik in paths:
    dictrec=list_of_dict_from_file(plik)
    

    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key=='260' or key=='264':
    
                
                for v in val:
                    
                    #val100.append(vi)

                    
                    #date = re.findall(pattern4, v)
                    name = re.findall(pattern4, v)
                    place = re.findall(pattern3, v)

                    #viaf=re.findall(pattern5, v)

                    
                    if name:
                        name=name[0]
                    else:
                        name='brak'
                    


                    if place:
                        place=place[0]
                    else:
                        place='brak'

                    
                    if name+' '+place not in allnames:
                            allnames[name+' '+place]=[1,name,place]
                    else:
                        allnames[name+' '+place][0]+=1
                        
viaf_nazwa_df=pd.DataFrame.from_dict(allnames, orient='index') 
viaf_nazwa_df.to_excel("fin_publishers.xlsx", engine='xlsxwriter')