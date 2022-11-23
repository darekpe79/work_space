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
#czesi tylko 080
classifications={'080': 'Universal Decimal Classification Number',
'082': 'Dewey Decimal Classification Number',
'083': 'Additional Dewey Decimal Classification Number',
'084': 'Other Classification Number'}
    
ukd_og = pd.read_excel ("C:/Users/dariu/ukd_og.xlsx", sheet_name='Arkusz1')
dict_ = dict(zip(ukd_og['num'].to_list(),ukd_og['class'].to_list()))
removed_value = dict_.pop(5)
pliki=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/arto_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/fennica_2022-09-02.mrk"]
pattern_a_marc=r'(?<=\$a).*?(?=\$|$)' 
output=[]
output2={}
for ks in pliki:
    ksiazki1=list_of_dict_from_file(ks)
    
    for rekord in tqdm(ksiazki1):
        
            
        
        for key, value in rekord.items():
            if key=='080':
            
                for v in value:
                    if type(v) is list:
                        print(v)

                        
         
                    field_a=re.findall(pattern_a_marc, v)
                    print(field_a)
                    
                    if field_a:
                        for num, classi in dict_.items():
                            #classif=''
                            print(field_a)
                            if field_a[0].startswith(str(num)):
                                classif=dict_[num]
                                if field_a[0] not in  output2:
                                    output2[field_a[0]]=[1,classif]
                                else:
                                    output2[field_a[0]][0]+=1
                            elif field_a[0].startswith(str('5')):
                                if field_a[0].startswith(str('51')):
                                    classif='MATHEMATICS'

                                    
                                else:
                                    classif='NATURAL SCIENCES'
                                if field_a[0] not in  output2:
                                    output2[field_a[0]]=[1,classif]
                                else:
                                    output2[field_a[0]][0]+=1
                                    
                                
                          
excel=pd.DataFrame.from_dict(output2,orient='index')

excel.to_excel('finclassification2.xlsx', sheet_name='classification')
#%% Szczegółowe UKD
ukd_og = pd.read_excel ("C:/Users/dariu/ukd.xlsx", sheet_name='Sheet_name_1')
dict_ = dict(zip(ukd_og['num'].to_list(),ukd_og['class'].to_list()))
removed_value = dict_.pop(5)
pliki=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/arto_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/fennica_2022-09-02.mrk"]
pattern_a_marc=r'(?<=\$a).*?(?=\$|$)' 
output=[]
output2={}
for ks in pliki:
    ksiazki1=list_of_dict_from_file(ks)
    
    for rekord in tqdm(ksiazki1):
        
            
        
        for key, value in rekord.items():
            if key=='080':
            
                for v in value:
                    
                       # print(v)

                        
         
                    field_a=re.findall(pattern_a_marc, v)
                    #print(field_a)
                    
                    if field_a:
                        for num, classi in dict_.items():
                            #print(num.strip('\''))
                            #classif=''
                            #print(field_a)
                            if field_a[0]==num.strip('\''):
                                classif=dict_[num]
                                if field_a[0] not in  output2:
                                    output2[field_a[0]]=[1,classif]
                                else:
                                    output2[field_a[0]][0]+=1
        
                                    
                                
                          
excel=pd.DataFrame.from_dict(output2,orient='index')

excel.to_excel('finclassification_detailed_literature.xlsx', sheet_name='classification')




















       
            if key=='080':# or key=='082' or key=='083' or key=='084':
            #if key=='610' or key=='611' or key=='647' or key=='648' or key=='651'or key=='600':
                
                for v in value:
                    if type(v) is list:
                        
                        print(rekord)
                        
                    field_a=re.findall(pattern_a_marc, v)
                    if field_a:
                        if field_a[0].startswith('8'):
                            if field_a[0] not in  output2:
                                output2[field_a[0]]=value
                            else:
                                output2[field_a[0]].append(value)
                    
                if key not in  output2:
                    
                    output2[key]=value
                else:
                    output2[key].append(value)
                
                if key not in output:
                    output[key]=1
                else:
                    output[key]+=1
excel=pd.DataFrame.from_dict(output2,orient='index')
ex=excel.T
df1 = pd.DataFrame(ex[0:1048570])
excel.to_excel('fennicaclassification.xlsx', sheet_name='classification')
        
       rekord.get('610')
       rekord.get('611')
       rekord.get('647', '648', '651')
       

to_file2('chiny.mrk', output)






    field650=rekord.get('650')
    field651=rekord.get('651')
    field655=rekord.get('655')