
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


pliki=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/fennica_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/arto_2022-09-02.mrk"]
    
def centuryFromYear(year):
    return (int(year) + 99) // 100
centuryFromYear(1200)
pattern_a_marc=r'(?<=\$a).*?(?=\$|$)' 
years=r'\d{4}|\d{3}|\d{2}|\d{1}'
'př. Kr.'
'století'
string='1. století př. Kr.'

x=''.join([l for l in string if l.isalpha()])
output={}
for ks in pliki:
    ksiazki1=list_of_dict_from_file(ks)
    
    for rekord in tqdm(ksiazki1):
        
            
        
        for key, value in rekord.items():
            if key=='648':
            
                for v in value:
                     
         
                    field_a=re.findall(pattern_a_marc, v)
                    
                    if field_a:
                        stringi=''.join([l for l in field_a[0] if l.isalpha()])
                        if stringi in ['pne','přKr']:
                            
                        
                            rok=re.findall(years, field_a[0])
                            
                            if rok:
                                if len(rok)>1:
                                    first=centuryFromYear(rok[0])
                                    second=centuryFromYear(rok[1])
                                    if first==second:
                                        output[field_a[0]]=[str(first)+' p.n.e.']
                                    else:
                                        output[field_a[0]]=[str(first)+'-'+str(second)+' p.n.e.']
                                else:
                                    year=centuryFromYear(rok[0])
                                    output[field_a[0]]=[str(year)+' p.n.e.']
                        elif stringi in ['stoletípřKr']:
                            
                        
                            rok=re.findall(years, field_a[0])
                            
                            if rok:
                                if len(rok)>1:
                                    first=rok[0]
                                    second=rok[1]
                                    if first==second:
                                        output[field_a[0]]=[str(first)+' p.n.e.']
                                    else:
                                        output[field_a[0]]=[str(first)+'-'+str(second)+' p.n.e.']
                                else:
                                    year=rok[0]
                                    output[field_a[0]]=[str(year)+' p.n.e.']
                        elif stringi in ['století']:
                            
                        
                            rok=re.findall(years, field_a[0])
                            
                            if rok:
                                if len(rok)>1:
                                    first=rok[0]
                                    second=rok[1]
                                    if first==second:
                                        output[field_a[0]]=[str(first)]
                                    else:
                                        output[field_a[0]]=[str(first)+'-'+str(second)]
                                else:
                                    year=rok[0]
                                    output[field_a[0]]=[str(year)]
                                    
                        elif stringi=='':
                            
                        
                            rok=re.findall(years, field_a[0])
                            
                            if rok:
                                if len(rok)>1:
                                    first=centuryFromYear(rok[0])
                                    second=centuryFromYear(rok[1])
                                    if first==second:
                                        output[field_a[0]]=[str(first)]
                                    else:
                                        output[field_a[0]]=[str(first)+'-'+str(second)]
                                else:
                                    year=centuryFromYear(rok[0])
                                    output[field_a[0]]=[str(year)]

                            
excel=pd.DataFrame.from_dict(output,orient='index')

excel.to_excel('pol_czech_centuries.xlsx', sheet_name='classification') 
def centuryFromYear(year):
    return (int(year) + 100) // 100
centuryFromYear(1200)
pattern_a_marc=r'(?<=\$a).*?(?=\$|$)'
years=r'\d{4}|\d{3}|\d{2}|\d{1}'  
output={}
for ks in pliki:
    ksiazki1=list_of_dict_from_file(ks)
    
    for rekord in tqdm(ksiazki1):
        
            
        
        for key, value in rekord.items():
            if key=='648':
            
                for v in value:
                     
         
                    field_a=re.findall(pattern_a_marc, v)
                    
 
                    
                
                    rok=re.findall(years, field_a[0])
                    
                    if rok:
                        if len(rok)>1:
                            first=centuryFromYear(rok[0])
                            second=centuryFromYear(rok[1])
                            if first==second:
                                output[field_a[0]]=[str(first)]
                            else:
                                output[field_a[0]]=[str(first)+'-'+str(second)]
                        else:
                            year=centuryFromYear(rok[0])
                            output[field_a[0]]=[str(year)]                    
                        
excel=pd.DataFrame.from_dict(output,orient='index')

excel.to_excel('fin_centuries.xlsx', sheet_name='classification') 