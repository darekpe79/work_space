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


#%%

#proba
files=["C:/Users/dariu/Desktop/praca/marki_29_07_2022/pbl_books.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/pbl_articles.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/fennica.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/cz_books.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/cz_chapters.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/cz_articles4.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/cz_articles3.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/cz_articles2.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/cz_articles1.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/cz_articles0.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/bn_chapters.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/bn_books.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/bn_articles.mrk",
"C:/Users/dariu/Desktop/praca/marki_29_07_2022/arto.mrk"]
pattern4=r'(?<=\$7).*?(?=\$|$)'
cale={}
bez_viaf={}
#bez_viaf=set()
#wszyscy=set()
for plik in files:

    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    


##dla jednego pliku

    path2=plik.split('/')
    pattern4=r'(?<=\$7).*?(?=\$|$)'
    pattern_daty=r'\(?[\(?\d\? ]{2,5}[-–.](\(?[\d\?]{3,5}\)?| \))?'
    pattern_daty_marc=r'(?<=\$d).*?(?=\$|$)'
    pattern_a_marc=r'(?<=\$a).*?(?=\$|$)'
    pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'

    switch=False
    tylkoViaf=[]
    #proba=[]
    counter=0
    
    for rekord in tqdm(dictrec):
        
        for key, val in rekord.items():
            if key=='700' or key=='600' or key=='100':
                
                 val2=val.split('❦')

                 for value in val2:

                     name= re.findall(pattern_a_marc, value)
                     id_viaf = re.findall(pattern5, value)
                     date=re.findall(pattern_daty_marc, value)
                         
                     if 'viaf.org' in value:
                         

                         if name and id_viaf and date:
                             cale[id_viaf[0]]=[]
                             cale[id_viaf[0]].append(name[0])
                             cale[id_viaf[0]].append(date[0])
                         elif name and id_viaf:
                            cale[id_viaf[0]]=[]
                            cale[id_viaf[0]].append(name[0])

                            
                            
                     else:
                         
                         if name and date:
                             if name[0] not in bez_viaf: 
                                 
                                 bez_viaf[name[0]]=[]
                                 bez_viaf[name[0]].append(date[0])
                                 bez_viaf[name[0]].append(1)
                             else:
                                 bez_viaf[name[0]][1]+=1
                                
                                 
                                 
                                 

                         elif name:
                             if name[0] not in bez_viaf: 
                                 
                                 bez_viaf[name[0]]=[]
                                 bez_viaf[name[0]].append('brak_daty')
                                 bez_viaf[name[0]].append(1)
                             else:
                                 bez_viaf[name[0]][1]+=1
                             
                             
                             
excel=pd.DataFrame.from_dict(bez_viaf, orient='index')
excel.to_excel('wszystko_Bez_viaf_data.xlsx', sheet_name='bezviaf')   
excel2= pd.DataFrame(list(bez_viaf))
excel2.to_excel('wszystko_bez_viaf.xlsx', sheet_name='655') 
słownik={}
słownik['proba']=[]
słownik['proba'].append('lala')
słownik['proba'].append('lala2')
słownik['proba'].append(1)
słownik['proba'][2]+=1
excel=pd.DataFrame.from_dict(słownik, orient='index')                        