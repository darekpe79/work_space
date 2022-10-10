# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:38:17 2022

@author: darek
"""

from definicje import *
import pandas as pd
import os
from tqdm import tqdm
import re 
plik=r"F:\Nowa_praca\fennica\Najnowsze_dobreVIAF_ISSNY\my_new_marcFENNICA_ALL0_good_VIAF"
lista=mark_to_list(plik+'.mrk')
dictrec=list_of_dict_from_list_of_lists(lista)
monography=LDR_monography(dictrec)
article=LDR_article(dictrec)


counter=0
for filename in tqdm(os.listdir(r"F:\Nowa_praca\fennica\Najnowsze_dobreVIAF_ISSNY")):
    if filename.endswith(".mrk"):
        
        path=os.path.join(r"F:\Nowa_praca\fennica\Najnowsze_dobreVIAF_ISSNY", filename)
        #print(path)
        file=filename.split('.')
        file=file[0]+str(counter)+'przerobione'
        counter+=1
        #print(file)
        lista=mark_to_list(path)
        monography=list_of_dict_from_list_of_lists(lista)
                               
#%% zabawa liczbami
# Sprawdzam wszystkie ósemki:
    records8084=[]
    for record in tqdm(monography):
        if '080' in record.keys() and '084' in record.keys():
            if any([field[4:].startswith('8') for field in record['080'].split('❦')]) or any([field[4:].startswith('8') for field in record['084'].split('❦')]):
                records8084.append(record)
                
        elif '080' in record.keys():
            if any([field[4:].startswith('8') for field in record['080'].split('❦')]):
                
                records8084.append(record)
    
        elif '084' in record.keys():
            if any([field[4:].startswith('8') for field in record['084'].split('❦')]):
                records8084.append(record)
                
    # sprawdzam YKL 
    ykl=[]
    yklpole80=[]
    for record in records8084:
        if '084'in record.keys() and '080' in record.keys():
            
            if 'ykl' in record['084'] or 'ykl' in record['080']:
                ykl.append(record)
        elif '080' in record.keys():
            if 'ykl' in record['080']:
                ykl.append(record)
                yklpole80.append(record)
        elif '084' in record.keys():
            if 'ykl' in record['084']:
                ykl.append(record)
                
    # szukam dobrych YKL (w polu 080 nie ma YKL)
              
    dobreYKL=[]
    ykl87_89=[]
    for yklrecord in ykl:
        if '084' in yklrecord.keys():
            if any([field[4:].startswith(("80", "81", "82", "83", "84", "85", "86")) for field in yklrecord['084'].split('❦')]):
                dobreYKL.append(yklrecord)
            else:
                ykl87_89.append(yklrecord)
    # z rekordów 8084 wykluczam te ze zlymi YKLami:
    tylko_dobreYKL=[record for record in records8084 if record not in ykl87_89]
    
    #%%Sprawdzam czy mam jakies 810,  (zeby je wykluczyc przy wyszukiwaniu 81, jesli brak moge od razu wyszukac 81, jesli brak jade dalej. Wszystko to w rekordach bez YKL (tam 81 jest dobre i porządane)
    rekordy_bez_YKL=[e for e in tqdm(tylko_dobreYKL) if e not in ykl ] 
    dobre810=[]
    
    for otherrecord in rekordy_bez_YKL:
        if '080' in otherrecord.keys() and '084' in otherrecord.keys():
            if any([field[4:].startswith('810') for field in otherrecord['080'].split('❦')]) or any([field[4:].startswith('81') for field in otherrecord['084'].split('❦')]):
                dobre810.append(otherrecord)
                
        elif '080' in otherrecord.keys():
            if any([field[4:].startswith('810') for field in otherrecord['080'].split('❦')]):
                
                dobre810.append(otherrecord)
    
        elif '084' in otherrecord.keys():
            if any([field[4:].startswith('810') for field in otherrecord['084'].split('❦')]):
                dobre810.append(otherrecord)
    #jesli są 810 wywalam je (z listy bez YKL- zeby nie usunac dobrych 810) i przeszukuje pozostale pod katem 81
    rekordy_bez_810= [e for e in rekordy_bez_YKL if e not in dobre810]
    zle81=[]
    
    for otherrecord in rekordy_bez_810:
        if '080' in otherrecord.keys() and '084' in otherrecord.keys():
            if any([field[4:].startswith('81') for field in otherrecord['080'].split('❦')]) or any([field[4:].startswith('81') for field in otherrecord['084'].split('❦')]):
                zle81.append(otherrecord)
                
        elif '080' in otherrecord.keys():
            if any([field[4:].startswith('81') for field in otherrecord['080'].split('❦')]):
                
                zle81.append(otherrecord)
    
        elif '084' in otherrecord.keys():
            if any([field[4:].startswith('81') for field in otherrecord['084'].split('❦')]):
                zle81.append(otherrecord)
    
    ostateczna_lista_po_numerkach=[e for e in tylko_dobreYKL if e not in zle81 ]
    
    #%% zabawa z hasłami po id
    pattern2='(?<=\/yso\/).*?(?=\$|$)'
    hasla=pd.read_excel(r"C:\Users\darek\yso_87_literature_ids.xlsx")
    listahasel=hasla['id'].to_list()
    record_dobry_po_hasle=[]
    
    for record in tqdm(monography):
        
        if '650' in record.keys():
               
            value=record['650'].split("❦")
            
            for v in value:
                #print(v)
                try:
                    id_rec = re.findall(pattern2, v)[0]
                    #print('id', id_rec)
    
                    if id_rec in listahasel:
                        
                        record_dobry_po_hasle.append(record)
                        break 
                            
                except:
                    pass
                    
                #record_dobry_po_hasle.append(record)
    # wykluczam czy te po numerkach sa w tych po hasle
    unique_rekord_dobry_po_hasle=[e for e in tqdm(record_dobry_po_hasle) if e not in ostateczna_lista_po_numerkach]
    ostateczna_lista=unique_rekord_dobry_po_hasle+ostateczna_lista_po_numerkach
    
    to_file(file+'.mrk' ,ostateczna_lista)
#%% Odrzucanie po typie 336
    pattern_336='(?<=\$a).*?(?=\$|$)'    
    typ_336=pd.read_excel(r"C:\Users\darek\336_stats.xlsx")
    
    lista_336=typ_336['wybrane do wywalenia'].dropna().to_list()
    lista_336_2=[]
    for l in lista_336:
        print(l)
        l=l.strip().lower()
        lista_336_2.append(l)
        
        
    
    
    record_dobry_336=[]
    
    for record in tqdm(ostateczna_lista):
        
        if '336' in record.keys():
               
            value=record['336'].split("❦")
            
            for v in value:
                #print(v)
                try:
                    typ = re.findall(pattern_336, v)[0]
                    #print('id', id_rec)
    
                    if typ.strip().lower() not in lista_336_2:
                        
                        record_dobry_336.append(record)
                        break 
                            
                except:
                    pass
    to_file(file+'336.mrk' ,record_dobry_336)
    