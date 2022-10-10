from definicje import *
import pandas as pd
import os
from tqdm import tqdm
plik=r'/home/darek/Nowa_praca /fennica/arto_odsiane_24082021/rekordyarto1odsiane.mrk'
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)
pola={}
for rekord in dictrec:
    for key, val in rekord.items():
        if key not in pola:
            pola[key]=0
        pola[key]+=1




#%%
list=['lala','lala','reks']
count={}



for x in list:
    if x not in count:
        count[x]=0
    count[x]+=1
    
    
    
    if x in count.keys():
        count[x]+=1
        
    else:
        count[x]=1