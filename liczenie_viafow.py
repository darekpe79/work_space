import json
from tqdm import tqdm
import requests
from definicje import *
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
from pprint import pprint
import pprint
from time import time


#%%
pole100_viaF = pd.read_excel (r"F:\Nowa_praca\libri\Iteracja 2021-07\05.04.2022_ujednolicone_czasopisma_ludzie\caly_PBL.xlsx", sheet_name=1)
viaf_lista=pole100_viaF['viaf'].tolist()


pole100_lista=pole100_viaF['imie2'].tolist()
ujedolicone_lista=pole100_viaF['ujednolicone'].tolist()
dict_pole100_viaf = dict(zip(pole100_lista,viaf_lista))
viaf_imie = dict(zip(viaf_lista,ujedolicone_lista))
#proba
files=["F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/BN_articles.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/BN_books.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/BN_chapters.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles0.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles1.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles2.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles3.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_articles4.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_books.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/cz_chapters.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/fennica.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/nowearto08.04.2022_cale.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/PBL_articles.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_14.04.2022/PBL_books.mrk"]
pattern4=r'(?<=\$7).*?(?=\$|$)'
cale=set()
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
                     if 'viaf.org' in value:
                         

                         id_viaf = re.findall(pattern5, value)
                         cale.add(id_viaf[0])
                         
                         