# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:14:58 2022

@author: darek
"""

from definicje import *
from tqdm import tqdm
plik=r"F:\Nowa_praca\libri\Iteracja 2021-07\21.12.2021\pbl_marc_books_2021-8-4.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)

plik2=r"F:\Nowa_praca\pliki_Nikodem_30.03.2022\wetransfer_libri_iteracja7_30-03-2022_2022-03-30_0720\libri_30-03-2022_iter7_final\PBL_books.mrk"
lista2=mark_to_list(plik2)
dictrec2=list_of_dict_from_list_of_lists(lista2)


id1=id_of_rec(dictrec)
id2=id_of_rec(dictrec2)
roznica2=[record for record in tqdm (id1) if record not in id2]
newdict=[]
for dic in dictrec:
    
    
    if dic['001'] in roznica2:
        newdict.append(dic)
to_file('brakujace_PBL_books.mrk',newdict )
        


