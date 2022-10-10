# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:50:42 2022

@author: darek
"""

from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests

dataframe_viafy_ujednolicone=pd.read_excel(r"C:\Users\darek\matchowanieVIAF13.05.2022.xlsx", sheet_name=0)
pole100_lista=dataframe_viafy_ujednolicone.nazwa.tolist()

dict_items = dict(zip(dataframe_viafy_ujednolicone.nazwa.tolist(),dataframe_viafy_ujednolicone.propozycja.tolist()))
nazwa=[]
propozycja=[]
for key, values in dict_items.items():
    for value in values.split('|'):
        print(key,'propozycja:', value)
        nazwa.append(key)
        propozycja.append(value)
df = pd.DataFrame(list(zip(nazwa, propozycja)), columns =['Nazwa', 'propozycja'])
viafs=pd.read_excel(r"C:\Users\darek\wszystko_z_VIAF.xlsx", sheet_name=0)
dict_viaf = dict(zip(viafs.nazwisko.tolist(),viafs.viaf.tolist()))
df['viaf']=df['Nazwa'].map(dict_viaf)
df.to_excel("pierwszy1000_po_weryf_zVIAF.xlsx", sheet_name='Sheet_name_1') 