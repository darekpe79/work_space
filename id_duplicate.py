from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
from time import time
from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()



plik=r"F:\Nowa_praca\NOWI CZESI_FENNICAPBL_BN_05.04.2022\arto.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)
unique=[]
duplicates=[]
for dic in tqdm(dictrec):
    if dic not in unique:
        unique.append(dic)
    else:
        duplicates.append(dic)
    


to_file('arto.mrk',unique)
                
