# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:19:57 2022

@author: dariu
"""
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
#from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
from time import time

from concurrent.futures import ThreadPoolExecutor
import threading
import xlsxwriter

#%% 


plik=["C:/Users/dariu/Desktop/msplit00000005.mrk"]
for p in tqdm(plik):
    file=p.split('/')[-1]
    x=list_of_dict_from_file(p)
    article=LDR_article(x)  
    to_file2('artykuly_'+file,article)
    
    
    
import os

# Ścieżka do katalogu
folder_path = r"D:/work_space/"

# Iteracja przez pliki w katalogu
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Sprawdzenie, czy nazwa pliku zawiera 'ldr' (ignorując wielkość liter)
        if 'ldr' in file.lower():
            # Wyświetlenie pełnej ścieżki do pliku
            print(f"Znaleziono: {os.path.join(root, file)}")
