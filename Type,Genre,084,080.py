# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:07:56 2022

@author: darek
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
from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()
from concurrent.futures import ThreadPoolExecutor
import threading
import xlsxwriter


paths=["F:/Nowa_praca/24.05.2022Marki/arto.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_articles.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_books.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_chapters.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles0.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles1.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles2.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles3.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles4.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_books.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_chapters.mrk",
"F:/Nowa_praca/24.05.2022Marki/fennica.mrk",
"F:/Nowa_praca/24.05.2022Marki/PBL_articles.mrk",
"F:/Nowa_praca/24.05.2022Marki/PBL_books.mrk"]

paths2=["F:/Nowa_praca/24.05.2022Marki/PBL_books.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_articles.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_books.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_chapters.mrk",
"F:/Nowa_praca/24.05.2022Marki/PBL_articles.mrk"]

pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$v).*?(?=\$|$)'
#pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'


output={'600v':{},'610v':{},'611v':{},'630v':{},'648v':{},'650v':{},'651v':{},'655a':{},'655v':{}}

#val100=[]
for plik in paths2:
list_of_dict_from_file
                            
                    
                    
                            
                            
                            
                
                
                
                
                
                
                
                
                
                
                
                
                

                

                
                

        



