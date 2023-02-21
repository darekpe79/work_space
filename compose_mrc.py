# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:10:34 2023

@author: dariu
"""

from pymarc import MARCReader
from tqdm import tqdm
import requests
import json
from pymarc import MARCReader
from pymarc import parse_json_to_array
from pymarc import TextWriter
from pymarc import XMLWriter
from pymarc import JSONWriter
from io import BytesIO
import warnings
from pymarc import MARCReader
from pymarc import Record, Field 
import pandas as pd
from copy import deepcopy
from definicje import *
my_marc_files = ["D:/Nowa_praca/marki_08.02.2023/arto_2022-09-02_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/pbl_articles_2022-09-02_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/pbl_books_2022-09-02_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/fennica_2022-09-02_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/cz_chapters_2022-09-02_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/cz_books_2022-08-26_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/cz_articles4_2022-08-26_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/cz_articles3_2022-08-26_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/cz_articles2_2022-08-26_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/cz_articles1_2022-08-26_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/cz_articles0_2022-08-26_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/bn_chapters_2022-08-26_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/bn_books_2022-08-26_30-12-2022_08-02-2023.mrc",
"D:/Nowa_praca/marki_08.02.2023/bn_articles_2022-08-26_30-12-2022_08-02-2023.mrc"]

#field650=pd.read_excel('C:/Users/dariu/Downloads/pbl_marc_articles.xlsx', sheet_name='Sheet1',dtype=str)
#listy=dict(zip(field650['001'].to_list(),field650['600'].to_list()))


zviaf={}
bezviaf={}
records=[]
allrec=[]
antoherbad=[]
records=[]
counter=0
for my_marc_file in tqdm(my_marc_files):
    savefile=my_marc_file.split('/')[-1]
    with open(savefile,'wb') as data1,open(my_marc_file, 'rb') as data:
        
            reader = MARCReader(data)
            
            for record in reader:
                if record:
                
                    #print(record)
                    for field in record:
        
                            try:
                            
                                field.subfields=compose_data(field.subfields)
                            except:
                                continue
                    data1.write(record.as_marc())
                            
                else:
                    counter+=1
                
                    
                
         