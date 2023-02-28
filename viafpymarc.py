# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:21:24 2023

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
my_marc_files = ["C:/Users/dariu/pbl_articles.mrc"]


zviaf={}
bezviaf={}
records=[]
allrec=[]
antoherbad=[]
records=[]
for my_marc_file in tqdm(my_marc_files):
    savefile=my_marc_file.split('/')[-1]
    with open('pbl_articles_21-02-2023.mrc','wb') as data1, open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in reader:
            print(record['001'].value())
            
            
                        
            #print(record)
            
            for field in record:

                if field.tag=='600':
                    print(field.subfields)
                    #if '0' in field.subfields:
                    x=get_indexes(field.subfields, '0')
                    print(x)
                    for y in  x:
                        field.subfields[y]='1'
                        field.subfields[y+1]=field.subfields[y+1].replace('viaf',r'http://viaf.org/viaf/')
                
                    
                    

            
            data1.write(record.as_marc())
       






