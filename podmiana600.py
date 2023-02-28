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
my_marc_files = ["C:/Users/dariu/pbl_articles_2022-09-02_30-12-2022_08-02-2023.mrc"]

field650=pd.read_excel('C:/Users/dariu/Downloads/pbl_marc_articles.xlsx', sheet_name='Sheet1',dtype=str)
listy=dict(zip(field650['001'].to_list(),field650['600'].to_list()))


zviaf={}
bezviaf={}
records=[]
allrec=[]
antoherbad=[]
records=[]
for my_marc_file in tqdm(my_marc_files):
    savefile=my_marc_file.split('/')[-1]
    with open('pblrpobaviafy.mrc','wb') as data1, open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in reader:
            print(record['001'].value())
            
            
                        
            #print(record)
            
            for field in record:

                if field.tag=='600':
                    
                    # for key,val in field.subfields_as_dict().items():
                    #    print(key,val)
                    #field.subfields_as_dict()['a']
                    try:  
                        ind_1 = field.indicator1
                        #print(ind_1)
                        if not ind_1.isdigit() and not ind_1==" ":
                            if record['001'].value() in listy:
                                newvalue=listy[record['001'].value()].split('$')
                                indicators=[e for e in newvalue[0]]
                                subfileds_values=[]
                                for values in newvalue[1:]:
                                    
                                    if len(values)>1:
                                        subfileds_values.append(values[0])
                                        subfileds_values.append(values[1:])
                            print(subfileds_values)
                            field.indicators=indicators
                            field.subfields=compose_data(subfileds_values)
                    except:
                        continue
            
            data1.write(record.as_marc())
       






