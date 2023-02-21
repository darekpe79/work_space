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
my_marc_files = ["D:/Nowa_praca/marki_08.02.2023/proba655.mrc"]

field650=pd.read_excel('C:/Users/dariu/Downloads/pbl_marc_articles.xlsx', sheet_name='Sheet1',dtype=str)
listy=dict(zip(field650['001'].to_list(),field650['600'].to_list()))


zviaf={}
bezviaf={}
records=[]
allrec=[]
antoherbad=[]
records=[]
for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open(my_marc_file.replace('.mrc','.mrk'),'wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in reader:
            #record['001'].value()
            
            if 'pl000008663' in listy:
                newvalue=listy['pl000008663'].split('$')
                indicators=[e for e in newvalue[0]]
                subfileds_values=[]
                for values in newvalue[1:]:
                    if len(values)>1:
                        
                        
                    
                    
                
            
            #print(record)
            for field in record:
                field.indicators
                field.subfields
                if field.tag=='655':
                    
                    # for key,val in field.subfields_as_dict().items():
                    #    print(key,val)
                    #field.subfields_as_dict()['a']
                    try:  
                        ind_1 = field.indicator1
                        #print(ind_1)
                        if not ind_1.isdigit() and not ind_1==" ":
                            #print(type(record))
                            print(field.as_marc())
                    except:
                        continue
            
            
            writer.write(record)
    writer.close()    
                
                #print(field['a'])
                for subfield in field.get_subfields('a'):
                    print(subfield)
               #print(fields.__str__())
                #print(fields.value())
                
                field.tag
                new_subfields=[]
                for subfield in field:
                    print(subfield)
                    new_subfield=compose_data(subfield[1])
                    new_subfields.extend([subfield[0],new_subfield])
                    

                field.subfields=new_subfields
                
                
                    #allrecords.append(record)
                ### and write each record to it
            writer.write(record)
    writer.close()
                    





from pymarc import Record

my_new_record = Record()
for field in my_new_record:
    field.add_subfield('u', 'http://www.loc.gov', 0)
    print(field)