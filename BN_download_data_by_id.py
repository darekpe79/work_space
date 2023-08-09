# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:32:42 2023

@author: dariu
"""

import requests
import json
from pymarc import MARCReader
from pymarc import parse_json_to_array
from pymarc import MARCReader,JSONReader
from tqdm import tqdm
from pymarc import Record, Field, Subfield
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
from definicje import *

url = "https://data.bn.org.pl/api/institutions/bibs.json?author=Rokita%2C+Zbigniew+%281989-+%29&amp;limit=100"
data = requests.get(url).json()
nexturl=data['nextPage']

list=data['bibs']
while len(nexturl)>0:
    data = requests.get(data['nextPage']).json()
    nexturl=data['nextPage']
    
    
    list.extend(data['bibs'])
from pymarc import parse_json_to_array
lista=[]
counter=0
for l in list:
#    counter+=1
#    print(counter)


    marc=l['marc']
    lista.append(marc)
x=json.dumps(lista)

records = parse_json_to_array(x)
for record in records:
    print (record)
with open('my_new_marcArto_ISSN_ALL11.mrc' , 'wb') as data:
    for my_record in records:
### and write each record to it
        data.write(my_record.as_marc())

#%%


my_marc_files = ["D:/Nowa_praca/marki_compose_19.05.2023/libri_marc_bn_chapters_2023-08-03.mrc",]
id_list=[]
for my_marc_file in tqdm(my_marc_files):
   
    
    with open(my_marc_file, 'rb')as data:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            id_list.append(record['001'].data)
        

records_list=[]
# specify the ID of the record you want to fetch
for ids in tqdm(id_list):
    record_id = ids
    
    # build the URL
    url = f'https://data.bn.org.pl/api/institutions/bibs.json?id={record_id}'
    
    # send the GET request
    response = requests.get(url)
    
    # parse the JSON response
    data = response.json()
    list=data['bibs']
    while len(nexturl)>0:
        data = requests.get(data['nextPage']).json()
        nexturl=data['nextPage']
        
        
        list.extend(data['bibs'])
    from pymarc import parse_json_to_array
    lista=[]
    counter=0
    for l in list:
    #    counter+=1
    #    print(counter)
    
    
        marc=l['marc']
        lista.append(marc)
    x=json.dumps(lista)
    
    records = parse_json_to_array(x)
    for record in records:
        
        records_list.append(record)
writer = TextWriter(open('libri_marc_bn_chapters_2023-08-03.mrk','wt',encoding="utf-8"))       
with open('libri_marc_bn_chapters_2023-08-03.mrc' , 'wb') as data:
    for my_record in records_list:
        
### and write each record to it
        data.write(my_record.as_marc())
        writer.write(my_record)    
writer.close()