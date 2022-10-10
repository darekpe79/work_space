# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:11:59 2021

@author: darek
"""
import pandas as pd
from random import randint
from time import sleep
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm
from pymarc import *
import regex as re
pattern=r'\$[w7td]'
issnexcel = pd.read_excel (r"F:\Nowa_praca\libri\Iteracja 2021-07\pbl_marc_books_2021-8-4.mrk")
dic2=dict(zip(issnexcel['773'].tolist(),issnexcel['issn_y'].tolist() ))


my_marc_file = r"F:\Nowa_praca\libri\Iteracja 2021-07\libri_marc_bn_articles_2021-08-05!.mrc"

my_marc_records = []
allrec=[]
with open(my_marc_file, 'rb') as data:
    reader = MARCReader(data)
    for record in tqdm(reader):
        if record['773'] is not None:
            field=record['773']
            searchfor=str(record['773'])[6:]
            
            #print(searchfor)
            if searchfor in dic2:
                
                #print(searchfor)
                lenght=len(re.findall(pattern, searchfor))
                    
                #splited=searchfor.split('$')
                
                #lenght=len(splited)
                #print(dic2[searchfor])
                

                field.add_subfield('x', dic2[searchfor], lenght)
                    
                
                my_marc_records.append(record)
        allrec.append(record)           
                #record['773']['x'] = dic2[searchfor]
       
with open('libri_marc_bn_articles_2021-08-05!ISSN.mrc' , 'wb') as data:
    for my_record in my_marc_records:
### and write each record to it
        data.write(my_record.as_marc())
        
with open('libri_marc_bn_articles_2021-08-05!ISSN_ALL.mrc' , 'wb') as data:
    for my_record in allrec:
### and write each record to it
        data.write(my_record.as_marc())
        
        
#%%Poniedziałek- przepisaći dać warunek if $d in searchfor:
    
def convertToMrc(input):
    output = input.replace('output/', 'usmarc/')
    output = output.replace('.xml', '.mrc')
    writer = pymarc.MARCWriter(file(output, 'wb')) 
    records = pymarc.map_xml(writer.write, input) 
    writer.close()  
    
my_marc_file = r"F:\pbl_marc_books_2021-8-4.mrc"

my_marc_records = []
allrec=[]
with open(my_marc_file, 'rb') as data:
    reader = MARCReader(data)
    for record in tqdm(reader):
        allrec.append(record)
 
from pymarc import TextWriter
for record in tqdm(allrec):
    record=str(record)
    with open('pbl_books.txt', 'a',encoding='utf-8') as file:
        #if not record.endswith('\n'):
        #    file.write('\n')
        file.write(record)
        file.write('\n')
#%%
records=mark_to_list(r"F:\Nowa_praca\libri\Iteracja 2021-07\pbl_marc_books_2021-8-4_good995_ALL_good_VIAF.mrk")
from pymarc import Record, Field
for rec in records:
    
    rec1 = Record()

    with open('file.dat', 'wb') as out:
        out.write(record.as_marc())   
        
my_marc_file = r"F:\Nowa_praca\libri\Iteracja 2021-07\libri_marc_bn_articles_2021-08-05!.mrk"

my_marc_records = []
allrec=[]
with open(my_marc_file, 'r') as data:
    reader = MARCReader(data)
    for record in tqdm(reader):
        print(record)
#%%
  
# writing to a file
writer = TextWriter(open("F:\pbl_marc_books_2021-8-4_good995_ALL.txt",'wt'))
writer.write(allrec)
writer.close()