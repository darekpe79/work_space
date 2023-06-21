# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:43:06 2023

@author: dariu
"""
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
my_marc_files = ["D:/Nowa_praca/finowie_towarzystwa_naukowe/authorities-all (1)towarzystwa_naukowe.mrc"]
name={}
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'towarzystwa_naukowe.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'towarzystwa_naukowe.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            print(record)
            
            switch=False
            my = record.get_fields('110')
            
            for field in my:
                

                sub_a=field.get_subfields('a')
                for a in sub_a:
                    name[a]={'viaf':[],'name_variants':[]}
            
            my = record.get_fields('024')
            
            for field in my:
                

                sub_a_024=field.get_subfields('a')
                for a024 in sub_a_024:
                    name[a]['viaf'].append(a024)
                    
            my = record.get_fields('410', '510')
            
            for field in my:
                print(field)
                

                sub_a_45=field.get_subfields('a','b')
                a45=" ".join(sub_a_45)
                # for a45 in sub_a_45:
                    
                name[a]['name_variants'].append(a45)
                   
fin_df=pd.DataFrame.from_dict(name, orient='index')
fin_df.to_excel("15062023_BN_authorities_towarzystwa_naukowe.xlsx")      
                 
            data1.write(record.as_marc()) 
            writer.write(record)    
writer.close() 
#%%
#compare name to names in all bnbooks file

my_marc_files = ["D:/Nowa_praca/finowie_towarzystwa_naukowe/bibs-ksiazka.mrc"]
records_id={}
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'towarzystwa_naukowe.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'towarzystwa_naukowe.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            #print(record['001'].value())
            
            switch=False
            my = record.get_fields('110')
           # checks=[]
            for field in my:
                
                

                sub_a=field.get_subfields('a')
                for a in sub_a:
                    a=a.strip(', ')
                    if a in name:
                        
                        if a not in records_id:
                            records_id[a]={'viaf':name[a]['viaf'],'count':1,'record_id':[record['001'].value()]}
                        else:
                            records_id[a]['count']+=1
                            records_id[a]['record_id'].append(record['001'].value())
                            
                            
                    else:
                        for key, val in name.items():
                            #print(key,'.....',val)
                            if a in val['name_variants']:
                               
                                #print(a)
                                if a not in records_id:
                                    records_id[a]={'viaf':name[key]['viaf'],'count':1,'record_id':[record['001'].value()]}
                                else:
                                    records_id[a]['count']+=1
                                    records_id[a]['record_id'].append(record['001'].value())
                                
                                
                                break
fin_df=pd.DataFrame.from_dict(records_id, orient='index')
fin_df.to_excel("15062023_BN_stats_260_towarzystwa_naukowe.xlsx")                                   
c=[1,2,3,4,5,6]
d=['a','b','c','d','f']

for x in c:
    for y in d:
        if y=='d':
            continue
        print(x,y)
                             
                            
                            
                    