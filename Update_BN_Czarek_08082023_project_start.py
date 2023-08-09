# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:51:14 2023

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
#viaf_combination
#tworzenie slowniczka
fields_to_check={}
my_marc_files = ["D:/Nowa_praca/marki_compose_19.05.2023/pbl_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/arto_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_articles_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_chapters_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles0_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles1_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles2_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles3_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles4_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_chapters_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/fennica_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/ksiazki_composed_unify2_do_wyslania.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_articles_21-02-2023compose.mrc", 'D:/Nowa_praca/marki_compose_19.05.2023/ksiazki_composed_unify2_do_wyslania.mrc']
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('700','600','100')
            
            
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_a=field.get_subfields('a')
                sub_d=field.get_subfields('d')
                sub_1=field.get_subfields('1')
                if sub_a and sub_d and sub_1:
                    text=''
                    for sub in sub_a+sub_d:
                        
                        for l in sub:
                            
                            if l.isalnum():
                                text=text+l
                    fields_to_check[text]=sub_1[0]
#uzycie slowniczka i dodanie viafow oczywistych                                
my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/libri_marc_bn_chapters_2023-08-07.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/libri_marc_bn_articles_2023-08-07.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/libri_marc_bn_books_2023-08-07.mrc"]
counter=0
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'new_viaf.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'new_viaf.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('700','600','100')
            
            
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_a=field.get_subfields('a')
                sub_d=field.get_subfields('d')
                sub_1=field.get_subfields('1')
                if sub_1:
                    continue
                else:
                    #print(field)
                    if sub_a and sub_d:
                        text=''
                        for sub in sub_a+sub_d:
                            
                            for l in sub:
                                
                                if l.isalnum():
                                    text=text+l
                        if text in fields_to_check:
                            counter+=1
                            print(text, fields_to_check[text])
                    
                            field.add_subfield('1', fields_to_check[text])
            #print(record)
            data1.write(record.as_marc())
            writer.write(record)    
writer.close()   
#%% 655 wzbogacenie
my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_chapters_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_books_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_articles_2023-08-07new_viaf.mrc"]

field650=pd.read_excel('D:/Nowa_praca/655_381_380_excele/bn_art_655_doMrk.xlsx', sheet_name='do_mrk',dtype=str)
listy=dict(zip(field650['desk655'].to_list(),field650['action2'].to_list()))
dictionary_to_check={}
for k,v in listy.items():
    #print(v)
    if type(v)!=float:
        dictionary_to_check[k]=v

for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open(my_marc_file+'genre_655.mrk','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data, open(my_marc_file+'genre_655.mrc','wb')as data1:
        reader = MARCReader(data)
        for record in tqdm(reader):
            
            
            # [e for e in record if e.tag=='381'][-1]['a']='test2'
            
            # for field in record:
                
            #     if field.tag=='381':
                    
            #         field['a']='test'
            #         field.subfields[3]='new'
            #         field.get_subfields('a')[0]='sraka'
            #         fie
            #         for sub in field.get_subfields('a'):
            #             print(sub)
                    
            
            # print(record)
            new_field=[]
            my = record.get_fields('655')
            
            for field in my:
                subfields=field.get_subfields('a')
                for subfield in subfields:
                    if subfield in dictionary_to_check:
                       
                        new_field.append(dictionary_to_check[subfield])
            if new_field:
                unique(new_field)
                my_new_380_field=None
                for new in new_field:
                    if 'Secondary literature' in new:
                        
                        
                    
                        my_new_380_field2 = Field(
            
                                tag = '380', 
            
                                indicators = ['\\','\\'],
            
                                subfields = [
                                               Subfield('i', 'Major genre'),
                                                Subfield('a', 'Secondary literature'),
                                                Subfield('l', 'eng'),
                                            ]
                                ) 

                        record.add_ordered_field(my_new_380_field2) 
                        
                    else:
                        
                        my_new_380_field = Field(
            
                                tag = '380', 
            
                                indicators = ['\\','\\'],
            
                                subfields = [
                                                Subfield('i', 'Major genre'),
                                                Subfield('a', 'Literature'),
                                               Subfield( 'l', 'eng'),
                                            ]
                                ) 

                        
                        
                        my_new_245_field = Field(
                            tag='381',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield(code='i', value='Major genre'),
                                Subfield(code='a', value=new),
                                Subfield(code='l', value='eng')
                            ]
                        )

                        record.add_ordered_field(my_new_245_field)
                if my_new_380_field:        
                    record.add_ordered_field(my_new_380_field)        

### adding the new field
            
            # record.add_ordered_field(my_new_245_field)
            # record['380']['a'] = 'The Zombie Programmer '
            # print(record['380'])
            data1.write(record.as_marc())
            writer.write(record)    
writer.close() 

#650

field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/650__do_pracy_wszystko.xlsx', sheet_name='bn2',dtype=str)
listy=dict(zip(field650['desk_650'].to_list(),field650['to_use'].to_list()))
dictionary_to_check={}
patterna=r'(?<=\$a).*?(?=\$|$)' 
for k,v in listy.items():
    k=re.findall(patterna, k)[0]
    #print(v)
    if type(v)!=float:
        dictionary_to_check[k]=v
        
my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_chapters_2023-08-07new_viaf.mrcgenre_655.mrc650genre.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_books_2023-08-07new_viaf.mrcgenre_655.mrc650genre.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_articles_2023-08-07new_viaf.mrcgenre_655.mrc650genre.mrc"]
for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open(my_marc_file+'650nation.mrk','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data, open(my_marc_file+'650nation.mrc','wb')as data1:
        reader = MARCReader(data)
        for record in tqdm(reader):
           # print(record)
            
            # [e for e in record if e.tag=='381'][-1]['a']='test2'
            
            # for field in record:
                
            #     if field.tag=='381':
                    
            #         field['a']='test'
            #         field.subfields[3]='new'
            #         field.get_subfields('a')[0]='sraka'
            #         fie
            #         for sub in field.get_subfields('a'):
            #             print(sub)
                    
            
            # print(record)
            new_field=[]
            my = record.get_fields('650')
            
            for field in my:
                subfields=field.get_subfields('a')
                for subfield in subfields:
                    if subfield in dictionary_to_check:                     
                          
                        new_field.append(dictionary_to_check[subfield].capitalize())
            if new_field:
                unique(new_field)
                print(new_field)
                for new in new_field:
                    
                        
                        
                    
                        my_new_245_field = Field(
            
                                tag = '650', 
            
                                indicators = [' ',' '],
            
                                subfields = [Subfield                                                
                                                ('a', new),
                                                Subfield('2', 'ELB-n')
                                            ])
                                        
                                            
                                

                        record.add_ordered_field(my_new_245_field)
                        
                  
                        

### adding the new field
            
            # record.add_ordered_field(my_new_245_field)
            # record['380']['a'] = 'The Zombie Programmer '
            # print(record['380'])
            data1.write(record.as_marc())
            writer.write(record)    
writer.close()                
