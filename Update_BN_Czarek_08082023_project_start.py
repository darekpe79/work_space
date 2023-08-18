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
                        
                  
                        


            data1.write(record.as_marc())
            writer.write(record)    
writer.close()                

#%%
#add issns by title
fields_to_check={}
my_marc_files = ["D:/Nowa_praca/marki_compose_19.05.2023/arto_21-02-2023compose.mrc",
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
"D:/Nowa_praca/marki_compose_19.05.2023/es_articles_sorted_31.05.2023.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/fennica_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_articles_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/sp_ksiazki_composed_unify2_do_wyslanianew_viaf.mrc"]
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('773')
            if len(my)==1:
            
            
                for field in my:
                   # field.add_subfield('d', '1989-2022')
                    sub_t=field.get_subfields('t')
                    sub_x=field.get_subfields('x')
                    
                    if sub_x and sub_t:
                        fields_to_check[sub_t[0]]=sub_x[0]


my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_chapters_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_libri_marc_bn_articles_2023-08-07new_viaf.mrc"]
counter=0
for my_marc_file in tqdm(my_marc_files):
    #filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(my_marc_file+'+773x.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(my_marc_file+'+773x.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('773')
        
        
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_t=field.get_subfields('t')
                
                sub_x=field.get_subfields('x')
                if sub_x:
                    
                    continue
                else:
                    #print(field)
                    
                    if sub_t:
                        if sub_t[0] in fields_to_check:
                            
                            
                    
                            field.add_subfield('x', fields_to_check[sub_t[0]])
            #print(record)
            data1.write(record.as_marc())
            writer.write(record)    
writer.close()   
#%%773 records s
fields_to_check={}
my_marc_files = ["D:/Nowa_praca/marki_compose_19.05.2023/arto_21-02-2023compose.mrc",
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
"D:/Nowa_praca/marki_compose_19.05.2023/es_articles_sorted_31.05.2023.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/fennica_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_articles_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/sp_ksiazki_composed_unify2_do_wyslanianew_viaf.mrc"]
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('773')
            if len(my)==1:
            
            
                for field in my:
                   # field.add_subfield('d', '1989-2022')
                    sub_x=field.get_subfields('x')
                    sub_s=field.get_subfields('s')
                    
                    if sub_x and sub_s:
                        fields_to_check[sub_x[0]]=sub_s[0]
                            
                        
my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_chapters_2023-08-07new_viaf.mrc+773x.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_libri_marc_bn_articles_2023-08-07new_viaf.mrc+773x.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf.mrc+773x.mrc"]
counter=0
for my_marc_file in tqdm(my_marc_files):
    #filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(my_marc_file+'+773s.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(my_marc_file+'+773s.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('773')
        
        
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_x=field.get_subfields('x')
                
                sub_s=field.get_subfields('s')
                if sub_s:
                    counter+=1
                    continue
                else:
                    #print(field)
                    
                    if sub_x:
                        if sub_x[0] in fields_to_check:
                            
                            
                    
                            field.add_subfield('s', fields_to_check[sub_x[0]])
            #print(record)
            data1.write(record.as_marc())
            writer.write(record)    
writer.close()   



#%% 710
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
import regex as re
from datetime import date

today = date.today()

# dd/mm/YY
d1 = today.strftime("%d-%m-%Y")
field26x=pd.read_excel('D:/Nowa_praca/publishers_work/do_ujednolicania_viafowania/wszystko_bez_710_matcher710-26x.xlsx', sheet_name='publisher',dtype=str)
dictionary26x=field26x.to_dict('records')
field710=pd.read_excel('D:/Nowa_praca/publishers_work/do_ujednolicania_viafowania/710_bezdupli_bez_Fin_po_ISNI_Publishers+instytucje.xlsx', sheet_name='publisher',dtype=str)
dictionary710=field710.to_dict('records')
fin11=pd.read_excel('D:/Nowa_praca/publishers_work/do_ujednolicania_viafowania/fin_po_isni_do viafowania710.xlsx', sheet_name='Arkusz1',dtype=str)
dictionaryfin11=fin11.to_dict('records')
concat26x_710=(pd.concat([field26x,field710]))
dictionaryconcat=concat26x_710.to_dict('records')
patterna=r'(?<=\$a).*?(?=\$|$)' 
#daty
patternb='(?<=\$b).*?(?=\$|$)'
patternviaf='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
patfin11=r'(?<=\(FIN11\)).*?(?=$| |\$)'

paths=["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_chapters_2023-08-07new_viaf.mrcgenre_655.mrc650genre.mrc650nation.mrc+773s",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_books_2023-08-07new_viaf.mrcgenre_655.mrc650genre.mrc650nation.mrc+773s",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_articles_2023-08-07new_viaf.mrcgenre_655.mrc650genre.mrc650nation.mrc+773s"]


#daty
pattern4='(?<=\$v).*?(?=\$|$)'
#pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
slownik={'Wojciech Sumliński Reporter':"12345"}



#val100=[]
for plik in paths:
    record=list_of_dict_from_file(plik)
    nametopath=plik.split('/')[-1].split('.')[0]+'_'
    
    for rec in tqdm(record):
        if '710' not in rec:
            fieldstocheck=[]
            ujednolicone_slownik={}
            if '260' or '264' in rec:
                
                
                field260=rec.get('260',['nicniema'])
                field264=rec.get('264',['nicniema'])
                fieldstocheck.extend(field260)
                fieldstocheck.extend(field264)
            for field in fieldstocheck:
                name = re.findall(patternb, field)
                if name:
                    for n in name:
                        name1=n.strip(', . :').casefold()
                        #print(name1)
                        for records in dictionary26x:
                            if name1 in records.values():
                         #      print(records)
                               
                               ujednolicone=records['Oryginal_710_po_VIAF_i_z_ujednolicone_bez_710']
                               if not isinstance(ujednolicone, float):
                                   ujednolicone=records['Oryginal_710_po_VIAF_i_z_ujednolicone_bez_710'].strip(', . :')
                               else:
                                   ujednolicone=records['Oryginal_260_264'].strip(', . :')
                            
                               ujednolicone_slownik[records['VIAF']]=ujednolicone
            if ujednolicone_slownik:
                
                rec['710']=[]
                for viaf,names in ujednolicone_slownik.items():
                    #print(viaf,names)
                    
                    rec['710'].append(r'2\$a'+names+r'$1http://viaf.org/viaf/'+viaf)


                                   
        else:      
                
            
                       
            for key, val in rec.items():
                
                
            #     '''badam czy są pola 260, 264 i 710 i porównam ich długosc, następnie należy zbadać czego ewentualnie brakuje
            #       (po słowniku- brać oryginał z 26x i szukać w ujednoliconym 710)jesli np. sa 3 260 a 2 710 w slowniku te dwa znajde i wiem które są 710, to wiem którego nie ma
            #     i jesli mam ujednolicone brakujace 260 to moge stworzyc nowe 710 w innym wypadku (jesli znajde, ale nie wszystko, lub nic, to nie nic nie zrobię z automatu, bo mogłoby się zdublować'''
            #     if {"710", "260"} <= rec.keys():
            #         #print(key, val)
                    
    
                    
            #         if key=='260':
            #             for v in val:
                    
            #                 sub260_b=re.findall(patternb, v)
            #                 len260=len(sub260_b)
            #                 #print(len260)
            #         if key=='710':
            #             len_val=len(val)
            # print(len_val)             #print(len_val)
                   
            # if len260 and len_val:
            #             print('ok')
            # if len260!=len_val:
            #     print(val,'BBBBBBBBBBBBBB', sub260_b)
                        
                        
                        
                    
                    
                
                if key=='710':
                    #print(rec)
                    
                    #new_val=[]
                    for v in val: 
                    
                        #print(v)
                        name = re.findall(patterna, v)
                        #print(name)
                        fin11finder=re.findall(patfin11, v)
                        
                        if fin11finder:
                            for records in dictionaryfin11:
                                if fin11finder[0] in records.values():
                                    #print(records['viaf'])
                                    index=[i for i, e in enumerate(val) if e == v]
                                    #print(index)
                                    for i in index:
                                     #   print(val[i])
                                        valstrip=val[i]
                                        new_val=val[i]+r'$1http://viaf.org/viaf/'+records['viaf']
                                        val[i]=new_val.replace(name[0], name[0].strip(', . :'))
                            
                        elif name:
                            name1=name[0].strip(', . :').casefold()
                            for records in dictionary710:
                                if name1 in records.values():
                                    #print(records['VIAF'])
                                    index=[i for i, e in enumerate(val) if e == v]
                                    #print(index)
                                    for i in index:
                                        #print(val[i])
                                        
                                        new_val=val[i]+r'$1http://viaf.org/viaf/'+records['VIAF']
                                        val[i]=new_val.replace(name[0], name[0].strip(', . :'))
                                        
        
                                
                            
                    
                    
                    
                    
                    
    to_file2(nametopath+d1+'.mrk',record)        


#%% 995 add
from pymarc import MARCReader, MARCWriter, Field, Subfield




def modify_995(file_path, output_path):
    with open(file_path, 'rb') as marc_file, open(output_path, 'wb') as out_file:
        reader = MARCReader(marc_file)
        writer = MARCWriter(out_file)
        
        for record in reader:
            
            field_995 = record['995']
            
            if field_995:
                # Extract all subfields
                original_subfields = field_995.subfields
        
                # Create new subfields
                new_subfields = [Subfield(code='a', value='Polska Bibliografia Literacka')]
        
                # The following subfield codes should be incremented ('b', 'c', ...)
                next_code = 'b'
                
                for sf in original_subfields:
                    new_subfields.append(Subfield(code=next_code, value=sf.value))
                    # Increment the next_code for the next iteration
                    next_code = chr(ord(next_code) + 1)
                
                # Create a new 995 field with these subfields
                new_field_995 = Field(tag='995', indicators=[' ', ' '], subfields=new_subfields)
                
                # Replace the old 995 field with the new one
                record.remove_field(field_995)
                record.add_field(new_field_995)
            
            writer.write(record)

        writer.close()
# Test the function
# modify_995('path_to_input_file', 'path_to_output_file')



# Example usage
modify_995('C:/Users/dariu/viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf_11-08-2023.mrc',
           'C:/Users/dariu/995viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf_11-08-2023.mrc')


#TO TRY:
# with open('C:/Users/dariu/proba_995.mrc', 'rb') as marc_file, open('C:/Users/dariu/proba_99522.mrc', 'wb') as out_file:
#     reader = MARCReader(marc_file)
#     writer = MARCWriter(out_file)
    
#     for record in reader:
#         print(record)
#         field_995 = record['995']
        
#         if field_995:
#             # Extract all subfields
#             original_subfields = field_995.subfields
    
#             # Create new subfields
#             new_subfields = [Subfield(code='a', value='Polska Bibliografia Literacka')]
    
#             # The following subfield codes should be incremented ('b', 'c', ...)
#             next_code = 'b'
            
#             for sf in original_subfields:
#                 new_subfields.append(Subfield(code=next_code, value=sf.value))
#                 # Increment the next_code for the next iteration
#                 next_code = chr(ord(next_code) + 1)
            
#             # Create a new 995 field with these subfields
#             new_field_995 = Field(tag='995', indicators=[' ', ' '], subfields=new_subfields)
            
#             # Replace the old 995 field with the new one
#             record.remove_field(field_995)
#             record.add_field(new_field_995)
    
#         writer.write(record)

# writer.close()

#%%
## select only what we dont have and split files
fields_to_check=[]
my_marc_files = ["D:/Nowa_praca/marki_compose_19.05.2023/arto_21-02-2023compose.mrc",
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
"D:/Nowa_praca/marki_compose_19.05.2023/es_articles_sorted_31.05.2023.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/fennica_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_articles_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/sp_ksiazki_composed_unify2_do_wyslanianew_viaf.mrc"]
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            try:
                fields_to_check.append(record['001'].data)
            except:
                continue

x=set(fields_to_check)


was=[]
new=[]
my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/processed/llibri_marc_bn_chapters_2023-08-07_processed.mrc"]
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            if record['001'].data in x:
                was.append(record)
                
            else:
                new.append(record)
                



with open('WAS-marc_bn_chapters_2023-08-07_processed.mrc', 'wb') as marc_out:
    writer = MARCWriter(marc_out)
    for record in was:
        writer.write(record)
    writer.close()

# 2. Saving records using TextWriter for a human-readable version:
with open('WAS-marc_bn_chapters_2023-08-07_processed.mrk', 'w', encoding='utf-8') as text_out:
    writer = TextWriter(text_out)
    for record in was:
        
        writer.write(record)
#%%
#move 773 x in old files to proper place    
from pymarc import MARCReader, MARCWriter, Field, Subfield

def move_x_after_t_in_773(record):
    field_773 = record.get('773', None)

    # If the record doesn't have a 773 field, return as is
    if not field_773:
        return record

    x_subfield = next((subfield for subfield in field_773.subfields if subfield.code == 'x'), None)
    t_subfield = next((subfield for subfield in field_773.subfields if subfield.code == 't'), None)

    # If either subfield x or t doesn't exist, return the record as is
    if not x_subfield or not t_subfield:
        return record

    # Remove the $x subfield
    field_773.subfields.remove(x_subfield)

    # Find the position of the $t subfield and insert the $x subfield after it
    t_position = field_773.subfields.index(t_subfield)
    field_773.subfields.insert(t_position + 1, x_subfield)

    return record

# Function to process a MARC file
def process_marc_file(input_path, output_path):
    with open(input_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
        reader = MARCReader(input_file)
        writer = MARCWriter(output_file)

        for record in reader:
            modified_record = move_x_after_t_in_773(record)
            writer.write(modified_record)

        writer.close()
def process_marc_files(file_list):
    for file_path in file_list:
        # For demonstration purposes, I'm assuming you want to overwrite each file with its processed data.
        # If you'd like a different output naming convention, adjust the output_path accordingly.
        output_path = file_path.replace('.mrc', '_processed.mrc')
        process_marc_file(file_path, output_path)

# Call the function specifying the input and output file paths
process_marc_file('C:/Users/dariu/773_proba.mrc', 'path_to_output.mrc')
files_to_process = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_chapters_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_libri_marc_bn_articles_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc"]
process_marc_files(files_to_process)