# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:10:45 2023

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
my_marc_files = ["C:/Users/dariu/BN_czarek/msplit00000015.mrc",
"C:/Users/dariu/BN_czarek/msplit00000000.mrc",
"C:/Users/dariu/BN_czarek/msplit00000001.mrc",
"C:/Users/dariu/BN_czarek/msplit00000002.mrc",
"C:/Users/dariu/BN_czarek/msplit00000003.mrc",
"C:/Users/dariu/BN_czarek/msplit00000004.mrc",
"C:/Users/dariu/BN_czarek/msplit00000005.mrc",
"C:/Users/dariu/BN_czarek/msplit00000006.mrc",
"C:/Users/dariu/BN_czarek/msplit00000007.mrc",
"C:/Users/dariu/BN_czarek/msplit00000008.mrc",
"C:/Users/dariu/BN_czarek/msplit00000009.mrc",
"C:/Users/dariu/BN_czarek/msplit00000010.mrc",
"C:/Users/dariu/BN_czarek/msplit00000011.mrc",
"C:/Users/dariu/BN_czarek/msplit00000012.mrc",
"C:/Users/dariu/BN_czarek/msplit00000013.mrc",
"C:/Users/dariu/BN_czarek/msplit00000014.mrc"]

field650=pd.read_excel('C:/Users/dariu/Downloads/pbl_marc_articles.xlsx', sheet_name='Sheet1',dtype=str)
listy=dict(zip(field650['001'].to_list(),field650['600'].to_list()))


zviaf={}
bezviaf={}
records=[]
allrec=[]
antoherbad=[]

for my_marc_file in tqdm(my_marc_files):
    savefile=my_marc_file.split('/')[-1].split('.')[0]
    records=[]
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in reader:
            
            records.append(record)
    writer = TextWriter(open(savefile+'mrk','wt',encoding="utf-8"))
    for record in records:
        #allrecords.append(record)
    ### and write each record to it
        writer.write(record)
    writer.close() 
            
            
            
            try:
                for my_field in record:
                #### Control fields (in the range 00x) don't have indicators. 
                #### We use this try/except catcher to allow us to elegantly handle both cases without encountering a breaking error, or a coding failure
                    try:  
                        ind_1 = my_field.indicator1
                        #print(ind_1)
                        if not ind_1.isdigit() and not ind_1==" ":
                            #print(type(record))
                            records.append(record)
                       
                    except AttributeError:
                        continue
            except:
                antoherbad.append(record)
                    
                    print ("\tTag #:", my_field.tag)
                    ind_2 = my_field.indicator2
    
                    #### Setting an empty indicator to a more conventional and readable "/"
                    if ind_1 == " ":
                        ind_1 = "/"
                    if ind_2 == " ":
                        ind_2 = "/"
    
                    print ("\tTag #:", my_field.tag, "Indicator 1:", ind_1 , "Indicator 2:", ind_2)
                except AttributeError:
                    print ("\tTag #:", my_field.tag)
    
                for my_subfield_key, my_subfield_value in my_field:
                    print ("\t\t", my_subfield_key, my_subfield_value)
                print ()
            print ()

                
            
            
            
            #print(record['600'])
            
        
            for field in record:
                
                x= field.indicators1
                if x:
                    print(x)
            
            with warnings.catch_warnings(record=True) as found_warnings:
                records.append(record)
            if record is None:
                print(
                    "Current chunk: ",
                    reader.current_chunk,
                    " was ignored because the following exception raised: ",
                    reader.current_exception
                )
            else:
                print(record['001'])
            
        for record in tqdm(reader):
            records.append(record)
            my_500s = record.get_fields('100','700')
            for my in my_500s:
                x= my.get_subfields('1')
                if x:
                    
                    #print (my['a']+'   '+ my['1'])
                    if my['a'] in zviaf:
                        zviaf[my['a']][1]+=1
                    else:
                        zviaf[my['a']]=[my['1'],1]
                        
                else:
                    if my['a'] in bezviaf:
                        bezviaf[my['a']]+=1
                    else:
                        bezviaf[my['a']]=1
                    
                    
                  
                    
               


            if subfields['100']['1']:
                print(fields)
            else:
                print(fields)
                
        print(record['245'].value())
        print(record.pos)
        print(record.pubyear())
            
        record.add_field(
            Field(
                tag = '650',
                indicators = ['0','1'],
                subfields = [
                    'a', 'The pragmatic programmer : ',
                    'b', 'from journeyman to master /',
                    'c', 'Andrew Hunt, David Thomas.'
                ]))
        print(record)
        print ("Subfield 'a':", record['245']['a'])
    	# print ("Subfield 'b':", record['245']['b'])
    	# print ("Subfield 'c':", record['245']['c'])
        

	    my_500s = record.get_fields('100', '776')
	    for my_500 in my_500s:
		    print (my_500)
        print ()
        print (record['245'])
        print (type(record['245']))
        print ()
        print (record['245'].value())
        print (type(record['245'].value()))
        print ()
        print (record['999'])
        print (type(record['245']['a']))
        print ()
        print (record.title())
        if record.title()=='Joka uniinsa uskoo : jännitysromaani /':
            print('lala')
        print (type(record.title()))
        #quit()
with open(my_marc_file, 'rb') as data:
    reader = MARCReader(data)        
    for record in reader:
        print(record.author())
        print(record.isbn())
        print(record.issn())
        print(record.issn_title())
        record.leader
        record.location()
        print(record.pos)
        record.publisher()
        record.pubyear()
        record.series()
        record.sudoc()
        record.title()
        record.uniformtitle()
        record.notes()
        print(record.subjects())
        print(record.physicaldescription())
        print (record['245'].get_subfields('a'))
        print ("Field 245 indicator 1: {}".format(record['245'].indicator1))
        print ("Field 245 indicator 2: {}".format(record['245'].indicator2))   
        my_245s = record.get_fields('245')
        for my_245 in my_245s:
            
            my_245_subfields = my_245.get_subfields('a', 'b', 'c', 'f', 'g', 'h', 'k', 'n', 'p', 's', '6', '8')
            for my_245_subfield in my_245_subfields:
                print (my_245_subfield)
        #quit()    
from pymarc import Record, Field

record = Record()
record.add_field(
    Field(
        tag = '245',
        indicators = ['0','1'],
        subfields = [
            'a', 'The pragmatic programmer : ',
            'b', 'from journeyman to master /',
            'c', 'Andrew Hunt, David Thomas.'
        ]))
writer = TextWriter(open('check.mrk','wt',encoding="utf-8"))
for record in records:
    #allrecords.append(record)
### and write each record to it
    writer.write(record)
writer.close() 

my_new_marc_filename = "my_new_marc_file.mrc" 
with open(my_new_marc_filename , 'wb') as data:
    for my_record in records:
        ### and write each record to it
        data.write(my_record.as_marc())
with open('f20232.xml' , 'wb') as data:
    for my_record in records:

        memory = BytesIO()
        writer = XMLWriter(memory)
        writer.write(my_record)
# writer.close(close_fh=False) 
        # data.write(my_record.as_marc())
        #data.write(my_record.as_dict())
        data.write(writer.as_xml())
        # data.write(my_record.as_marc())
        # data.write(my_record.as_marc21())
writer = XMLWriter(open('f20232.xml', 'wb'))
for record in records[:3]:
    
    #allrecords.append(record)
### and write each record to it
    writer.write(record)
writer = JSONWriter(open('check.json','wt',encoding="utf-8"))
for record in records[:3]:
    #allrecords.append(record)
### and write each record to it
    writer.write(record)
writer.close() 
v='1'
v.isdigit()

with open('espana.mrc','wb') as data1, open('D:/Nowa_praca/Espana/MONOMODERN/MONOMODERN.mrc', 'rb') as data:
    reader = MARCReader(data)
    counter=0
    for record in tqdm(reader):
        switch=False
        try:
            my = record.get_fields('080')
            for field in my:
                subfields=field.get_subfields('a')
                field.subfields
                
                for subfield in subfields:
                    if subfield.startswith('82'):
                        #print(subfield)
                        switch=True
            if switch:
                counter+=1
                print(counter)
                
                
                data1.write(record.as_marc())
        except:
            pass
