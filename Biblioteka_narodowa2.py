# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 09:16:31 2023

@author: dariu
"""
import requests
import json
from pymarc import MARCReader
from pymarc import parse_json_to_array
from pymarc import TextWriter
import pandas as pd
from definicje import *

with open('C:/Users/dariu/literaturoznawstwo_polon.json', encoding='utf-8') as fh:
    dataname = json.load(fh)
namesall=[]   
for names in dataname:
    print (names['firstName'])
    print (names['lastName'])
    namesall.append(names['firstName']+' '+ names['lastName'])
    
names=pd.DataFrame(dataname) #orient='index')    
names.to_excel("lista_literaturoznawstwo.xlsx", sheet_name='Sheet_name_1') 
allrecords=[]
for name in namesall:      
    print(name)
    
    url =f"https://data.bn.org.pl/api/institutions/bibs.json?author={name}+%281989-+%29&amp;limit=100"
    print(url)
    data = requests.get(url).json()
    
    if data['bibs']:
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
        #writer = TextWriter(open(name+'.mrk','wt',encoding="utf-8"))
        for record in records:
            allrecords.append(record)
        ### and write each record to it
            writer.write(record)

        writer.close()
writer = TextWriter(open('all_literaturoznawstwo1900records.mrk','wt',encoding="utf-8"))
for record in allrecords:
    #allrecords.append(record)
### and write each record to it
    writer.write(record)
writer.close()       

with open(name+'.mrc' , 'wb') as data1:
    for my_record in records:
### and write each record to it

        data1.write(my_record.as_marc())
    writer.close()  
            
path='C:/Users/dariu/all_literaturoznawstwo1900records.mrk'
listofphrases=['Artykuł z czasopisma kulturalnego','Artykuł z czasopisma społeczno-kulturalnego']
dictrec=list_of_dict_from_file(path) 
recordskultural=[]
rodzaj={}
for record in dictrec:
    #record.get('955')
    if record.get('955'):
        # if any(phrase in line for phrase in listofphrases):
        #     recordskultural.append(record)
        for key,val in record.items():
            if key=='955':
                for line955 in val:
                    print(key)
                    
                    if line955 in rodzaj:
                        rodzaj[line955]+=1
                    else:
                        rodzaj[line955]=1
                    
            #print(line)
           
                
    
to_file2('onlycultural_literaturoznawstwo.mrk',recordskultural)

data=pd.DataFrame(dictrec)
data.to_excel("records_literaturoznawstwo.xlsx", sheet_name='Sheet_name_1') 

data=pd.DataFrame.from_dict(rodzaj,orient='index')
data.to_excel("655stats.xlsx", sheet_name='Sheet_name_1') 


