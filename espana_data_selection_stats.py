# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:43:10 2023

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
from definicje import *
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
switch=False      
with open('D:/Nowa_praca/Espana/espana.mrc', 'rb') as data, open('espanaviaf_7.mrc','wb') as data1:
    reader = MARCReader(data)
    counter=0
    publish_place={}
    for record in tqdm(reader):
        if record['001'].value()=='a6006543':
            switch=True
            continue
        if switch:
            #print(record)
            try:
                my = record.get_fields('100', '700','600')
                for field in my:
                    subfields=field.get_subfields('0')
                    orginal_field=field.subfields
                    
                    viaf=[]
                    for subfield in subfields:
                        if subfield.startswith('http'):
                            identifier=subfield.split('/')[-1]
                            url =f"https://datos.bne.es/resource/{identifier}.jsonld"
                            #print(url)
                            data = requests.get(url).json()['@graph']
                            for d in data:
                                if 'P5024' in d:
                                    external_identifiers=d['P5024']
                                    if type(external_identifiers)==list:
                                        for external in external_identifiers:
                                            if external.startswith('http://viaf'):
                                                
                                                #print(external)
                                                viaf.append('1')
                                                viaf.append(external)
                                                
                                    else:
                                        
                                            if external_identifiers.startswith('http://viaf'):
                                                viaf.append('1')
                                                viaf.append(external_identifiers)
                        else:
                            
                            #print(subfield)
                            url =f"https://datos.bne.es/resource/{subfield}.jsonld"
                            #print(url)
                            data = requests.get(url).json()['@graph']
                            for d in data:
                                if 'P5024' in d:
                                    external_identifiers=d['P5024']
                                    if type(external_identifiers)==list:
                                        for external in external_identifiers:
                                            if external.startswith('http://viaf'):
                                                
                                                #print(external)
                                                viaf.append('1')
                                                viaf.append(external)
                                                
                                    else:
                                        
                                            if external_identifiers.startswith('http://viaf'):
                                                viaf.append('1')
                                                viaf.append(external_identifiers)
                            
                                            
                    if viaf:
                        field.subfields=orginal_field+viaf
                try:        
                    data1.write(record.as_marc())
                except:
                    print(record)
            except:
                try:
                     data1.write(record.as_marc())   
                except:
                    print(record)
                        # if subfield not in publish_place:
                    #     publish_place[subfield]=1
                    # else:
                    #     publish_place[subfield]+=1
                        
                  
            
places=pd.DataFrame.from_dict(publish_place, orient='index')
places.to_excel("publication_places_espana.xlsx")
url =f"https://datos.bne.es/resource/XX919455.jsonld"
#print(url)
data = requests.get(url).json()['@graph']


## PROBA
switch=False      
with open('D:/Nowa_praca/Espana/espana.mrc', 'rb') as data, open('espanaviaf_7.mrc','wb') as data1:
    reader = MARCReader(data)
    counter=0
    publish_place={}
    for record in tqdm(reader):
        if record['001'].value()=='a6006543':
            switch=True
            continue
        if switch:
            #print(record)
            
                my = record.get_fields('100', '700','600')
                for field in my:
                    subfields=field.get_subfields('0')
                    orginal_field=field.subfields
                    
                    viaf=[]
                    for subfield in subfields:
                        if subfield.startswith('http'):
                            identifier=subfield.split('/')[-1]
                            try:
                                url =f"https://datos.bne.es/resource/{identifier}.jsonld"
                                #print(url)
                                data = requests.get(url).json()['@graph']
                            except:
                                data=[]
                            if data:
                                for d in data:
                                    if 'P5024' in d:
                                        external_identifiers=d['P5024']
                                        if type(external_identifiers)==list:
                                            for external in external_identifiers:
                                                if external.startswith('http://viaf'):
                                                    
                                                    #print(external)
                                                    viaf.append('1')
                                                    viaf.append(external)
                                                    
                                        else:
                                            
                                                if external_identifiers.startswith('http://viaf'):
                                                    viaf.append('1')
                                                    viaf.append(external_identifiers)
                        else:
                            
                            #print(subfield)
                            try:
                                url =f"https://datos.bne.es/resource/{subfield}.jsonld"
                                #print(url)
                                data = requests.get(url).json()['@graph']
                            except:
                                data=[]
                            if data:
                                for d in data:
                                    if 'P5024' in d:
                                        external_identifiers=d['P5024']
                                        if type(external_identifiers)==list:
                                            for external in external_identifiers:
                                                if external.startswith('http://viaf'):
                                                    
                                                    #print(external)
                                                    viaf.append('1')
                                                    viaf.append(external)
                                                    
                                        else:
                                            
                                                if external_identifiers.startswith('http://viaf'):
                                                    viaf.append('1')
                                                    viaf.append(external_identifiers)
                                
                                            
                    if viaf:
                        field.subfields=orginal_field+viaf
                      
                data1.write(record.as_marc())
               

 
    
