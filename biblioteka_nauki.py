# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:51:52 2021

@author: darek
"""

from sickle import Sickle
#from pymarc import MARCReader
#from pymarc import exceptions as exc
#from pymarc import parse_xml_to_array
import xml.etree.ElementTree as ET
from tqdm import tqdm
from bs4 import BeautifulSoup
from definicje import *
sickle = Sickle('https://bibliotekanauki.pl/api/oai/articles')

ns1 = '{http://www.openarchives.org/OAI/2.0/}'
#list_Sets=sickle.ListSets()
#fields = root.find(f'.//{ns1}metadata/')
list_dataformat = sickle.ListMetadataFormats()
#print([x for x in list_dataformat])
list_of_records=sickle.ListRecords(metadataPrefix='jats')

for record in tqdm(list_of_records):
    #print(record)
    
    recordraw=record.raw
    with open('biblioteka_nauki.txt', 'a',encoding="utf-8") as f:
        f.write(recordraw)
        f.write('\n\n')

        
    
    
    root=ET.fromstring(recordraw)
    soup=BeautifulSoup(recordraw, 'xml')
    result=soup.find('journal-title')
    fields = root.find(f'.//{ns1}metadata/')
    
    print(root.tag)
    for child in root:
        for c in child:
            #print(c.text)
            for a in c:
                print(a)
        fields = child.iterfind('journal-title')
        for f in fields:
            print(f)
    "http://www.w3.org/2001/XMLSchema-instance"
    fields = root.find('http://www.openarchives.org/OAI/2.0/')
    print(fields)






list_record=sickle.GetRecord(metadataPrefix='oai_dc', identifier='oai:bibliotekanauki.pl:202060')

for record in list_record:
    print(record[0],':',record[1])
    lista=record[1]
    #print(ident.tag)
    recordraw=record.raw
    print(recordraw)
    root=ET.fromstring(recordraw)
    #print(root)
    #root1=root.findall(f'.//{ns1}')

#for ident in list_identifires:
    #print(ident.tag)
#    recordraw=ident.raw
    
 #   root=ET.fromstring(recordraw)
  #  root1=root.findall(f'.//{ns1}')
    
   # for child in root:
       #print(child.tag, child.text)
    #    prefix, has_namespace, postfix = child.tag.partition('}')
     #   print(prefix, has_namespace, postfix)
        

    
    #print(root.tag)

def biblioteka_nauki_to_list(path):
    records = []
    with open(path, 'r', encoding = 'utf-8') as mrk:
        record = []
        for line in mrk.readlines():
            if line == '\n':
                pass
            elif line.startswith('<record xmlns="http://www.openarchives.org/OAI/2.0/') and record: 
                records.append(record)
                record = []
                record.append(line)
            else:
                record.append(line)
        records.append(record)       
    return records
    #print(recordraw)

x=biblioteka_nauki_to_list('C:/Users/dariu/biblioteka_nauki/biblioteka_nauki.txt')