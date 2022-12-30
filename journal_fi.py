from sickle import Sickle
#from pymarc import MARCReader
#from pymarc import exceptions as exc
#from pymarc import parse_xml_to_array
import xml.etree.ElementTree as ET
from tqdm import tqdm
from bs4 import BeautifulSoup
import json
sickle = Sickle('https://journal.fi/index/oai')
records = sickle.ListRecords(metadataPrefix='marcxml' )

#record=records.next()
lista=[]

for record in tqdm(records):
    #print(record)

    recordraw=record.raw
    soup=BeautifulSoup(recordraw, 'xml')
    ident=soup.header['status']
    #ident=soup.find('header')['status']
    if 'header status="deleted"' not in recordraw:
        #soup=BeautifulSoup(recordraw, 'xml')
        #ident=soup.find('identifier').text

        lista.append(recordraw)
records = []
for line in lista:
    record = []
    if line == '\n':
        pass
    elif line.startswith('<record xmlns="http://www.openarchives.org/OAI/2.0/') and record: 
        records.append(' '.join(record))
        record = []
        record.append(line)
    else:
        record.append(line)
    records.append(' '.join(record))    
    
for record in tqdm(records):
    with open('journal_fi.txt', 'a',encoding="utf-8") as f:
     f.write(record)
     f.write('\n\n')
     
       
with open("journal_fi.json", "w",encoding='utf-8') as outfile:
    json.dump(records,outfile,ensure_ascii=False)
    
def biblioteka_nauki_to_list(path):
    records = []
    with open(path, 'r', encoding = 'utf-8') as mrk:
        record = []
        for line in mrk.readlines():
            if line == '\n':
                pass
            elif line.startswith('<record xmlns="http://www.openarchives.org/OAI/2.0/') and record: 
                records.append(' '.join(record))
                record = []
                record.append(line)
            else:
                record.append(line)
        records.append(' '.join(record))      
    return records
    #print(recordraw)

x=biblioteka_nauki_to_list('C:/Users/dariu/journal_fi.txt')
