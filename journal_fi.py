from sickle import Sickle
from pymarc import MARCReader
from pymarc import exceptions as exc
from pymarc import parse_xml_to_array
import xml.etree.ElementTree as ET
from tqdm import tqdm
sickle = Sickle('https://journal.fi/index/oai')
records = sickle.ListRecords(metadataPrefix='marcxml' )

#record=records.next()
lista=[]
count=0
count2=0
for record in tqdm(records):
    #print(record)
    count+=1
    count2+=1
    recordraw=record.raw
    lista.append(recordraw)

        

