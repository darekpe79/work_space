from sickle import Sickle
from pymarc import MARCReader
from pymarc import exceptions as exc
from pymarc import parse_xml_to_array
import xml.etree.ElementTree as ET
from tqdm import tqdm
sickle = Sickle('https://oai-pmh.api.melinda.kansalliskirjasto.fi/bib')
records = sickle.ListRecords(metadataPrefix='marc21', set='arto')

#record=records.next()
lista=[]
count=0
count2=0
for record in tqdm(records):
    count+=1
    count2+=1
    recordraw=record.raw
    lista.append(recordraw)
    if count>=20000:
        

        f=open(str(count2)+'.txt',"w",encoding='utf-8')
        s1='❦'.join(lista)
        f.write(s1)
        f.close()
        count=0
        lista=[]
            
    tree=ET.ElementTree(ET.fromstring(recordraw))
    ns1 = '{http://www.openarchives.org/OAI/2.0/}'
    ns2 = '{http://www.loc.gov/MARC21/slim}'
    root = tree.getroot()
    # fields = root.find(f'.//{ns1}metadata/')
    # fields[0].tag
    # fields[0].text
    # fields[0].attrib
    # for e in fields:
    #     print(e)
            
    #     print(e.text)
    #     print(e.tag)
    #     print(type(e.attrib['tag']))
    #     if e.attrib['tag']=='935':
    #         print(e.text)
            
    fields = root.find(f'.//{ns1}metadata/')
    for field in fields:
        
       # print(field)
        if field.tag == f'{ns2}leader':
            print('LDR')
            print(field.text)
            print('-------------')
        elif field.tag == f'{ns2}controlfield':
            print(field.attrib['tag'])   
            print(field.text)
            print('-------------')
        elif field.tag == f'{ns2}datafield':
            print(field.attrib['tag'], field.attrib['ind1'], field.attrib['ind2'])
            for subfield in field:
                print(subfield.attrib['code'], subfield.text) 
            print('-------------')
            if field.attrib['tag']=='995':
                
                for subfield in field:
                    print(subfield.text)


    
    
    
    reader = MARCReader(record, 'rb')
    print(reader)
    print(reader['650'])
    with open('response.xml', 'w') as fp:
        fp.write(record.raw)