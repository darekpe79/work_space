from sickle import Sickle
from pymarc import MARCReader
from pymarc import exceptions as exc
from pymarc import parse_xml_to_array
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os

#folderpath = r"F:\Nowa_praca\fennica\arto" 
#os.listdir(folderpath)
#filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
all_files =[]
all_files1=[]
#for path in [r"C:\Users\darek\journal_fi.txt"]:
with open(r"C:\Users\darek\journal_fi.txt", 'r', encoding='utf-8') as f:
    file = f.readlines()
    all_files.append(file)
    #all_files1.extend(file)
#with open(r"F:\Nowa_praca\fennica\60000.txt", 'r', encoding='utf-8') as f:
#   data = f.readlines()
alldata=[]
for data in all_files:
        

        records = line.split('❦')
        alldata+=records
        #alldata.extend(records)

        listarekordow=[]
        for record in alldata[1900]:
            
            tree=ET.ElementTree(ET.fromstring(record))
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
            recordmarclist=[]
            for field in fields:
                
               # print(field)
                if field.tag == f'{ns2}leader':
                    #print('LDR')
                    #print(field.text)
                    #print('-------------')
                    LDR='=LDR  '+field.text
                    recordmarclist.append(LDR)
                elif field.tag == f'{ns2}controlfield':
                    # print(field.attrib['tag'])   
                    # print(field.text)
                    # print('-------------')
                    controlfield=f"={field.attrib['tag']}  "+field.text
                    recordmarclist.append(controlfield)
                elif field.tag == f'{ns2}datafield':
                   # print(field.attrib['tag'], field.attrib['ind1'], field.attrib['ind2'])
                    if field.attrib['ind1']==' ':
                        ind1="\\"
                    else:
                        ind1=field.attrib['ind1']
                    if field.attrib['ind2']==' ':
                        ind2="\\"
                    else:
                        ind2=field.attrib['ind2']
                    
                    datafield=f"={field.attrib['tag']}  {ind1}{ind2}"
                    for subfield in field:
                    #    print(subfield.attrib['code'], subfield.text) 
                        datafield=datafield+f"${subfield.attrib['code']}{subfield.text}"
                    recordmarclist.append(datafield)

            listarekordow.append(recordmarclist)
#%%
file1 = open("rekordyarto11.mrk", "w", encoding='utf-8')
for record in listarekordow:
    for line in record:
        file1.writelines(line+'\n')
    file1.writelines('\n')
            
file1.close()

lista=[1,2,3,4,5,6,7,8,9]
print(lista[3])
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            

