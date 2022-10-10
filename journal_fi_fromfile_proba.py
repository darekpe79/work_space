import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
from requests import get
#folderpath = r"F:\Nowa_praca\fennica\arto" 
#os.listdir(folderpath)
#filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]

#for path in [r"C:\Users\darek\journal_fi.txt"]:
with open(r"C:\Users\darek\journal_fi.txt", 'r', encoding='utf-8') as f:
    file = f.readlines()
    
all_lines="\n".join(file).split('❦')
    #all_files1.extend(file)
#with open(r"F:\Nowa_praca\fennica\60000.txt", 'r', encoding='utf-8') as f:
#   data = f.readlines()
#alldata=[]
#for data in all_lines:
#    print(data)
    

pattern_a_marc=r'\d*(?=$)'

    #alldata.extend(records)
  
listarekordow=[]
for record in tqdm(all_lines):
    #print(record)
    
    tree=ET.ElementTree(ET.fromstring(record))
    ns1 = '{http://www.openarchives.org/OAI/2.0/}'
    ns2 = '{http://www.loc.gov/MARC21/slim}'
    root = tree.getroot()
    proba=root[0].attrib
    if proba:

        if proba['status']=='deleted':
            continue
    header = root.find(f'.//{ns1}header/')
    header=header.text
    header=re.findall(pattern_a_marc, header)
    
    #print(record)

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
    count=1
    for field in fields:
        
        #print(field)
        if field.tag == f'{ns2}leader':
            #print('LDR')
            #print(field.text)
            #print('-------------')
            
            LDR='=LDR'+field.text
            
            recordmarclist.append(LDR)
            recordmarclist.append('=001  article'+header[0])
        elif field.tag == f'{ns2}controlfield':
            # print(field.attrib['tag'])   
            # print(field.text)
            # print('-------------')
            controlfield=f"={field.attrib['tag']}  "+field.text
            if controlfield.startswith('=008'):
                    controlfield=controlfield.replace('"', '')
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
                subfield_good=subfield.text
                #print(datafield)
                #print(subfield.attrib['code'], subfield.text)
                
                    
                
                
                if datafield.startswith('=520'):
                    count+=1
                    
                    subfield_good=subfield.text.replace("\n", "")
                    #print(subfield_good)
                    #proba21.append(subfield.text.replace("\n", ""))
                    
                    
                    
                    
                    
                    
                datafield=datafield+f"${subfield.attrib['code']}{subfield_good}"
                recordmarclist.append(datafield)    
    article_no='article'+header[0]
    URL=fr'https://kansalliskirjasto.finna.fi/Record/journalfi.{article_no}#details'
    page=get(URL)
    bs=BeautifulSoup(page.content)
    #print(bs.prettify()[:100])
    #bs.title.string
    block=bs.find_all('div', {'class':"subjectLine"})
    if block:
        for b in block:
            field653='=653  00$a '+b.text.strip('\n')
            recordmarclist.append(field653)
    
            
    listarekordow.append(recordmarclist)
#%%
file1 = open("rekordy_journal_dobry.mrk", "w", encoding='utf-8')
for record in listarekordow:
    for line in record:
        file1.writelines(line+'\n')
    file1.writelines('\n')
            
file1.close()

lista={'sdfsrwefsdfwer':'lala','key':'wefwefrwefwef'}

if lista['sdfsrwefsdfwerw']:
    print(lista)
                            
                            
print(datafield)  
if datafield.startswith('=540'):
    print('ok')

l='friiii   '  
if l=='friiii   ' :
    print('ok')                        
                            
len('=540  \\')                            
                            
                            
                            
                            
                            
                            
                            
                            

