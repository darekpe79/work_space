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
from tqdm import tqdm
to_compare = pd.read_excel ("C:/Users/dariu/Tomasz_literaturoznawstwo_Kulczycki/new_best_27.03/orcid_prace_literaturoznawstwo.xlsx", sheet_name='tylko_z_pracami')
dicto = to_compare.to_dict('records')
new_to_work_list=[]
for rec in tqdm(dicto):
    new_to_work={}
    list_of_works=[]
    for key, val in rec.items():
        print(key,val)
        if type(val) !=float:
            if type(key) is int:
                alphanum_title=''
                for letter in val:
                    if letter.isalnum():
                        alphanum_title+=letter
                list_of_works.append(alphanum_title.lower())
            else:
                new_to_work[key]=val
    new_to_work['works']=list_of_works        
    new_to_work_list.append(new_to_work)
counter=0

# with open('nauki o sztuce_polon.json', encoding='utf-8') as fh:
#     dataname = json.load(fh)
# namesall=[]   
# for names in dataname:
#     print (names['firstName'])
#     print (names['lastName'])
#     if names['firstName']:
#         namesall.append(names['firstName']+' '+ names['lastName'])
'becomingrealtooneselfemersonthoreauhawthorne'=='becomingrealtooneselfemersonthoreauhawthorne'  
# names=pd.DataFrame(dataname) #orient='index')    
# names.to_excel("lista_nauki o sztuce.xlsx", sheet_name='Sheet_name_1') 
allrecords=[]
people_without_works=[]
people_with_works=[]
kardiologia={}
for name in tqdm(new_to_work_list):      
    name=name['without_middle']
    #print(name)
    
    url =f"https://data.bn.org.pl/api/institutions/bibs.json?author={name}&amp;limit=100"
    #print(url)
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
        records_for_one=[]
        switch=False
        for record in records:
            #print(record)
            records_for_one.append(record)
            my_500s = record.get_fields('245')
            for my in my_500s:
                #print(my.get_subfields('a', 'd', '1'))
                a= my.get_subfields('a')
                b= my.get_subfields('b')
                
                
                title=''
                for t1 in a:
                    title=title+t1
                    if b:
                        for t2 in b:
                            title=title+t2
                alphanum_title=''
                for letter in title:
                    if letter.isalnum():
                        alphanum_title+=letter
                title=alphanum_title.lower()    
                for data in new_to_work_list:
                    if data['without_middle']==name:
                        if title in data['works']:
                            switch=True
        
        
        
         
        if switch==True:
            #check for kardiologia #650
            # for record in records_for_one:
            #     my_500s = record.get_fields('650')
            #     for my in my_500s:
            #         #print(my.get_subfields('a', 'd', '1'))
            #         a= my.get_subfields('a')
            #         if "Kardiologia" in a or "Onkologia" in a:
            #             print(a)
            #             kardiologia[name+' '+ record['001'].value()]=record['245'].value()
            #             allrecords.append(record)
            #end checking        
                
            allrecords.extend(records_for_one)
            people_with_works.append(name)
        else:
            people_without_works.append(name)
            
        ### and write each record to it
names=pd.DataFrame(people_with_works)
names.to_excel("ludzie_z pracami_Orcid_Bn_dwa_iminona.xlsx", sheet_name='Sheet_name_1')  


writer = TextWriter(open('literaturoznawstwo_2_names_all_match_ORCID_BN.mrk','wt',encoding="utf-8"))
for record in allrecords:
    #allrecords.append(record)
### and write each record to it
    writer.write(record)
writer.close()       

with open('literaturoznawstwo_2_names_all_match_ORCID_BN.mrc' , 'wb') as data1:
    for my_record in allrecords:
### and write each record to it

        data1.write(my_record.as_marc())
 
#wyciąganie wybranych rekordów       
path='C:/Users/dariu/po_UKD82-821_bez_nauk_bez_hist3974.mrk'
listofphrases=[r'\7$aArtykuł publicystyczny$2DBN',
r'\7$aArtykuł z gazety$2DBN',
r'\7$aArtykuł z czasopisma społeczno-kulturalnego$2DBN',
r'\7$aArtykuł z tygodnika opinii$2DBN',
r'\'7$aPublicystyka$2DBN',
r'\7$aArtykuł z czasopisma popularnonaukowego$2DBN',
]
listofphrases2=['pieś', 'piosenk','film','muzyka','album','przewodnik turystyczny',
                'przewodnik po wystawie']


dictrec=list_of_dict_from_file(path) 
unique=[]
for x in tqdm(dictrec):
    if x not in unique:
        unique.append(x)
recordskultural=[]
rodzaj={}
count=0
for record in tqdm(dictrec):
        # if '650' not in record:
        #     recordskultural.append(record)
    switch=False
    #record.get('955')
    #if record.get('655'):
        # if any(phrase in line for phrase in listofphrases):
        #     recordskultural.append(record)
    for key,val in record.items():

            

                
        for line955 in val:
           
           #if line955 in listofphrases:
           if any(phrase in line955.lower() for phrase in listofphrases2):
               #print(val)
               count+=1
               #print(val)
               switch=True
                
    if switch:
        switch2=False
        for key,val in record.items():
            if key=='380' or key=='980':
                #print(val)
                if "\\\\$aPublikacje naukowe" in val or ['\\\\$aArtykuły']==val or ['\\\\$aKsiążki']==val :
                    print(val)
                else:
                    switch2=True
        if switch2:
            recordskultural.append(record)                
                
                
                
                    # if line955 in rodzaj:
                    #     rodzaj[line955]+=1
                    # else:
                    #     rodzaj[line955]=1
                    
            #print(line)
           
                
    
to_file2('UKD_literaturoznawstwo_2682.mrk',recordskultural)


#statystyki
rodzaj={}
for record in tqdm(literatura_ksiazki):
    switch=False
    #record.get('955')
    if record.get('655'):
        # if any(phrase in line for phrase in listofphrases):
        #     recordskultural.append(record)
        for key,val in record.items():
            
            if key=='655':
                
                 for line955 in val:
                    
                    #if line955 in listofphrases:
                        if line955 in rodzaj:
                            rodzaj[line955]+=1
                        else:
                            rodzaj[line955]=1
                
names=pd.DataFrame.from_dict(rodzaj, orient='index')
names.to_excel('655_UKD.xlsx')  
                

#sprawdzenie autorów

to_compare = pd.read_excel ("C:/Users/dariu/Tomasz_literaturoznawstwo_Kulczycki/new_best_27.03/literaturoznawstwo_id_names_orcid.xlsx", sheet_name='Sheet_name_1')
ludzie=to_compare['lastNamePrefix'].tolist()
#ludzie_two_names=to_compare['two_names']
# ludzie_two_names_list=[]
# for czlowiek in ludzie_two_names:
#     print(czlowiek)
    
    
    
#     if type(czlowiek) !=float:
#         ludzie_two_names_list.append(czlowiek)
ludzie_lista=[]
for czlowiek in ludzie:
    czlowiek=czlowiek.split()
    czlowiek=czlowiek[1]+', '+czlowiek[0]
    ludzie_lista.append(czlowiek)
    
ludzie=set(ludzie_lista)#+ludzie_two_names_list       
#spr_ludzie=[]
do_excela=[]
proba={}
proba2={}
pattern_e_marc=r'(?<=\$e).*?(?=\$|$)'
counter_ludzie=set()
for record in tqdm(recordskultural):

    
    
    new = record.copy()
    #record.get('955')
    
        # if any(phrase in line for phrase in listofphrases):
        #     recordskultural.append(record)
    for key,val in record.items():
        new[key]=val
        if key=='700' or key=='100':
            #print(key,val)
            for line955 in val:
               # print(key)
                for czlowiek in ludzie:

                        
                    
                    
                    
                    #print(czlowiek)
                    
                    if czlowiek in line955:
                        print(czlowiek, line955)
                        x=new|{'autor':line955}
                        #new['autor']=line955
                        do_excela.append(x)
    
                        #spr_ludzie.append(record)
                        counter_ludzie.add(czlowiek)
                        
                        print(line955)
                        print(czlowiek)
                        field_e=re.findall(pattern_e_marc, line955)
                        if field_e:
                            for e in field_e:
                                if e not in proba2:
                                    proba2[e]=1
                                else:
                                    proba2[e]+=1
                                if czlowiek+' '+e not in proba:
                                
                                    proba[czlowiek+' '+e]=[czlowiek,e,1]
                                else:
                                    proba[czlowiek+' '+e][2]+=1
                        else:
                            if 'brak_funkcji' not in proba2:
                                proba2['brak_funkcji']=1
                            else:
                                proba2['brak_funkcji']+=1
                            if czlowiek+' brak_funkcji' not in proba:
                            
                                proba[czlowiek+' brak_funkcji']=[czlowiek,'brak_funkcji',1]
                            else:
                                proba[czlowiek+' brak_funkcji'][2]+=1
                            
proba2["unikatowi_autorzy"]=len(counter_ludzie)                                    
                                
                        
                        
                        
                        

                   
          
                        
data=pd.DataFrame(do_excela)
data.to_excel("UKD_literaturoznawstwo_2682.xlsx", sheet_name='Sheet_name_1')
data=pd.DataFrame.from_dict(proba, orient='index')
data.to_excel("UKD_literaturoznawstwo_ludzi_funkcje_statsy.xlsx", sheet_name='Sheet_name_1')
#to_file2('glam_735.mrk',unique)                           
#PO UKD
tylko_monografie=LDR_monography(unique)
pattern_a_marc=r'(?<=\$a).*?(?=\$|$)' 
literatura_ksiazki=[]
for record in tqdm(tylko_monografie):
    switch=False
    
    #new = record.copy()
    #record.get('955')
    
        # if any(phrase in line for phrase in listofphrases):
        #     recordskultural.append(record)
    for key,val in record.items():
        if key=='080':
            for v in val:
                field_a=re.findall(pattern_a_marc, v)
                for a in field_a:
                    if a.startswith('82-') or a.startswith('821') and '(091)' not in a:
                        switch=True
                        print(a)
    if switch:
        switch2=False
        for key,val in record.items():
            if key=='380' or key=='980':
                if "\\\\$aPublikacje naukowe" in val:
                    print(val)
                else:
                    switch2=True
        if switch2:
            literatura_ksiazki.append(record)           

unique=[]
for x in tqdm(recordskultural):
    if x not in unique:
        unique.append(x)                   
to_file2('po_UKD82-821_bez_nauk_bez_hist3974.mrk',literatura_ksiazki)   
#podmiot gatunek narodowość
d1 = today.strftime("%d-%m-%Y")
field650=pd.read_excel('D:/Nowa_praca/650_dokumenty/650__do_pracy_wszystko.xlsx', sheet_name='bn2',dtype=str)
nationality=field650['tonew650_national'].to_list()
listy=dict(zip(field650['desk_650'].to_list(),field650['tonew650_national'].to_list()))
def isNaN(num):
    return num!= num
national_list=[]
for l in nationality:
    if type(l) is not float and l!="nan":
        national_list.append(r'\7$a'+capfirst(l))
national_set=set(national_list)       
dicto={}
dictionary650=field650.to_dict('records')
for elem in dictionary650:
    
    
    if isNaN(elem['tonew650_genre']) is  False:
        if elem['desk_650'] in dicto:
            dicto[elem['desk_650']].append(elem['tonew650_genre'])
        else:
            #dicto[elem['desk_650']]=[]
            dicto[elem['desk_650']]=[elem['tonew650_genre']]
    if isNaN(elem['tonew650_national']) is False:
        if elem['desk_650'] in dicto:
            dicto[elem['desk_650']].append(elem['tonew650_national'])
        else:
            dicto[elem['desk_650']]=[elem['tonew650_national']]
            

towork={}        
for key, val in dicto.items():
    if val:
        towork[key]=val
        
def capfirst(s):
    return s[:1].upper() + s[1:]    
        
        


paths=['C:/Users/dariu/UKD_literaturoznawstwo_2682_ELB655.mrk']

pattern4='\$2ELB'

pattern_a_marc=r'(?<=\$a).*?(?=\$|$)'
#val100=[]
counter=set()
wspolne=[]
statselb={}
statselb2={}
for plik in paths:
    record=list_of_dict_from_file(plik)
    nametopath=plik.split('/')[-1].split('.')[0]+'_'
    
    for rec in tqdm(record):
   
            
        if '655' not in rec:
            continue
        else:
            for key,val in rec.items():
                if key=='655':
                    for v in val:
                         name = re.findall(pattern4, v)
                         if name:
                             if v in national_set:
                                 name_a = re.findall(pattern_a_marc, v)
                                 if name_a[0].replace("(dp)", "") in statselb2:
                                     statselb2[name_a[0].replace("(dp)", "")]+=1
                                 else:
                                     statselb2[name_a[0].replace("(dp)", "")]=1
                                 
                             else:
                                 
                                 name_a = re.findall(pattern_a_marc, v)
                                 if name_a[0].replace("(dp)", "") in statselb:
                                     statselb[name_a[0].replace("(dp)", "")]+=1
                                 else:
                                     statselb[name_a[0].replace("(dp)", "")]=1
                        #      if name_a[0] in statselb:
                        #          statselb[name_a[0]]+=1
                        #      else:
                        #          statselb[name_a[0]]=1

                        # # if len(name)>1:
                        #     print(rec['001'][0])
                        
                        if v in towork:
                            counter.add(rec['001'][0])
                            
                            
                            for v2 in towork[v]:
                                #print(r'\7$a'+v2+r'$2Libri')
                                toappend=r'\7$a'+capfirst(v2)
                                rec[key].append(toappend)
                    unique(rec[key])
                
    to_file2(nametopath+"ELB655"'.mrk',record)   
data=pd.DataFrame.from_dict(statselb2, orient='index')
data.to_excel("country_statsy.xlsx", sheet_name='Sheet_name_1')