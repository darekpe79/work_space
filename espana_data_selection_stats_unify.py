# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:43:10 2023

@author: dariu
"""
from pymarc import MARCReader,JSONReader
from tqdm import tqdm
from pymarc import Record, Field, Subfield
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
               
### Słowniki
words650=set()
words655=set()
words650_dict={}
with open('D:/Nowa_praca/Espana/espana_viaf_all.mrc', 'rb') as data:
    reader = MARCReader(data)

    for record in tqdm(reader):

            
                my = record.get_fields('650')
                for field in my:
                    subfields=field.get_subfields('a')
                    for subfield in subfields:
                        if subfield in words650_dict:
                            words650_dict[subfield]+=1
                        else:
                            words650_dict[subfield]=1
                       #words650.add(subfield)
                       #words655.add(subfield)
                       
                        

from rdflib import Graph, plugin
from rdflib.serializer import Serializer
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
from rdflib import Dataset
from rdflib import URIRef
from rdflib import Literal

g = Graph()
g.parse("D:/Nowa_praca/Espana/LEM/LEM.rdf")


len(g)
v = g.serialize(format="json-ld")
y = json.loads(v)
#subject predicate object
words={} 
for word in tqdm(words655):
    objects=Literal(word, lang='es')
   # subject = URIRef("http://id.sgcb.mcu.es/Autoridades/LEM201014730/concept")
   # predicate=URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
    
    close_matches=[]
    loc_library=[]
    for sub, pred, obj in g.triples((None, None, objects)):  
        for s,p,o in g.triples(((sub, SKOS.closeMatch, None))):
            my_close_matches=str(o)
            if my_close_matches:#.startswith("http://id.loc.gov"):
                close_matches.append(my_close_matches)
            if my_close_matches.startswith("http://id.loc.gov"):
                print(my_close_matches)
                
                response=requests.get(my_close_matches.replace('#concept','.json')).json()
                for resp in response:
                    #print(resp['@id'])
                    
                    if resp['@id'].replace('/subjects','')==my_close_matches.replace('#concept',''):
                        #print(resp['http://www.loc.gov/mads/rdf/v1#authoritativeLabel'])
                        
                        authoritativeLabel=resp.get('http://www.loc.gov/mads/rdf/v1#authoritativeLabel')
                    elif resp['@id'].replace('/childrensSubjects','')==my_close_matches.replace('#concept',''):
                        #print(resp['http://www.loc.gov/mads/rdf/v1#authoritativeLabel'])
                        
                        authoritativeLabel=resp.get('http://www.loc.gov/mads/rdf/v1#authoritativeLabel')    
                        # for labels in authoritativeLabel:
                        #     print(labels)
                        
                    if authoritativeLabel:
                        loc_library.extend(authoritativeLabel) 
                        authoritativeLabel=[]
    if loc_library:
        #close_matches.append(loc_library)
        close_matches.insert(0,loc_library)
    if close_matches:
        words[str(obj)]=close_matches

words_6xx=pd.DataFrame.from_dict(words650_dict, orient='index')
words_6xx.to_excel("words_650_stats.xlsx")   
with open ('words_655_EN.json', 'w', encoding='utf-8') as file:
    json.dump(words,file,ensure_ascii=False)   
    
#%% Translate
from concurrent.futures import ThreadPoolExecutor
import translators as ts
genre = pd.read_excel ('D:/Nowa_praca/Espana/650,655 staystyki_english_etc/words_650_stats.xlsx', sheet_name='Arkusz1')
list650=genre['field_650'].to_list()
def  translate_my_friend3 (k):
    
        results={}
        results[k]=[]
        translated_en=ts.translate_text(k, translator='google', from_language='es', to_language='en')
        results[k].append(translated_en)

        return results
list_without_nan = [x for x in list650 if type(x) is not float]   
with ThreadPoolExecutor(1) as executor:
 
    results=list(tqdm(executor.map(translate_my_friend3,list_without_nan),total=len(list_without_nan)))

output={}
for li in results:
    for k,v in li.items():
        output[k]=v
        
        
from translate import Translator
results={}
for k in tqdm(list_without_nan):
    try:
        # translator= Translator(from_lang="es",to_lang="en")
        # translated_en = translator.translate(k)

        translated_en=ts.translate_text(k, translator= 'google', from_language= 'es', to_language= 'en')
        results[k]=translated_en
    except:
        try:
            translated_en=ts.translate_text(k, translator= 'alibaba', from_language= 'es', to_language= 'en')
            results[k]=translated_en
        except KeyboardInterrupt:
             break
        except:
            pass
        
    # except KeyboardInterrupt:
    #     break
    
    # except:
    #     pass

words_6xx=pd.DataFrame.from_dict(results, orient='index')
words_6xx.to_excel("przetlumaczone5_650.xlsx") 
#%% Lematyzacja i porównania
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import  regex as re
to_compare = pd.read_excel ("D:/Nowa_praca/Espana/650,655 staystyki_english_etc/words_650_stats.xlsx", sheet_name='Sheet1')
pl_l=to_compare['google_translate_loc']
#fin_l=to_compare['LCSH_BN']
list_without_nan_fi = [x for x in pl_l if not isinstance(x, float)] 
genre_nationality=pd.read_excel('D:/Nowa_praca/650_dokumenty/genre,nationality.xlsx', sheet_name='genre')
genre=genre_nationality['Genre']
lemmatizer = WordNetLemmatizer()
zlematyzowane={}
lemmatize650=[]
output={}
for g in tqdm(genre):
    words = word_tokenize(g)
    lemmat=[]
    for w in words:
        w=w.casefold().strip()
        
        
        lemma1=lemmatizer.lemmatize(w)
        #print(lemma1)
        lemmat.append(lemma1)
    
    lemmatized=' '.join(lemmat)
    
    
    
    for word in list_without_nan_fi:
        words2 = word_tokenize(word)
        lemmat2=[]
        for w2 in words2:
            
        
            word2=w2.casefold().strip()
            lemma2=lemmatizer.lemmatize(word2)
            #print(lemma2)
            lemmat2.append(lemma2)
        lemmatized2=' '.join(lemmat2)
        lemmatize650.append(lemmatized2)
       
        if re.search(rf"(?<= |^|-|\(){lemmatized}(?= |$|\))", lemmatized2, re.IGNORECASE):
     
            output[word]=[lemmatized2,lemmatized]
            zlematyzowane[lemmatized]=lemmatized2
            
excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("genre_in_650_lemmatized_Espana.xlsx", sheet_name='es') 
#%% wzbogacenie rekordów
#655
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
from copy import deepcopy
from definicje import *
my_marc_files = ["D:/Nowa_praca/Espana/espana_viaf_all.mrc"]

field650=pd.read_excel('D:/Nowa_praca/Espana/650,655 staystyki_english_etc/words_655_stats2.xlsx', sheet_name='Sheet1',dtype=str)
listy=dict(zip(field650['field_655'].to_list(),field650['major genre'].to_list()))
dictionary_to_check={}
for k,v in listy.items():
    #print(v)
    if type(v)!=float:
        dictionary_to_check[k]=v

for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open('test.mrk','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):
            
            # [e for e in record if e.tag=='381'][-1]['a']='test2'
            
            # for field in record:
                
            #     if field.tag=='381':
                    
            #         field['a']='test'
            #         field.subfields[3]='new'
            #         field.get_subfields('a')[0]='sraka'
            #         fie
            #         for sub in field.get_subfields('a'):
            #             print(sub)
                    
            
            # print(record)
            new_field=[]
            my = record.get_fields('655')
            
            for field in my:
                subfields=field.get_subfields('a')
                for subfield in subfields:
                    if subfield in dictionary_to_check:
                       
                        new_field.append(dictionary_to_check[subfield])
            if new_field:
                unique(new_field)
                my_new_380_field=None
                for new in new_field:
                    if 'Secondary literature' in new:
                        
                        
                    
                        my_new_380_field2 = Field(
            
                                tag = '380', 
            
                                indicators = ['\\','\\'],
            
                                subfields = [
                                                'i', 'Major genre',
                                                'a', 'Secondary literature',
                                                'l', 'eng',
                                            ]
                                ) 

                        record.add_ordered_field(my_new_380_field2)
                        
                    else:
                        
                        my_new_380_field = Field(
            
                                tag = '380', 
            
                                indicators = ['\\','\\'],
            
                                subfields = [
                                                'i', 'Major genre',
                                                'a', 'Literature',
                                                'l', 'eng',
                                            ]
                                ) 

                        
                        
                        my_new_245_field = Field(
            
                                tag = '381', 
            
                                indicators = ['\\','\\'],
            
                                subfields = [
                                                'i', 'Major genre',
                                                'a', new,
                                                'l', 'eng',
                                            ]
                                ) 

                        record.add_ordered_field(my_new_245_field)
                record.add_ordered_field(my_new_380_field)        

### adding the new field
            
            # record.add_ordered_field(my_new_245_field)
            # record['380']['a'] = 'The Zombie Programmer '
            # print(record['380'])
            writer.write(record)    
writer.close()

#650

field650=pd.read_excel('D:/Nowa_praca/Espana/650,655 staystyki_english_etc/words_650_stats.xlsx', sheet_name='Sheet1',dtype=str)
listy=dict(zip(field650['field_650'].to_list(),field650['genre_to_work'].to_list()))
dictionary_to_check={}
for k,v in listy.items():
    #print(v)
    if type(v)!=float:
        dictionary_to_check[k]=v
        
my_marc_files = ["C:/Users/dariu/major_nATION_viaf.mrc"]
for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open('test_nation_major_genre2.mrk','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):
            print(record)
            
            # [e for e in record if e.tag=='381'][-1]['a']='test2'
            
            # for field in record:
                
            #     if field.tag=='381':
                    
            #         field['a']='test'
            #         field.subfields[3]='new'
            #         field.get_subfields('a')[0]='sraka'
            #         fie
            #         for sub in field.get_subfields('a'):
            #             print(sub)
                    
            
            # print(record)
            new_field=[]
            my = record.get_fields('650')
            
            for field in my:
                subfields=field.get_subfields('a')
                for subfield in subfields:
                    if subfield in dictionary_to_check:
                        if dictionary_to_check[subfield]=="Puerto Rican literature":
                            new_field.append(dictionary_to_check[subfield])
                            
                        else:    
                            new_field.append(dictionary_to_check[subfield].capitalize())
            if new_field:
                unique(new_field)
                for new in new_field:
                    
                        
                        
                    
                        my_new_245_field = Field(
            
                                tag = '650', 
            
                                indicators = [' ',' '],
            
                                subfields = [
                                                
                                                'a', new,
                                                '2', 'ELB-g'
                                            ])
                                        
                                            
                                

                        record.add_ordered_field(my_new_245_field)
                        
                  
                        

### adding the new field
            
            # record.add_ordered_field(my_new_245_field)
            # record['380']['a'] = 'The Zombie Programmer '
            # print(record['380'])
            writer.write(record)    
writer.close()

#naprawiam duplikaty w 380
my_marc_files = ["C:/Users/dariu/major_nATION_viaf_g.mrc"]
for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open('test_nation_major_genre2.mrk','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):

            new_field=[]
            my = record.get_fields('380')
            new_380=[]
            for field in my:
                new_380.append(field.subfields)
            for field in my:
                record.remove_field(field)   
            if new_380:
                unique(new_380)
                for new in new_380:
                    
                        
                        
                    
                        my_new_245_field = Field(
            
                                tag = '380', 
            
                                indicators = [' ',' '],
            
                                subfields = new)
                                        
                                            
                                

                        record.add_ordered_field(my_new_245_field)
                        
                  
                        

### adding the new field
            
            # record.add_ordered_field(my_new_245_field)
            # record['380']['a'] = 'The Zombie Programmer '
            # print(record['380'])
            writer.write(record)    
writer.close()
my_marc_files = ["C:/Users/dariu/articles.mrc"]
for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):

                
                
                
                
                # field=field['a']+'ddd'
                # field.add_subfield('a', 'daro')
                # field.subfields=[Subfield('a', 'Biblioteca Nacional de España')]
                # for i, s in enumarate(field.subfields):
                #     print(i)
                #     if s.code=='a':
                #         print (s)
                #         s.value=s.value+'dupa'
                
                # field.subfields[0]=Subfield('a', 'Biblioteca Nacional de España') 
             
            print(record)
            record.remove_field(record['003'])
            # for field in record:
            #     if field.tag=='003':
            #         print(field)
            #         field.data='ES-LoD'
          

            my_003=Field(tag='003', data='ES-LoD')

            print(record)
            record.add_ordered_field(my_003)
            my_new_995_field = Field(

                        tag = '995', 

                        indicators = [' ',' '],

                        subfields = [Subfield('a', 'Biblioteca Nacional de España'),])
                        
                        
            record.add_ordered_field(my_new_995_field)            
            writer.write(record)    
writer.close()
L = [1, "term1", 3, "term2", 4, "term3", 5, "termN"]
it = iter(L)
x=list(zip(it, it))
print(x)

record = Record()
record.add_field(
    Field(
        tag='245',
        indicators=['0', '1'],
        subfields=[
            Subfield(code='a', value='The pragmatic programmer : '),
            Subfield(code='b', value='from journeyman to master /'),
            Subfield(code='c', value='Andrew Hunt, David Thomas.')
        ]))


#%%
#viaf_combination
#tworzenie slowniczka
fields_to_check={}
my_marc_files = ["D:/Nowa_praca/21082023_nowe marki nowy viaf/sp_ksiazki_composed_unify2_do_wyslanianew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_articles_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_books_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_chapters_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/BOSLIT_dataset.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles0_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles1_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles2_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles3_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles4_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_books_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_chapters_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/es_articles_sorted_31.05.2023_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_arto_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_fennica_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/NEW-marc_bn_articles_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/NEW-marc_bn_books_2023-08-07_processednew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/NEW-marc_bn_chapters_2023-08-07_processednew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_articles_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_books_21-02-2023composenew_viafnew_viaf_processed.mrc"]
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('700','600','100')
            
            
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_a=field.get_subfields('a')
                sub_d=field.get_subfields('d')
                sub_1=field.get_subfields('1')
                if sub_a and sub_d and sub_1:
                    text=''
                    for sub in sub_a+sub_d:
                        
                        for l in sub:
                            
                            if l.isalnum():
                                text=text+l
                    fields_to_check[text]=sub_1[0]
#uzycie slowniczka i dodanie viafow oczywistych                                
my_marc_files =["D:/Nowa_praca/21082023_nowe marki nowy viaf/sp_ksiazki_composed_unify2_do_wyslanianew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_articles_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_books_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_chapters_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/BOSLIT_dataset.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles0_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles1_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles2_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles3_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles4_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_books_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_chapters_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/es_articles_sorted_31.05.2023_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_arto_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_fennica_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/NEW-marc_bn_articles_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/NEW-marc_bn_books_2023-08-07_processednew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/NEW-marc_bn_chapters_2023-08-07_processednew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_articles_21-02-2023composenew_viafnew_viaf_processed.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_books_21-02-2023composenew_viafnew_viaf_processed.mrc"]
counter=0
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'new_viaf.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'new_viaf.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('700','600','100')
            
            
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_a=field.get_subfields('a')
                sub_d=field.get_subfields('d')
                sub_1=field.get_subfields('1')
                if sub_1:
                    continue
                else:
                    #print(field)
                    if sub_a and sub_d:
                        text=''
                        for sub in sub_a+sub_d:
                            
                            for l in sub:
                                
                                if l.isalnum():
                                    text=text+l
                        if text in fields_to_check:
                            counter+=1
                            print(text, fields_to_check[text])
                    
                            field.add_subfield('1', fields_to_check[text])
            #print(record)
            data1.write(record.as_marc())
            writer.write(record)    
writer.close()                    
L = ['a', "term1", 'b', "term2", 'c', "term3", 'd', "termN"]
it = iter(L)
x=list(zip(it, it))
print(x)
output=[Subfield(*e) for e in x]
print(output)

txt = "Company 12"
txt2=
text=''
test=[]
for l in txt:
    if l.isalnum():
        text=text+l
        # test.extend(l)
        
test.append(text)
x = txt.isalnum()

print(x)
#%%
#compose_data pymarc4
my_marc_files = ["D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/pbl_articles_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/arto_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/pbl_books_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/fennica_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_chapters_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_books_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles4_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles3_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles2_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles1_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/cz_articles0_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/bn_chapters_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/bn_books_21-02-2023.mrc",
"D:/Nowa_praca/marki_21.02.2023/nowe_marki21-02-2023/bn_articles_21-02-2023.mrc"]
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'compose.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'compose.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            #print(record)
            
            for field in record:
                try:
                    #print(field.subfields)
                    field.subfields=compose_data(field.subfields)
                except:
                    continue
                    
            data1.write(record.as_marc())  
            writer.write(record)    
writer.close() 
            
#%%correct field "d"  
my_marc_files = ["C:/Users/dariu/BOSLIT_datasetnew_viaf.mrc"]
records_double_d=set()
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(filename+'d_unify2.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(filename+'d_unify2.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            print(record)
            
            
            my = record.get_fields('700','600','100')
            for field in my:
                #print(record['001'].value())
               # field.add_subfield('d', '1989-2022')
                #sub_a=field.get_subfields('a')
                sub_d=field.get_subfields('d')
                if len(sub_d)>1:
                    field.delete_subfield('d')
                    
                    if field['d'][0].isnumeric():
                        field['d']="("+field['d']+")"
                    #records_double_d.add(record['001'].value())
                    
                    
                    continue
                else:
                #field.delete_subfield('d')
                

                    if sub_d:
                        if field['d'][0].isnumeric():
                            field['d']="("+field['d'].strip('.')+")"
                   
            data1.write(record.as_marc()) 
            writer.write(record)    
writer.close() 
#%%           
#JSON_Try COMPOSE_naprawa kodowania

my_marc_files = ["D:/Nowa_praca/Espana/ksiazki i artykuly do wyslania_17.05.2023/ksiazki_composed_unify2_do_wyslania.mrc"]
numerki=['bimo0000648814', 'bimo0000384693','bimo0001559136']
for my_marc_file in tqdm(my_marc_files):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    #writer = TextWriter(open(filename+'d_unify2.mrk','wt',encoding="utf-8"))
    writer2 = JSONWriter(open(filename+'d_unify2.json','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb')as data, open(filename+'d_unify2.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            if record['001'].value() in numerki:
                print(record)
                record.remove_field(record['995'])
                my_new_995_field = Field(

                            tag = '995', 

                            indicators = [' ',' '],

                            subfields = [Subfield('a', 'Biblioteca Nacional de España'),])
                            
                            
                record.add_ordered_field(my_new_995_field)            

                   
            data1.write(record.as_marc()) 
            #writer.write(record)
            writer2.write(record)
writer2.close()
#writer.close() 

for my_marc_file in tqdm(['C:/Users/dariu/article_compose_espana.json']):
    
    writer = TextWriter(open('article_.mrk','wt',encoding="utf-8"))
    writer2 = JSONWriter(open('article_.json','wt',encoding="utf-8"))
    with open(my_marc_file)as data, open('article_.mrc','wb')as data1:
        reader = JSONReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            print(record)

                   
            data1.write(record.as_marc()) 
            writer.write(record)
            writer2.write(record)
writer2.close()
writer.close()

#Compose_Json
with open ('C:/Users/dariu/check.json', 'r', encoding='utf-8') as json_file:
    data_article=json.load(json_file)
    data_article_composed=compose_data(data_article)
    
with open('article_compose_espana.json', 'w', encoding='utf-8') as f:
    json.dump(data_article_composed, f)
    
    
    
#save all as json     
    
for my_marc_file in tqdm(["D:/Nowa_praca/nowe marki nowy viaf/bn_articles_21-02-2023composenew_viaf.mrc",
"D:/Nowa_praca/nowe marki nowy viaf/bn_books_21-02-2023composenew_viaf.mrc"]):
    filename=my_marc_file.split('/')[-1].split('.')[0]
    
    writer2 = JSONWriter(open(filename+'.json','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb')as data:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
           # print(record)

                   

            writer2.write(record)
    writer2.close()
writer.close()