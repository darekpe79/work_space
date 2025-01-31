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
with open('espana2.mrc','wb') as data1, open('D:/Nowa_praca/Espana/update_16_12_2024/MONOMODERN-mrc_new.mrc', 'rb') as data:
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

                
                
                
                
                data1.write(record.as_marc())
        except:
            pass
records2 = set()

# Wczytaj rekordy ze starego pliku i zbuduj zbiór 'records2'
with open('D:/Nowa_praca/08.02.2024_marki/es_ksiazki__08-02-2024.mrc', 'rb') as data:
    reader = MARCReader(data)
    for record in tqdm(reader):
        try:
            # Pobierz identyfikator z pola '001'
            my001 = record.get_fields('001')
            if my001:  # Jeśli pole '001' istnieje
                record_id = my001[0].value()
                records2.add(record_id)  # Dodaj identyfikator do zbioru
        except Exception as e:
            print(f"Error processing record in old file: {e}")
            pass

print(f"Collected {len(records2)} unique IDs from the old file.")

# Przetwarzanie nowego pliku
with open('D:/Nowa_praca/Espana/update_16_12_2024/espana.mrc', 'rb') as data, \
     open('filtered_records.mrc', 'wb') as output_mrc, \
     open('no_935_records.mrc', 'wb') as no_935_mrc:
    
    # Tworzenie writerów dla MRK
    output_mrk = TextWriter(open('filtered_records.mrk', 'wt', encoding='utf-8'))
    no_935_mrk = TextWriter(open('no_935_records.mrk', 'wt', encoding='utf-8'))
    
    reader = MARCReader(data)
    counter = 0
    filtered_counter = 0
    no_935_counter = 0

    for record in tqdm(reader):
        try:
            # Pobierz identyfikatory z pól '001' i '935'
            field_001 = record.get_fields('001')
            field_935 = record.get_fields('935')

            record_id_001 = field_001[0].value() if field_001 else None
            record_id_935 = field_935[0].value() if field_935 else None

            # Filtruj rekordy bez '935' i sprawdzaj '001'
            if not record_id_935:
                if record_id_001 and record_id_001 not in records2:
                    no_935_mrc.write(record.as_marc())  # Zapisz do osobnego pliku MARC
                    no_935_mrk.write(record)  # Zapisz do osobnego pliku MRK
                    no_935_counter += 1

            # Filtruj rekordy na podstawie '935' i 'records2'
            if record_id_935 and record_id_935 not in records2:
                output_mrc.write(record.as_marc())  # Zapisz do głównego pliku MARC
                output_mrk.write(record)  # Zapisz do głównego pliku MRK
                filtered_counter += 1

            counter += 1
        except Exception as e:
            print(f"Error processing record in new file: {e}")
            pass

    # Zamknięcie writerów MRK
    output_mrk.close()
    no_935_mrk.close()

# Wyświetl podsumowanie
print(f"Processed {counter} records from the new file.")
print(f"Filtered and saved {filtered_counter} records to 'filtered_records.mrc' and 'filtered_records.mrk'.")
print(f"Records without field 935 saved to 'no_935_records.mrc' and 'no_935_records.mrk': {no_935_counter}.")

#%%Połączenie plików
from pymarc import MARCReader, TextWriter

# Pliki wejściowe i wyjściowe
filtered_file = 'filtered_records.mrc'
no_935_file = 'no_935_records.mrc'
output_mrc_file = '17_12_2024_espana.mrc'
output_mrk_file = '17_12_2024_espana.mrk'

# Otwarcie pliku wyjściowego dla MARC
output_mrc = open(output_mrc_file, 'wb')

# Tworzenie TextWriter dla MRK
output_mrk = TextWriter(open(output_mrk_file, 'wt', encoding='utf-8'))

try:
    # Przetwarzanie pliku 'filtered_records.mrc'
    with open(filtered_file, 'rb') as filtered_data:
        reader = MARCReader(filtered_data)
        for record in reader:
            output_mrc.write(record.as_marc())  # Zapisz rekord do pliku MRC
            output_mrk.write(record)  # Zapisz rekord do pliku MRK

    # Przetwarzanie pliku 'no_935_records.mrc'
    with open(no_935_file, 'rb') as no_935_data:
        reader = MARCReader(no_935_data)
        for record in reader:
            output_mrc.write(record.as_marc())  # Zapisz rekord do pliku MRC
            output_mrk.write(record)  # Zapisz rekord do pliku MRK

    print(f"Combined records saved to '{output_mrc_file}' and '{output_mrk_file}'.")

except Exception as e:
    print(f"Error during processing: {e}")

finally:
    # Zamknięcie plików
    output_mrc.close()
    output_mrk.close()

#%% PROBA
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
from pymarc import MARCReader, TextWriter, Field, Subfield,MARCWriter
from tqdm import tqdm
import requests
import json
from pymarc import parse_json_to_array
from pymarc import TextWriter
from pymarc import XMLWriter
from pymarc import JSONWriter
from io import BytesIO
import warnings
from pymarc import Record, Field 
import pandas as pd
from copy import deepcopy
from definicje import *
my_marc_files = ["D:/Nowa_praca/Espana/update_16_12_2024/records_celar_ready_to_enrichment/17_12_2024_espana.mrc"]

field650=pd.read_excel('D:/Nowa_praca/update_fennica/Major_genre_wszystko.xlsx', sheet_name='655_spain',dtype=str)
listy=dict(zip(field650['field_655'].to_list(),field650['major genre'].to_list()))
dictionary_to_check={}
for k,v in listy.items():
    #print(v)
    if type(v)!=float:
        dictionary_to_check[k]=v
output_mrc_file = '17_12_2024_espana_380_1.mrc'
mrc_writer = MARCWriter(open(output_mrc_file, 'wb'))
for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open('17_12_2024_espana_380_1.mrk','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):
            new_field = []
            my = record.get_fields('655')  # Pobierz pola '655'
            
            for field in my:
                subfields = field.get_subfields('a')  # Pobierz subfieldy 'a'
                for subfield in subfields:
                    if subfield in dictionary_to_check:
                        new_field.append(dictionary_to_check[subfield])
            
            # Dodawanie nowych pól tylko, jeśli znaleziono dopasowania
            if new_field:
                #print(new_field)
                unique(new_field)  # Usuwanie duplikatów
                my_new_380_field = None
                
                for new in new_field:
                    if 'Secondary literature' in new:
                        my_new_380_field2 = Field(
                            tag='380',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield(code='i', value='Major genre'),
                                Subfield(code='a', value='Secondary literature'),
                                Subfield(code='l', value='eng'),
                            ]
                        )
                        record.add_ordered_field(my_new_380_field2)
                    else:
                        my_new_380_field = Field(
                            tag='380',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield(code='i', value='Major genre'),
                                Subfield(code='a', value='Literature'),
                                Subfield(code='l', value='eng'),
                            ]
                        )

                        my_new_245_field = Field(
                            tag='381',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield(code='i', value='Major genre'),
                                Subfield(code='a', value=new),
                                Subfield(code='l', value='eng'),
                            ]
                        )
                        record.add_ordered_field(my_new_245_field)
                        
                if my_new_380_field:  # Dodaj pole '380' jeśli istnieje
                    record.add_ordered_field(my_new_380_field)

            # Zapis rekordu
            writer.write(record)
            mrc_writer.write(record)

# Zamknięcie writera
writer.close()
mrc_writer.close()

#%%650 N G

from pymarc import MARCReader, Field, TextWriter, MARCWriter
import pandas as pd
import re
from pymarc import MARCReader, TextWriter, Field, Subfield,MARCWriter
# Wczytanie Excela
excel_path = "D:/Nowa_praca/update_fennica/all_650_new_karolina.xlsx"
              
arkusz1 = pd.read_excel(excel_path, sheet_name="spain_do_laczenia")
arkusz2 = pd.read_excel(excel_path, sheet_name="wszystko_karolina")

# Wyciągamy tylko potrzebne kolumny
arkusz1 = arkusz1[['all', 'desk_650']]
arkusz2 = arkusz2[['all', 'KPto650', 'nationalityto650']]

# Łączenie arkuszy na podstawie kolumny "all"
merged = pd.merge(arkusz1, arkusz2, on="all", how="left")
# Funkcja do przetwarzania pola MARC na format zgodny z desk_650
def clean_text(value):
    """
    Usuwa zbędne spacje, kropki i przecinki z początku i końca tekstu.
    """
    if not value:
        return None
    return re.sub(r'[.,]+$', '', value.strip())  # Usuwa kropki/przecinki na końcu i zbędne spacje

# Funkcja do wyciągania podpola 'a'
def extract_subfield_a(value):
    """
    Wyciąga zawartość podpola 'a' z wartości MARC (desk_650 lub pole 650).
    """
    value = re.sub(r'^\\[0-9]*', '', value)  # Usunięcie wskaźnika (\7, \0)
    match = re.search(r'\$a([^$]+)', value)  # Szukanie zawartości podpola 'a'
    if match:
        return clean_text(match.group(1))  # Czyszczenie i zwracanie zawartości podpola 'a'
    return None

# Czyszczenie desk_650 w Excelu
merged['desk_650_normalized'] = merged['desk_650'].apply(lambda x: clean_text(x))





def is_duplicate_field(record, new_field):
    """
    Sprawdza, czy rekord MARC zawiera już pole identyczne z new_field.
    """
    for field in record.get_fields(new_field.tag):
        # Porównujemy wskaźniki i podpola
        if field.indicators == new_field.indicators and field.subfields == new_field.subfields:
            return True
    return False

# Funkcja przetwarzająca MARC dla danego wskaźnika i kolumny Excela
def process_marc(input_file, output_mrk, output_mrc, merge_column, indicator_value):
    """
    Przetwarza plik MARC, dodając pola 650 na podstawie podanej kolumny Excela.
    """
    with open(input_file, "rb") as marc_file:
        reader = MARCReader(marc_file)

        # Przygotowanie writerów
        with open(output_mrk, "w", encoding="utf-8") as text_output, \
             open(output_mrc, "wb") as binary_output:
            
            text_writer = TextWriter(text_output)
            mrc_writer = MARCWriter(binary_output)

            # Liczniki
            total_records = 0
            records_with_new_fields = 0

            # Iteracja przez rekordy MARC
            for record in reader:
                total_records += 1
                added_new_field = False  # Flaga dla rekordu

                for field in record.get_fields('650'):
                    # Wyciągnięcie podpola 'a'
                    subfield_a = clean_text(extract_subfield_a(str(field)))

                    if subfield_a:
                        # Dopasowanie do Excela
                        nowy_wiersz = merged.loc[merged['desk_650_normalized'] == subfield_a]

                        if not nowy_wiersz.empty:
                            # Obsługa tłumaczeń
                            new_fields = nowy_wiersz[merge_column].dropna().unique()
                            for new_value in new_fields:
                                if new_value:
                                    # Formatowanie i czyszczenie wartości
                                    formatted_value = clean_text(new_value).capitalize()

                                    # Tworzenie pola 650
                                    new_field = Field(
                                        tag='650',
                                        indicators=[' ', ' '],
                                        subfields=[
                                            Subfield('a', formatted_value),
                                            Subfield('2', indicator_value)
                                        ]
                                    )

                                    # Sprawdzanie duplikatów
                                    if not is_duplicate_field(record, new_field):
                                        record.add_ordered_field(new_field)
                                        added_new_field = True

                # Zliczanie rekordów z nowymi polami
                if added_new_field:
                    records_with_new_fields += 1

                # Zapis rekordu
                text_writer.write(record)
                mrc_writer.write(record)

            # Zamknięcie writerów
            text_writer.close()
            mrc_writer.close()

    # Wynik w konsoli
    print(f"{indicator_value}: Total records processed: {total_records}")
    print(f"{indicator_value}: Records with new fields: {records_with_new_fields}")

# Czyszczenie kolumny 'desk_650'
merged['desk_650_normalized'] = merged['desk_650'].apply(lambda x: clean_text(extract_subfield_a(x)))

# Pierwszy proces: ELB-g
process_marc(
    input_file="D:/Nowa_praca/update_fennica/uniqueFennica.mrc",
    output_mrk="wynikowy_elb_g.mrk",
    output_mrc="wynikowy_elb_g.mrc",
    merge_column="KPto650",
    indicator_value="ELB-g"
)

# Drugi proces: ELB-n (bazując na wynikowym pliku z ELB-g)
process_marc(
    input_file="wynikowy_elb_g.mrc",
    output_mrk="wynikowy_elb_n_g.mrk",
    output_mrc="wynikowy_elb_n_g.mrc",
    merge_column="nationalityto650",
    indicator_value="ELB-n")









# #650

# field650=pd.read_excel('D:/Nowa_praca/Espana/650,655 staystyki_english_etc/words_650_stats.xlsx', sheet_name='Sheet1',dtype=str)
# listy=dict(zip(field650['field_650'].to_list(),field650['genre_to_work'].to_list()))
# dictionary_to_check={}
# for k,v in listy.items():
#     #print(v)
#     if type(v)!=float:
#         dictionary_to_check[k]=v
        
# my_marc_files = ["D:/Nowa_praca/Espana/update_16_12_2024/records_380_381/17_12_2024_espana_380_1.mrc"]
# output_mrc_file = '17_12_2024_espana_380_1_650g.mrc'
# mrc_writer = MARCWriter(open(output_mrc_file, 'wb'))
# for my_marc_file in tqdm(my_marc_files):
#     writer = TextWriter(open('17_12_2024_espana_380_1_650g.mrk','wt',encoding="utf-8"))
#     with open(my_marc_file, 'rb') as data:
#         reader = MARCReader(data)
#         for record in tqdm(reader):
#             print(record)
            
#             # [e for e in record if e.tag=='381'][-1]['a']='test2'
            
#             # for field in record:
                
#             #     if field.tag=='381':
                    
#             #         field['a']='test'
#             #         field.subfields[3]='new'
#             #         field.get_subfields('a')[0]='sraka'
#             #         fie
#             #         for sub in field.get_subfields('a'):
#             #             print(sub)
                    
            
#             # print(record)
#             new_field=[]
#             my = record.get_fields('650')
            
#             for field in my:
#                 subfields=field.get_subfields('a')
#                 for subfield in subfields:
#                     if subfield in dictionary_to_check:
#                         if dictionary_to_check[subfield]=="Puerto Rican literature":
#                             new_field.append(dictionary_to_check[subfield])
                            
#                         else:    
#                             new_field.append(dictionary_to_check[subfield].capitalize())
#             if new_field:
#                 unique(new_field)
#                 for new in new_field:
                    
                        
                        
                    
#                         my_new_245_field = Field(
            
#                                 tag = '650', 
            
#                                 indicators = [' ',' '],
            
#                                 subfields = [Subfield(
                                                
#                                                 'a', new),
#                                                 Subfield('2', 'ELB-g')
#                                             ])
                                        
                                            
                                

#                         record.add_ordered_field(my_new_245_field)
                        
                  
                        

# ### adding the new field
            
#             # record.add_ordered_field(my_new_245_field)
#             # record['380']['a'] = 'The Zombie Programmer '
#             # print(record['380'])
#             writer.write(record) 
#             mrc_writer.write(record)
# writer.close()
# mrc_writer.close()

# #ELB-n adding  
# field650=pd.read_excel('D:/Nowa_praca/Espana/650,655 staystyki_english_etc/words_650_stats.xlsx', sheet_name='Sheet1',dtype=str)
# listy=dict(zip(field650['field_650'].to_list(),field650['nationality_to_work'].to_list()))
# dictionary_to_check={}
# for k,v in listy.items():
#     #print(v)
#     if type(v)!=float:
#         dictionary_to_check[k]=v
      
# my_marc_files = ["D:/Nowa_praca/Espana/update_16_12_2024/records_380_301_650ng/17_12_2024_espana_380_1_650g.mrc"]
# output_mrc_file = '17_12_2024_espana_380_1_650gn.mrc'
# mrc_writer = MARCWriter(open(output_mrc_file, 'wb'))
# for my_marc_file in tqdm(my_marc_files):
#     writer = TextWriter(open('17_12_2024_espana_380_1_650gn.mrk','wt',encoding="utf-8"))
#     with open(my_marc_file, 'rb') as data:
#         reader = MARCReader(data)
#         for record in tqdm(reader):
#             print(record)
            

#             new_field=[]
#             my = record.get_fields('650')
            
#             for field in my:
#                 subfields=field.get_subfields('a')
#                 for subfield in subfields:
#                     if subfield in dictionary_to_check:
#                         if dictionary_to_check[subfield]=="Puerto Rican literature":
#                             new_field.append(dictionary_to_check[subfield])
                            
#                         else:    
#                             new_field.append(dictionary_to_check[subfield].capitalize())
#             if new_field:
#                 unique(new_field)
#                 for new in new_field:
                    
                        
                        
                    
#                         my_new_245_field = Field(
            
#                                 tag = '650', 
            
#                                 indicators = [' ',' '],
            
#                                 subfields = [Subfield(
                                                
#                                                 'a', new),
#                                                 Subfield('2', 'ELB-n')
#                                             ])
                                        
                                            
                                

#                         record.add_ordered_field(my_new_245_field)
                        
                  
                        


#             writer.write(record) 
#             mrc_writer.write(record)
# writer.close()
# mrc_writer.close()

my_marc_files = ["D:/Nowa_praca/Espana/update_16_12_2024/records_380_301_650ng/17_12_2024_espana_380_1_650gn.mrc"]
output_mrc_file = '17_12_2024_espana_380_1_650gn_995.mrc'
mrc_writer = MARCWriter(open(output_mrc_file, 'wb'))
for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open('17_12_2024_espana_380_1_650gn_995.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
        for record in tqdm(reader):

                

            my_new_995_field = Field(

                        tag = '995', 

                        indicators = [' ',' '],

                        subfields = [Subfield('a', 'Biblioteca Nacional de España'),])
                        
                        
            record.add_ordered_field(my_new_995_field)            
            writer.write(record) 
            mrc_writer.write(record)
writer.close()
mrc_writer.close()
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
my_marc_files = ["D:/Nowa_praca/08.02.2024_marki/pbl_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_articles_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_chapters_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles0_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles1_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles2_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles3_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles4_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_books__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_chapters__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/es_articles__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/es_ksiazki__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/fi_arto__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/fi_fennica_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_articles_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_chapters__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/pbl_articles_08-02-2024.mrc"]
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
my_marc_files =["D:/Nowa_praca/Espana/update_16_12_2024/records_380_301_650ng_995/17_12_2024_espana_380_1_650gn_995.mrc"]
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
my_marc_files = ["D:/Nowa_praca/Espana/update_16_12_2024/records_380_301_650ng_995/17_12_2024_espana_380_1_650gn_995new_viaf.mrc"]
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