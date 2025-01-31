# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:34:02 2025

@author: darek
"""
#BN update
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
from pymarc import MARCReader, MARCWriter
#%% leader
input_file = 'C:/Users/darek/Downloads/libri_marc_bn_articles_2024-12-09.mrc'   # plik MARC z błędnym leaderem
output_file = 'C:/Users/darek/Downloads/libri_marc_bn_articles_2024-01-29.mrc'  # plik wyjściowy z poprawionym leaderem

with open(input_file, 'rb') as fh_in, open(output_file, 'wb') as fh_out:
    reader = MARCReader(fh_in)
    writer = MARCWriter(fh_out)
    
    for record in reader:
        # Sprawdź, czy w 7 pozycji leadera (indeks 7) jest 'a'
        if record.leader[7] == 'a':
            # Nadpisujemy tylko siódmą pozycję w leaderze
            record.leader = record.leader[:7] + 'b' + record.leader[8:]
        
        # Zapisujemy rekord z poprawionym leaderem
        writer.write(record)
    
    writer.close()




#viaf_combination
#tworzenie slowniczka
fields_to_check={}
my_marc_files =["D:/Nowa_praca/marki_po_updatach 2025,2024/pbl_books_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/17_12_2024_espana_380_1_650gn_995new_viafd_unify2.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/bn_articles_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/bn_books_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/bn_chapters_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/cz_articles0_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/cz_articles1_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/cz_articles2_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/cz_articles3_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/cz_articles4_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/cz_books__08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/cz_chapters__08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/es_articles__08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/es_ksiazki__08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/fennica_update_do_wyslania.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/fi_arto__08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/fi_fennica_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/NEW-marc_bn_articles_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/NEW-marc_bn_books_08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/NEW-marc_bn_chapters__08-02-2024.mrc",
"D:/Nowa_praca/marki_po_updatach 2025,2024/pbl_articles_08-02-2024.mrc"]
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
my_marc_files =["D:/Nowa_praca/21082023_nowe marki nowy viaf/sp_ksiazki_composed_unify2_do_wyslanianew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_articles_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_books_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/bn_chapters_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles0_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles1_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles2_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles3_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_articles4_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_books_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/cz_chapters_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_arto_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/fi_fennica_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_articles_21-02-2023composenew_viafnew_viaf.mrc",
"D:/Nowa_praca/21082023_nowe marki nowy viaf/pbl_books_21-02-2023composenew_viafnew_viaf.mrc"]
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


#%% 655 wzbogacenie
my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_chapters_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_books_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/libri_marc_bn_articles_2023-08-07new_viaf.mrc"]

field650=pd.read_excel('D:/Nowa_praca/update_fennica/Major_genre_wszystko.xlsx', sheet_name='655_BN',dtype=str)
listy=dict(zip(field650['desk655'].to_list(),field650['action2'].to_list()))
dictionary_to_check={}

def unique(list1):
  
    
    unique_list = []
      
    
    for x in list1:
        
        if x not in unique_list:
            unique_list.append(x)
    list1[:]=unique_list     
    
    
for k,v in listy.items():
    #print(v)
    if type(v)!=float:
        dictionary_to_check[k]=v

for my_marc_file in tqdm(my_marc_files):
    writer = TextWriter(open(my_marc_file+'genre_655.mrk','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data, open(my_marc_file+'genre_655.mrc','wb')as data1:
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
                                               Subfield('i', 'Major genre'),
                                                Subfield('a', 'Secondary literature'),
                                                Subfield('l', 'eng'),
                                            ]
                                ) 

                        record.add_ordered_field(my_new_380_field2) 
                        
                    else:
                        
                        my_new_380_field = Field(
            
                                tag = '380', 
            
                                indicators = ['\\','\\'],
            
                                subfields = [
                                                Subfield('i', 'Major genre'),
                                                Subfield('a', 'Literature'),
                                               Subfield( 'l', 'eng'),
                                            ]
                                ) 

                        
                        
                        my_new_245_field = Field(
                            tag='381',
                            indicators=['\\', '\\'],
                            subfields=[
                                Subfield(code='i', value='Major genre'),
                                Subfield(code='a', value=new),
                                Subfield(code='l', value='eng')
                            ]
                        )

                        record.add_ordered_field(my_new_245_field)
                if my_new_380_field:        
                    record.add_ordered_field(my_new_380_field)        

### adding the new field
            
            # record.add_ordered_field(my_new_245_field)
            # record['380']['a'] = 'The Zombie Programmer '
            # print(record['380'])
            data1.write(record.as_marc())
            writer.write(record)    
writer.close() 

#%%650 N G

from pymarc import MARCReader, Field, TextWriter, MARCWriter
import pandas as pd
import re
from pymarc import MARCReader, TextWriter, Field, Subfield,MARCWriter
# Wczytanie Excela
excel_path = "D:/Nowa_praca/update_fennica/all_650_new_karolina.xlsx"
              
arkusz1 = pd.read_excel(excel_path, sheet_name="bn_do_laczenia")
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
merged['desk_650_normalized'] = merged['desk_650'].apply(lambda x: clean_text(extract_subfield_a(x)))




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


#%%correct field "d"  
my_marc_files = ["C:/Users/darek/fennica_650_380_381new_viaf.mrc"]
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
#%%995
from pymarc import MARCReader, MARCWriter, Field, Subfield, Record
import string
def shift_subfield_code(code: str) -> str:
    """
    Funkcja przesuwa literowy kod subpola o 1 pozycję w alfabecie: a->b, b->c, ..., y->z.
    Cyfry i inne znaki (np. '5', '0') pozostają bez zmian.
    Zostawiamy 'z' jako 'z' (możesz ewentualnie zaimplementować zawijanie).
    """
    # Jeżeli kod jest pojedynczą literą (a-z):
    if len(code) == 1 and code.isalpha():
        # Znajdź indeks w alfabecie
        idx = string.ascii_lowercase.find(code.lower())
        if idx == -1:
            # Nie znaleziono w a-z
            return code
        
        # Jeżeli to 'z', zostawiamy jako 'z' (lub zawijaj do 'a' – wg potrzeb)
        if code.lower() == 'z':
            return 'z'
        
        # Przesunięcie o 1
        new_char = string.ascii_lowercase[idx + 1]
        
        # Jeśli kod był wielką literą (raczej w MARC subfields się nie spotyka), to podtrzymujemy
        if code.isupper():
            new_char = new_char.upper()
        return new_char
    else:
        # Jeśli to cyfra lub inny znak, nie zmieniamy
        return code

def modify_or_add_995(file_path, output_path):
    with open(file_path, 'rb') as marc_file, open(output_path, 'wb') as out_file:
        reader = MARCReader(marc_file, to_unicode=True, force_utf8=True)
        writer = MARCWriter(out_file)
        
        for record in reader:
            fields_995 = record.get_fields('995')
            
            if not fields_995:
                #
                # 1. Rekord NIE ma pola 995
                #    -> tworzymy nowe pole 995 i dodajemy subfield 'a' = 'Kansalliskirjasto'
                #
                new_995 = Field(
                    tag='995',
                    indicators=[' ', ' '],
                    subfields=[Subfield(code='a', value='Kansalliskirjasto')]
                )
                record.add_field(new_995)
            else:
                #
                # 2. Rekord ma przynajmniej jedno pole 995
                #    -> pracujemy z PIERWSZYM polem 995 (jeśli trzeba obsłużyć wszystkie – pętla)
                #
                field_995 = fields_995[0]
                subfields_list = field_995.subfields  # to jest lista obiektów Subfield od pymarc 4.x

                # Sprawdzamy, czy istnieje subfield a z wartością "Kansalliskirjasto"
                sub_a_exists = any(
                    (sf.code == 'a' and sf.value == 'Kansalliskirjasto') 
                    for sf in subfields_list
                )

                if not sub_a_exists:
                    #
                    # 2a. Nie ma subfield a="Kansalliskirjasto"
                    #     -> przesuwamy literowe kody subpól, dodajemy nowe subfield 'a'
                    #
                    new_subfields = []
                    
                    # Przesuwamy literowe kody
                    for sf in subfields_list:
                        shifted_code = shift_subfield_code(sf.code)
                        new_subfields.append(Subfield(code=shifted_code, value=sf.value))
                    
                    # Wstawiamy *na początek* nowy subfield a="Kansalliskirjasto"
                    new_subfields.insert(0, Subfield(code='a', value='Kansalliskirjasto'))
                    
                    # Tworzymy nowe pole 995 z przesuniętymi kodami + nowym subfieldem 'a'
                    new_field_995 = Field(
                        tag='995',
                        indicators=field_995.indicators,  # zachowaj oryginalne wskaźniki
                        subfields=new_subfields
                    )
                    
                    # Usuwamy stare pole 995 i dodajemy nowe
                    record.remove_field(field_995)
                    record.add_field(new_field_995)
            
            # Zapisujemy zmodyfikowany rekord
            writer.write(record)
        
        writer.close()

# Przykład wywołania
if __name__ == "__main__":
    input_path = 'C:/Users/darek/fennica_650_380_381new_viafd_unify2.mrc'
    output_path = 'C:/Users/darek/fennica_650_380_381new_viafd_unify+995.mrc'
    modify_or_add_995(input_path, output_path)
    
    
    
#%%
#add issns by title
fields_to_check={}
my_marc_files = ["D:/Nowa_praca/marki_compose_19.05.2023/arto_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_articles_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_chapters_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles0_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles1_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles2_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles3_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles4_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_chapters_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/es_articles_sorted_31.05.2023.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/fennica_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_articles_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/sp_ksiazki_composed_unify2_do_wyslanianew_viaf.mrc"]
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('773')
            if len(my)==1:
            
            
                for field in my:
                   # field.add_subfield('d', '1989-2022')
                    sub_t=field.get_subfields('t')
                    sub_x=field.get_subfields('x')
                    
                    if sub_x and sub_t:
                        fields_to_check[sub_t[0]]=sub_x[0]


my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_chapters_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_libri_marc_bn_articles_2023-08-07new_viaf.mrc"]
counter=0
for my_marc_file in tqdm(my_marc_files):
    #filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(my_marc_file+'+773x.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(my_marc_file+'+773x.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('773')
        
        
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_t=field.get_subfields('t')
                
                sub_x=field.get_subfields('x')
                if sub_x:
                    
                    continue
                else:
                    #print(field)
                    
                    if sub_t:
                        if sub_t[0] in fields_to_check:
                            
                            
                    
                            field.add_subfield('x', fields_to_check[sub_t[0]])
            #print(record)
            data1.write(record.as_marc())
            writer.write(record)    
writer.close()   


#%%773 records s
fields_to_check={}
my_marc_files = ["D:/Nowa_praca/marki_compose_19.05.2023/arto_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_articles_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/bn_chapters_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles0_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles1_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles2_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles3_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_articles4_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/cz_chapters_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/es_articles_sorted_31.05.2023.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/fennica_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_articles_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/pbl_books_21-02-2023compose.mrc",
"D:/Nowa_praca/marki_compose_19.05.2023/sp_ksiazki_composed_unify2_do_wyslanianew_viaf.mrc"]
for my_marc_file in tqdm(my_marc_files):
   # writer = TextWriter(open('artykuly_hiszpania_do_wyslania.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb') as data:
        reader = MARCReader(data)
       # fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('773')
            if len(my)==1:
            
            
                for field in my:
                   # field.add_subfield('d', '1989-2022')
                    sub_x=field.get_subfields('x')
                    sub_s=field.get_subfields('s')
                    
                    if sub_x and sub_s:
                        fields_to_check[sub_x[0]]=sub_s[0]
                            
                        
my_marc_files = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_chapters_2023-08-07new_viaf.mrc+773x.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_libri_marc_bn_articles_2023-08-07new_viaf.mrc+773x.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf.mrc+773x.mrc"]
counter=0
for my_marc_file in tqdm(my_marc_files):
    #filename=my_marc_file.split('/')[-1].split('.')[0]
    writer = TextWriter(open(my_marc_file+'+773s.mrk','wt',encoding="utf-8"))
    
    with open(my_marc_file, 'rb')as data, open(my_marc_file+'+773s.mrc','wb')as data1:
        reader = MARCReader(data)
        #fields_to_check={}
        for record in tqdm(reader):
            my = record.get_fields('773')
        
        
            for field in my:
               # field.add_subfield('d', '1989-2022')
                sub_x=field.get_subfields('x')
                
                sub_s=field.get_subfields('s')
                if sub_s:
                    counter+=1
                    continue
                else:
                    #print(field)
                    
                    if sub_x:
                        if sub_x[0] in fields_to_check:
                            
                            
                    
                            field.add_subfield('s', fields_to_check[sub_x[0]])
            #print(record)
            data1.write(record.as_marc())
            writer.write(record)    
writer.close()   


#%%
#move 773 x in old files to proper place    
from pymarc import MARCReader, MARCWriter, Field, Subfield

def move_x_after_t_in_773(record):
    field_773 = record.get('773', None)

    # If the record doesn't have a 773 field, return as is
    if not field_773:
        return record

    x_subfield = next((subfield for subfield in field_773.subfields if subfield.code == 'x'), None)
    t_subfield = next((subfield for subfield in field_773.subfields if subfield.code == 't'), None)

    # If either subfield x or t doesn't exist, return the record as is
    if not x_subfield or not t_subfield:
        return record

    # Remove the $x subfield
    field_773.subfields.remove(x_subfield)

    # Find the position of the $t subfield and insert the $x subfield after it
    t_position = field_773.subfields.index(t_subfield)
    field_773.subfields.insert(t_position + 1, x_subfield)

    return record

# Function to process a MARC file
def process_marc_file(input_path, output_path):
    with open(input_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
        reader = MARCReader(input_file)
        writer = MARCWriter(output_file)

        for record in reader:
            modified_record = move_x_after_t_in_773(record)
            writer.write(modified_record)

        writer.close()
def process_marc_files(file_list):
    for file_path in file_list:
        # For demonstration purposes, I'm assuming you want to overwrite each file with its processed data.
        # If you'd like a different output naming convention, adjust the output_path accordingly.
        output_path = file_path.replace('.mrc', '_processed.mrc')
        process_marc_file(file_path, output_path)

# Call the function specifying the input and output file paths
process_marc_file('C:/Users/dariu/773_proba.mrc', 'path_to_output.mrc')
files_to_process = ["D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_chapters_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_llibri_marc_bn_books_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc",
"D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/11082023_995viaf_655_650_773_710_libri_marc_bn_articles_2023-08-07new_viaf.mrc+773x.mrc+773s.mrc"]
process_marc_files(files_to_process)
