# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:19:08 2024

@author: dariu
"""
from pymarc import parse_xml_to_array, MARCWriter, TextWriter,map_xml
import pandas as pd
from pymarc import parse_xml_to_array
from tqdm import tqdm

def parse_marcxml(file_path):
    records = []
    processed_count = 0
    matching_count = 0
    
    # Parse the MARC XML file
    marc_records = parse_xml_to_array(file_path)
    
    for record in marc_records:
        processed_count += 1
        field_072_value = None
        field_080_value = None
        field_150_value = None
        
        # Extracting value for field 072
        subfields_072 = record.get_fields('072')
        for field in subfields_072:
            subfields = field.get_subfields('9')
            for subfield in subfields:
                if subfield in ['25', '26']:
                    field_072_value = subfield
                    break
            if field_072_value:
                break  # If a matching subfield is found, exit the loop
        
        # Extracting value for field 080
        subfields_080 = record.get_fields('080')
        for field in subfields_080:
            subfields = field.get_subfields('a')
            for subfield in subfields:
                if subfield == '808' or subfield.startswith('82'):
                    field_080_value = subfield
                    break
            if field_080_value:
                break  # If a matching subfield is found, exit the loop
        
        # Extracting value for field 150
        subfields_150 = record.get_fields('150')
        for field in subfields_150:
            subfields = field.get_subfields('a')
            for subfield in subfields:
                field_150_value = subfield
                break  # Assuming we take the first match only
            if field_150_value:
                break  # If a matching subfield is found, exit the loop
        
        # Append record if any of the fields match the criteria
        if field_072_value or field_080_value:
            matching_count += 1
            records.append({
                '072': field_072_value,
                '080': field_080_value,
                '150': field_150_value
            })
    
    print(f"Total processed records: {processed_count}")
    print(f"Total matching records: {matching_count}")
    
    return pd.DataFrame(records)

# Example usage

# Example usage
file_path = 'D:/Nowa_praca/czech_works/aut_ph.xml/aut_ph.xml'
df = parse_marcxml(file_path)
# filter_df=df
# filter_literature_080 = set(filter_df[filter_df['080'].str.startswith('821', na=False)]['150'].dropna().unique())
output_file = 'Aaut_ph_selected.xlsx'
df.to_excel(output_file, index=False)

# Function to parse the new MARC XML and filter based on existing DataFrame
def filter_marcxml(file_path, filter_df, output_marc21_file, output_mrk_file):
    records = []
    processed_count = 0
    matching_count = 0
    
    # Parse the MARC XML file
    marc_records = parse_xml_to_array(file_path)
    
    filter_072 = set(filter_df['072'].dropna().unique())
    filter_080 = set(filter_df['080'].dropna().unique())
    filter_150 = set(filter_df['150'].dropna().unique())
    
    writer = MARCWriter(open(output_marc21_file, 'wb'))
    writer_mrk = TextWriter(open(output_mrk_file, 'wt', encoding='utf-8'))
    
    for record in tqdm(marc_records, desc="Processing records", total=len(marc_records)):
        try:
            processed_count += 1
            record_id = record['001'].value() if record['001'] else None
            field_072_values = [subfield for field in record.get_fields('072') for subfield in field.get_subfields('9') if subfield in filter_072]
            field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a') if subfield in filter_080]
            field_650_values = [subfield for field in record.get_fields('650') for subfield in field.get_subfields('a') if subfield in filter_150]

            if field_072_values or field_080_values or field_650_values:
                matching_count += 1
                records.append({
                    'Record ID': record_id,
                    '072': ', '.join(field_072_values),
                    '080': ', '.join(field_080_values),
                    '650': ', '.join(field_650_values)
                })
                writer.write(record)
                writer_mrk.write(record)
        
        except Exception as e:
            print(f"Error processing record ID {record_id}: {e}")
    
    writer_mrk.close()
    writer.close()
    
    print(f"Total processed records: {processed_count}")
    print(f"Total matching records: {matching_count}")
    
    return pd.DataFrame(records)

# Example usage
file_path='D:/Nowa_praca/czech_works/nkc.xml/nkc.xml'

filtered_df = filter_marcxml('D:/Nowa_praca/czech_works/nkc.xml/nkc.xml',df, 'output_books_nkc.mrc','output_books_nkc.mrk')
print(filtered_df)


#file splitting
#liczenie rekordóW

import os
from pymarc.marcxml import XmlHandler, map_xml
from tqdm import tqdm

class CountRecordsXmlHandler(XmlHandler):
    def __init__(self):
        super().__init__()
        self.record_count = 0
        self.pbar = tqdm(desc="Counting records", unit="record")
    
    def process_record(self, record):
        self.record_count += 1
        self.pbar.update(1)
    
    def endDocument(self):
        self.pbar.close()
        print(f"Total records in the original file: {self.record_count}")

# Przykładowe użycie do liczenia rekordów
file_path = 'D:/Nowa_praca/czech_works/nkc.xml/nkc.xml'
count_handler = CountRecordsXmlHandler()
map_xml(count_handler.process_record, file_path)



import os
from pymarc import TextWriter, MARCWriter, Field
from pymarc.marcxml import XmlHandler, map_xml
from tqdm import tqdm

class CustomXmlHandler(XmlHandler):
    def __init__(self, chunk_size=500000, output_dir='chunks'):
        super().__init__()
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.chunk_records = []
        self.chunk_count = 0
        self.record_count = 0
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def write_chunk(self):
        if not self.chunk_records:
            return

        self.chunk_count += 1
        chunk_file_mrk = os.path.join(self.output_dir, f'chunk_{self.chunk_count}.mrk')
        chunk_file_mrc = os.path.join(self.output_dir, f'chunk_{self.chunk_count}.mrc')
        
        with open(chunk_file_mrk, 'wt', encoding='utf-8') as f:
            writer = TextWriter(f)
            for record in self.chunk_records:
                try:
                    writer.write(record)
                except Exception as e:
                    print(f"Error writing record to MRK file: {e}")
            writer.close()
        
        with open(chunk_file_mrc, 'wb') as f:
            writer = MARCWriter(f)
            for record in self.chunk_records:
                try:
                    writer.write(record)
                except Exception as e:
                    print(f"Error writing record to MARC21 file: {e}")
            writer.close()
        
        print(f"Written chunk {self.chunk_count} with {len(self.chunk_records)} records.")
        self.chunk_records = []
    
    def process_record(self, record):
        try:
            # Validate record structure here
            for field in record.get_fields():
                if isinstance(field, Field) and field.tag.isdigit():
                    if hasattr(field, 'indicators') and len(field.indicators) != 2:
                        raise ValueError(f"Field {field.tag} is missing indicators.")
            
            self.chunk_records.append(record)
            self.record_count += 1
            
            if len(self.chunk_records) >= self.chunk_size:
                self.write_chunk()
        
        except Exception as e:
            print(f"Error processing record: {e}")
    
    def endDocument(self):
        self.write_chunk()  # Make sure to write any remaining records
        print(f"Podzielono rekordy na {self.chunk_count} części.")
        print(f"Total records processed: {self.record_count}")

# Przykładowe użycie
file_path = 'D:/Nowa_praca/czech_works/nkc.xml/nkc.xml'
handler = CustomXmlHandler(chunk_size=500000, output_dir='chunks')
map_xml(handler.process_record, file_path)
handler.endDocument()  # Ręczne wywołanie endDocument aby upewnić się, że wszystkie rekordy są zapisane
#%%
#simple,, splitting records more memory:
from pymarc import XmlHandler, map_xml,parse_xml,MARCWriter,TextWriter
from tqdm import tqdm
marc_file = 'records'
file_path = 'D:/Nowa_praca/czech_works/anl.xml/anl.xml'
from pymarc import XmlHandler, parse_xml, MARCWriter, TextWriter
from tqdm import tqdm
import logging

# Ustawienia logowania
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

# Define the file paths and settings
marc_file = 'records'
file_path = 'D:/Nowa_praca/czech_works/anl.xml/anl.xml'
chunk_size = 500000

# Initialize the XML parser
parserxml = XmlHandler()
try:
    parse_xml(file_path, parserxml)
except Exception as e:
    logging.error(f"Error parsing XML file: {e}")
    raise
# Inicjalizacja parsera
parserxml = XmlHandler()
parse_xml(file_path, parserxml)

# Ustawienia chunkowania
chunk_size = 500000
counter = 0
file_count = 1

for record in parserxml.records:
    if counter % chunk_size == 0 and counter > 0:
        # Zapisywanie rekordów do plików co 10000 rekordów
        mrk_file_name = f"{marc_file}_{file_count}.mrk"
        mrc_file_name = f"{marc_file}_{file_count}.mrc"
        
        with open(mrk_file_name, 'wt', encoding="utf-8") as mrk_file, open(mrc_file_name, 'wb') as mrc_file:
            text_writer = TextWriter(mrk_file)
            marc_writer = MARCWriter(mrc_file)
            for rec in parserxml.records[counter - chunk_size:counter]:
                try:
                    text_writer.write(rec)
                    marc_writer.write(rec)
                except: pass
            text_writer.close()
            marc_writer.close()
        file_count += 1
    counter += 1

# Zapisywanie pozostałych rekordów
if counter % chunk_size != 0:
    mrk_file_name = f"{marc_file}_{file_count}.mrk"
    mrc_file_name = f"{marc_file}_{file_count}.mrc"
    
    with open(mrk_file_name, 'wt', encoding="utf-8") as mrk_file, open(mrc_file_name, 'wb') as mrc_file:
        text_writer = TextWriter(mrk_file)
        marc_writer = MARCWriter(mrc_file)
        for rec in parserxml.records[counter - (counter % chunk_size):counter]:
            try: 
                text_writer.write(rec)
                marc_writer.write(rec)
            except: pass    
        text_writer.close()
        marc_writer.close()
#%% BAD FILE WITH ERROR:
    
from pymarc import XmlHandler, parse_xml, MARCWriter, TextWriter
from tqdm import tqdm
import logging
from xml.sax import SAXParseException

# Ustawienia logowania
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

# Define the file paths and settings
marc_file = 'records'
file_path = 'D:/Nowa_praca/czech_works/anl.xml/anl.xml'
chunk_size = 500000

# Custom XmlHandler to skip problematic records
class SafeXmlHandler(XmlHandler):
    def __init__(self):
        super().__init__()
        self.safe_records = []

    def endElementNS(self, name, qname):
        try:
            super().endElementNS(name, qname)
        except Exception as e:
            logging.error(f"Error parsing record: {e}")

    def characters(self, content):
        try:
            super().characters(content)
        except Exception as e:
            logging.error(f"Error in characters method: {e}")

# Initialize the custom XML parser
parserxml = SafeXmlHandler()
try:
    parse_xml(file_path, parserxml)
except SAXParseException as e:
    logging.error(f"Error parsing XML file: {e}")
    raise

# Initialize counters
counter = 0
file_count = 1

# Process records in chunks
try:
    for record in parserxml.records:
        if counter % chunk_size == 0 and counter > 0:
            mrk_file_name = f"{marc_file}_{file_count}.mrk"
            mrc_file_name = f"{marc_file}_{file_count}.mrc"

            with open(mrk_file_name, 'wt', encoding="utf-8") as mrk_file, open(mrc_file_name, 'wb') as mrc_file:
                text_writer = TextWriter(mrk_file)
                marc_writer = MARCWriter(mrc_file)
                for rec in parserxml.records[counter - chunk_size:counter]:
                    try:
                        text_writer.write(rec)
                        marc_writer.write(rec)
                    except Exception as e:
                        logging.error(f"Error writing record at position {counter}: {e}")
                text_writer.close()
                marc_writer.close()
            file_count += 1
        counter += 1

    # Process remaining records
    if counter % chunk_size != 0:
        mrk_file_name = f"{marc_file}_{file_count}.mrk"
        mrc_file_name = f"{marc_file}_{file_count}.mrc"

        with open(mrk_file_name, 'wt', encoding="utf-8") as mrk_file, open(mrc_file_name, 'wb') as mrc_file:
            text_writer = TextWriter(mrk_file)
            marc_writer = MARCWriter(mrc_file)
            for rec in parserxml.records[counter - (counter % chunk_size):counter]:
                try:
                    text_writer.write(rec)
                    marc_writer.write(rec)
                except Exception as e:
                    logging.error(f"Error writing record at position {counter}: {e}")
            text_writer.close()
            marc_writer.close()
except Exception as e:
    logging.error(f"Error processing records: {e}")

# Logowanie zakończonego procesu
print("Processing complete. Check error_log.txt for details.")
#%%
    
#%% 
criteria_description = """
CRITERION A: 695 - Literature/Literary Science/Non-Literary
NKC - only

Literature 
anything containing 25 or 26 in 072-9 OR 
anything containing value starting with 821 in 080 field (UDC) OR
anything containing literal values adequate to aforementioned values 080 in 650 field

Literary Science
anything containing 11 in 072-9 AND 
anything containing value starting with 82, but not with 821 in 080 field 
OR anything containing literal values adequate to aforementioned 080 in 650 field

Non-Literary Production of Literary Authors (only from 1) literature)
create a list of ID-s from 100-7 subfield from records from category “Literature” -> search for any other record in the data containing this ID in 100-7 or 700-7 subfield

CRITERION B: 690 - Czech Literature/World Literature

Czech Literature
Anything containing field starting with 821.162.3 in 080 or with “česk*” in 655a

World Literature
anything containing any other value starting with 821 in 080 field (UDC)

CRITERION C: NOT NEEDED AT THE MOMENT

CRITERION D: 691 Printed (m,a,b,s) / Other forms (always 1 per each record)

Printed (Books, Magazines, Articles)
-> LDR-8 is m, a, b or s
Other forms 
-> LDR-8 is containing else

Add to excel author 100 (a) title (all subfields) 245
"""
print(criteria_description)
   
import os
from pymarc import MARCReader, TextWriter, MARCWriter, Field, Subfield
from tqdm import tqdm
import pandas as pd

def filter_marcxml(file_path, filter_df, writer_marc, writer_mrk):
    records = []
    processed_count = 0
    matching_count = 0
    literary_author_ids = set()
    literature_record_ids = set()  # Dodana kolekcja do zbierania ID rekordów literackich
    literature_authors = []  # Lista do przechowywania autorów literatury
    
    # Filtracja wartości z pola 150 na podstawie wartości w polu 080
    filter_literature_080 = set(filter_df[filter_df['080'].str.startswith('821', na=False)]['150'].dropna().unique())
    filter_literary_science_080 = set(filter_df[filter_df['080'].str.startswith('82', na=False) & ~filter_df['080'].str.startswith('821', na=False)]['150'].dropna().unique())
    
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for record in tqdm(reader, desc="Processing records"):
            try:
                if any(field.get_subfields('a')[0] == '0/9-053.2' or 'Literatura pro děti a mládež (naučná)' in field.get_subfields('x') for field in record.get_fields('072')):
                    continue  # Pomijanie rekordów spełniających warunek wykluczenia

                processed_count += 1
                record_id = record['001'].value() if record['001'] else None
                link = record['998'].value() if record['998'] else None
                
                field_015_values = [subfield for field in record.get_fields('015') for subfield in field.get_subfields('a')]
                field_072_values = [subfield for field in record.get_fields('072') for subfield in field.get_subfields('9')]
                field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')]
                field_650_values = [subfield for field in record.get_fields('650') for subfield in field.get_subfields('a')]
                field_655_values = [subfield for field in record.get_fields('655') for subfield in field.get_subfields('a')]
                field_100_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')]
                field_245_values = ' '.join(field.value() for field in record.get_fields('245'))
                field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                # CRITERION A: 695
                is_literature = any(val in ['25', '26'] for val in field_072_values) or \
                                any(val.startswith('821') for val in field_080_values) or \
                                any(val in filter_literature_080 for val in field_650_values)
                
                is_literary_science = ('11' in field_072_values and (
                    any(val.startswith('82') and not val.startswith('821') for val in field_080_values)
                )) or any(val in filter_literary_science_080 for val in field_650_values)
                
                if is_literature or is_literary_science:
                    literature_record_ids.add(record_id)
                    if is_literature:
                        for id_ in field_100_7_values:
                            literary_author_ids.add(id_)
                        for author, id_ in zip(field_100_values, field_100_7_values):
                            literature_authors.append({'Author': author, 'ID': id_})

                if is_literature:
                    record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literature')]))

                if is_literary_science:
                    record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literary Science')]))

                # CRITERION B: 690
                is_czech_literature = any(val.startswith('821.162.3') for val in field_080_values) or \
                                      any('česk' in val for val in field_655_values)
                
                is_world_literature = any(val.startswith('821') and not val.startswith('821.162.3') for val in field_080_values)
                
                if is_czech_literature:
                    record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='česká literatura')]))
                if is_world_literature:
                    record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='světová literatura')]))

                # CRITERION D: 691
                ldr_8 = record.leader[7]
                if ldr_8 in ['m', 'a', 'b', 's']:
                    record.add_field(Field(tag='691', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Printed')]))
                else:
                    record.add_field(Field(tag='691', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Other forms')]))

                if is_literature or is_literary_science or is_czech_literature or is_world_literature:
                    matching_count += 1
                    records.append({
                        'Record ID': record_id,
                        'Link': link,
                        '015': ', '.join(field_015_values),
                        '072': ', '.join(field_072_values),
                        '080': ', '.join(field_080_values),
                        '650': ', '.join(field_650_values),
                        '655': ', '.join(field_655_values),
                        '100': ', '.join(field_100_values),
                        '245': field_245_values,
                        '695': ', '.join(filter(None, ['Literature' if is_literature else '', 'Literary Science' if is_literary_science else ''])).strip(', '),
                        '690': ', '.join(filter(None, ['Czech Literature' if is_czech_literature else '', 'World Literature' if is_world_literature else ''])).strip(', '),
                        '691': 'Printed' if ldr_8 in ['m', 'a', 'b', 's'] else 'Other forms'
                    })
                    writer_marc.write(record)
                    writer_mrk.write(record)
            
            except Exception as e:
                print(f"Error processing record ID {record_id}: {e}")
    
    literature_authors_df = pd.DataFrame(literature_authors).drop_duplicates()
    return pd.DataFrame(records), literary_author_ids, literature_authors_df, literature_record_ids

def process_chunks(chunk_files, filter_df, output_marc21_file, output_mrk_file):
    all_records = []
    all_literary_author_ids = set()
    all_literature_record_ids = set()  # Dodana kolekcja do zbierania ID rekordów literackich
    all_literature_authors = []  # Lista do przechowywania autorów literatury
    
    with open(output_marc21_file, 'wb') as marc_fh, open(output_mrk_file, 'wt', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)
        
        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
            chunk_df, literary_author_ids, literature_authors_df, literature_record_ids = filter_marcxml(chunk_file_mrc, filter_df, writer_marc, writer_mrk)
            all_records.append(chunk_df)
            all_literary_author_ids.update(literary_author_ids)
            all_literature_record_ids.update(literature_record_ids)
            all_literature_authors.append(literature_authors_df)
        
        writer_marc.close()
        writer_mrk.close()
    
    combined_df = pd.concat(all_records, ignore_index=True)
    combined_literature_authors_df = pd.concat(all_literature_authors, ignore_index=True).drop_duplicates()
    
    # Reprocess the chunks to find non-literary works of literary authors
    reprocess_records = []
    with open(output_marc21_file, 'ab') as marc_fh, open(output_mrk_file, 'at', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)
        
        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Reprocessing części {i+1} z {len(chunk_files)}")
            with open(chunk_file_mrc, 'rb') as fh:
                reader = MARCReader(fh)
                for record in tqdm(reader, desc="Reprocessing records"):
                    try:
                        record_id = record['001'].value() if record['001'] else None
                        if record_id in all_literature_record_ids:
                            continue  # Pomijanie rekordów literackich i literatury naukowej
                        
                        link = record['998'].value() if record['998'] else None
                        
                        field_015_values = [subfield for field in record.get_fields('015') for subfield in field.get_subfields('a')]
                        field_072_values = [subfield for field in record.get_fields('072') for subfield in field.get_subfields('9')]
                        field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')]
                        field_650_values = [subfield for field in record.get_fields('650') for subfield in field.get_subfields('a')]
                        field_655_values = [subfield for field in record.get_fields('655') for subfield in field.get_subfields('a')]
                        field_100_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')]
                        field_245_values = ' '.join(field.value() for field in record.get_fields('245'))
                        field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                        field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                        # Check if the record belongs to non-literary production of literary authors
                        if any(author_id in all_literary_author_ids for author_id in field_100_7_values + field_700_7_values):
                            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Non-Literary')]))
                            ldr_8 = record.leader[7]
                            reprocess_records.append({
                                'Record ID': record_id,
                                'Link': link,
                                '015': ', '.join(field_015_values),
                                '072': ', '.join(field_072_values),
                                '080': ', '.join(field_080_values),
                                '650': ', '.join(field_650_values),
                                '655': ', '.join(field_655_values),
                                '100': ', '.join(field_100_values),
                                '245': field_245_values,
                                '695': 'Non-Literary',
                                '690': '',  # Not applicable in reprocessing
                                '691': 'Printed' if ldr_8 in ['m', 'a', 'b', 's'] else 'Other forms'  # Dodanie pola 691
                            })
                            writer_marc.write(record)
                            writer_mrk.write(record)
            
                    except Exception as e:
                        print(f"Error reprocessing record ID {record_id}: {e}")
    
    final_df = pd.DataFrame(reprocess_records)
    combined_df = pd.concat([combined_df, final_df], ignore_index=True)
    return combined_df, combined_literature_authors_df

# Przykładowe użycie

chunk_files_path = 'D:/Nowa_praca/czech_works/chunks_NKC'
chunk_files = [os.path.join(chunk_files_path, f'chunk_{i+1}.mrc') for i in range(6)]

# Przetwarzanie plików chunków i zapisanie wyników do jednego zbiorczego pliku mrc i mrk
combined_df, combined_literature_authors_df = process_chunks(chunk_files, df, 'filtered_combined.mrc', 'filtered_combined.mrk')
combined_df.to_excel('filtered_combined_12-06.xlsx', index=False)
combined_literature_authors_df.to_excel('literature_authors.xlsx', index=False)
with pd.ExcelWriter('filtered_combined_19-06.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    workbook.strings_to_urls = False  # Wyłącz konwersję ciągów znaków na URL
    combined_df.to_excel(writer, index=False)
#%%Conditions 23.07.2024

import os
from pymarc import MARCReader, TextWriter, MARCWriter, Field, Subfield
from tqdm import tqdm
import pandas as pd


def filter_marcxml(file_path, filter_df, writer_marc, writer_mrk, saved_count):
    records = []
    processed_count = 0
    matching_count = 0
    
    literature_record_ids = set()
    literary_author_ids = set()
    literature_authors = []  # Lista do przechowywania autorów literatury
    unidentified_record_ids = set()  # Zestaw do przechowywania ID rekordów jako "Unidentified"
    
    # Filtracja wartości z pola 150 na podstawie wartości w polu 080
    filter_literature_080 = set(filter_df[filter_df['080'].str.startswith('821', na=False)]['150'].dropna().unique())
    filter_literary_science_080 = set(filter_df[filter_df['080'].str.startswith('82', na=False) & ~filter_df['080'].str.startswith('821', na=False)]['150'].dropna().unique())
    
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for record in tqdm(reader, desc="Processing records"):
            try:
                if any(field.get_subfields('a')[0] == '0/9-053.2' or 'Literatura pro děti a mládež (naučná)' in field.get_subfields('x') for field in record.get_fields('072')):
                    continue  # Pomijanie rekordów spełniających warunek wykluczenia

                processed_count += 1
                record_id = record['001'].value() if record['001'] else None
                link = record['998'].value() if record['998'] else None
                
                field_015_values = [subfield for field in record.get_fields('015') for subfield in field.get_subfields('a')]
                field_072_values = [subfield for field in record.get_fields('072') for subfield in field.get_subfields('9')]
                field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')]
                field_650_values = [subfield for field in record.get_fields('650') for subfield in field.get_subfields('a')]
                field_655_values = [subfield for field in record.get_fields('655') for subfield in field.get_subfields('a')]
                field_100_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')]
                field_245_values = ' '.join(field.value() for field in record.get_fields('245'))
                field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                # CRITERION A: 695
                is_literature = any(val in ['25', '26'] for val in field_072_values) or \
                                any('česk' in val for val in field_655_values) or \
                                any(val.startswith('821') for val in field_080_values) or \
                                any(val in filter_literature_080 for val in field_650_values) or \
                                any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' in val for val in field_080_values)

                is_literary_science = ('11' in field_072_values and 
                                      (any(val.startswith('82') and not val.startswith(('820', '821')) for val in field_080_values) or 
                                       any(val in filter_literary_science_080 for val in field_650_values))) or \
                                      any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' not in val for val in field_080_values) or \
                                      any(val.startswith('82.') for val in field_080_values)
                
                has_fields = field_072_values or field_080_values or field_655_values

                if is_literature:
                    record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literature')]))
                    literature_record_ids.add(record_id)
                    for id_ in field_100_7_values:
                        literary_author_ids.add(id_)
                    for author, id_ in zip(field_100_values, field_100_7_values):
                        literature_authors.append({'Author': author, 'ID': id_})

                if is_literary_science:
                    record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literary Science')]))
                    literature_record_ids.add(record_id)

                unidentified = not has_fields and not is_literature and not is_literary_science
                if unidentified:
                    
                    record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Unidentified')]))
                    unidentified_record_ids.add(record_id)

                # CRITERION B: 690
                is_czech_literature = any(val.startswith('821.162.3') for val in field_080_values) or \
                                      any('česk' in val for val in field_655_values) or \
                                      any(val.startswith('885.0-') for val in field_080_values)
                
                is_world_literature = any(val.startswith('821') and not val.startswith('821.162.3') for val in field_080_values) or \
                                      any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' in val and not val.startswith('885.0-') for val in field_080_values)
                
                if is_czech_literature:
                    record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Czech Literature')]))
                if is_world_literature:
                    record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='World Literature')]))

                # CRITERION D: 691
                ldr_8 = record.leader[7]
                form_691 = 'Printed' if ldr_8 in ['m', 'a', 'b', 's'] else 'Other forms'
                if any(val.startswith('elektron') or val.startswith('stream') for val in field_655_values) or \
                   any('[zvukový záznam]' in field_245_values or '[elektronický zdroj]' in field_245_values for val in field_245_values):
                    form_691 = 'Other forms'
                record.add_field(Field(tag='691', indicators=[' ', ' '], subfields=[Subfield(code='a', value=form_691)]))

                if is_literature or is_literary_science or is_czech_literature or is_world_literature or unidentified:
                    matching_count += 1
                    records.append({
                        'Record ID': record_id,
                        'Link': link,
                        '015': ', '.join(field_015_values),
                        '072': ', '.join(field_072_values),
                        '080': ', '.join(field_080_values),
                        '650': ', '.join(field_650_values),
                        '655': ', '.join(field_655_values),
                        '100': ', '.join(field_100_values),
                        '245': field_245_values,
                        '695': ', '.join(filter(None, [
                            'Literature' if is_literature else '', 
                            'Literary Science' if is_literary_science else '', 
                            'Unidentified'  if unidentified else ''
                        ])).strip(', '),
                        '690': ', '.join(filter(None, [
                            'Czech Literature' if is_czech_literature else '', 
                            'World Literature' if is_world_literature else ''
                        ])).strip(', '),
                        '691': form_691
                    })
                writer_marc.write(record)
                writer_mrk.write(record)
                saved_count[0] += 1
                print(saved_count[0])
                
            except Exception as e:
                print(f"Error processing record ID {record_id}: {e}")
    
    literature_authors_df = pd.DataFrame(literature_authors).drop_duplicates()
    return pd.DataFrame(records), literary_author_ids, literature_authors_df, literature_record_ids, unidentified_record_ids



def process_chunks(chunk_files, filter_df, output_marc21_file, output_mrk_file):
    all_records = []
    all_literature_record_ids = set()
    all_literary_author_ids = set()
    all_literature_authors = []  # Lista do przechowywania autorów literatury
    all_unidentified_record_ids = set()
    saved_count = [0]
    # Przetwarzanie plików chunków i zapisanie wyników w plikach mrc/mrk
    with open(output_marc21_file, 'wb') as marc_fh, open(output_mrk_file, 'wt', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)
        
        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
            chunk_df, literary_author_ids, literature_authors_df, literature_record_ids, unidentified_record_ids = filter_marcxml(chunk_file_mrc, filter_df, writer_marc, writer_mrk, saved_count)
            all_records.append(chunk_df)
            all_literary_author_ids.update(literary_author_ids)
            all_literature_record_ids.update(literature_record_ids)
            all_literature_authors.append(literature_authors_df)
            all_unidentified_record_ids.update(unidentified_record_ids)
        
        writer_marc.close()
        writer_mrk.close()
    
    combined_df = pd.concat(all_records, ignore_index=True)
    combined_literature_authors_df = pd.concat(all_literature_authors, ignore_index=True).drop_duplicates()
    
    # Reprocess the saved files to add "Non-Literary" information
    reprocess_records = []
    reprocess_ids = set()
    with open(output_marc21_file, 'rb') as marc_fh, open(output_mrk_file, 'rt', encoding='utf-8') as mrk_fh:
        reader_marc = MARCReader(marc_fh)
        
        with open('reprocessed_' + output_marc21_file, 'wb') as reprocessed_marc_fh, open('reprocessed_' + output_mrk_file, 'wt', encoding='utf-8') as reprocessed_mrk_fh:
            writer_marc = MARCWriter(reprocessed_marc_fh)
            writer_mrk = TextWriter(reprocessed_mrk_fh)
            
            for record in tqdm(reader_marc, desc="Reprocessing MARC records"):
                record_id = record['001'].value() if record['001'] else None
                if record_id in all_literature_record_ids:
                    writer_marc.write(record)
                    writer_mrk.write(record)
                    continue
                
                field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]
                
                if any(author_id in all_literary_author_ids for author_id in field_100_7_values + field_700_7_values):
                    ldr_8 = record.leader[7]
                    form_691 = 'Printed' if ldr_8 in ['m', 'a', 'b', 's'] else 'Other forms'
                    if any(val.startswith('elektron') or val.startswith('stream') for val in (subfield for field in record.get_fields('655') for subfield in field.get_subfields('a'))) or \
                       any('[zvukový záznam]' in field.value() or '[elektronický zdroj]' in field.value() for field in record.get_fields('245')):
                        form_691 = 'Other forms'
                    
                    if record_id in all_unidentified_record_ids:
                        mask = combined_df['Record ID'] == record_id
                        if mask.any():
                            combined_df.loc[mask, '695'] = combined_df.loc[mask, '695'] + ', Non-Literary'
                            combined_df.loc[mask, '691'] = form_691
                            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Non-Literary')]))
                            writer_marc.write(record)
                            writer_mrk.write(record)
                            continue
                    else:
                        reprocess_ids.add(record_id)
                        reprocess_records.append({
                            'Record ID': record_id,
                            'Link': record['998'].value() if record['998'] else None,
                            '015': ', '.join(subfield for field in record.get_fields('015') for subfield in field.get_subfields('a')),
                            '072': ', '.join(subfield for field in record.get_fields('072') for subfield in field.get_subfields('9')),
                            '080': ', '.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')),
                            '650': ', '.join(subfield for field in record.get_fields('650') for subfield in field.get_subfields('a')),
                            '655': ', '.join(subfield for field in record.get_fields('655') for subfield in field.get_subfields('a')),
                            '100': ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')),
                            '245': ' '.join(field.value() for field in record.get_fields('245')),
                            '695': 'Non-Literary',
                            '691': form_691  # Dodanie pola 691
                        })
                        record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Non-Literary')]))
                        record.add_field(Field(tag='691', indicators=[' ', ' '], subfields=[Subfield(code='a', value=form_691)]))
                writer_marc.write(record)
                writer_mrk.write(record)

            writer_marc.close()
            writer_mrk.close()
    
    final_df = pd.DataFrame(reprocess_records)
    combined_df = pd.concat([combined_df, final_df], ignore_index=True)

    # Przefiltrowanie wynikowych plików przez nasze zbiory identyfikatorów
    #filtered_records = []
    with open('reprocessed_' + output_marc21_file, 'rb') as reprocessed_marc_fh:
        reader_marc = MARCReader(reprocessed_marc_fh)
        
        with open('filtered_' + output_marc21_file, 'wb') as filtered_marc_fh, open('filtered_' + output_mrk_file, 'wt', encoding='utf-8') as filtered_mrk_fh:
            writer_marc = MARCWriter(filtered_marc_fh)
            writer_mrk = TextWriter(filtered_mrk_fh)
            
            for record in tqdm(reader_marc, desc="Filtering MARC records"):
                record_id = record['001'].value() if record['001'] else None
                if record_id:
                    if record_id in all_literature_record_ids or record_id in all_unidentified_record_ids or record_id in reprocess_ids:
                        writer_marc.write(record)
                        writer_mrk.write(record)
                        #filtered_records.append(record)

            writer_marc.close()
            writer_mrk.close()

    return combined_df, combined_literature_authors_df

# Przykładowe użycie

chunk_files_path = 'D:/Nowa_praca/czech_works/chunks_NKC'
chunk_files = [os.path.join(chunk_files_path, f'chunk_{i+1}.mrc') for i in range(6)]
# Przetwarzanie plików chunków i zapisanie wyników do jednego zbiorczego pliku mrc i mrk
combined_df, combined_literature_authors_df = process_chunks(chunk_files, df, 'filtered_combined.mrc', 'filtered_combined.mrk')
combined_literature_authors_df.to_excel('literature_authors.xlsx', index=False)
with pd.ExcelWriter('filtered_combined_1-08.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    workbook.strings_to_urls = False  # Disable string to URL conversion
    max_rows = 1048575  # Excel's row limit minus 1 for the header

    # Calculate the number of sheets needed
    num_sheets = len(combined_df) // max_rows + 1

    for sheet_number in range(num_sheets):
        start_row = sheet_number * max_rows
        end_row = (sheet_number + 1) * max_rows
        # Write a portion of the DataFrame to a specific Excel sheet
        sheet_name = f'Sheet{sheet_number + 1}'
        combined_df.iloc[start_row:end_row].to_excel(writer, sheet_name=sheet_name, index=False)

#%% przeniesienie "unindentified" do nowego arkusza, w drugim arkuszu zostaje wszystko zidentyfikowane i niezidentyfikowane z nonliterary works of literary authors
import pandas as pd

# Wczytanie danych z pliku Excel (zakładając, że jest w formacie .xlsx)
file_path = 'C:/Users/dariu/Downloads/filtered_combined_1-08.xlsx'

# Wczytaj oba arkusze z pliku
sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
sheet2 = pd.read_excel(file_path, sheet_name='Sheet2')

# Połącz oba arkusze w jeden DataFrame
combined_df = pd.concat([sheet1, sheet2], ignore_index=True)

# Podziel dane na dwa DataFrame na podstawie wartości w kolumnie 695
df_unidentified = combined_df[combined_df['695'] == 'Unidentified']
df_others = combined_df[combined_df['695'] != 'Unidentified']

# Wyświetl wyniki
print("DataFrame z 'Unidentified':")
print(df_unidentified)

print("\nDataFrame z innymi wartościami:")
print(df_others)


df_unidentified.to_excel('unidentified_data.xlsx', index=False)
df_others.to_excel('identified_data.xlsx', index=False)
with pd.ExcelWriter('identified_data.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    workbook.strings_to_urls = False  # Disable string to URL conversion
    max_rows = 1048575  # Excel's row limit minus 1 for the header

    # Calculate the number of sheets needed
    num_sheets = len(df_others) // max_rows + 1

    for sheet_number in range(num_sheets):
        start_row = sheet_number * max_rows
        end_row = (sheet_number + 1) * max_rows
        # Write a portion of the DataFrame to a specific Excel sheet
        sheet_name = f'Sheet{sheet_number + 1}'
        df_others.iloc[start_row:end_row].to_excel(writer, sheet_name=sheet_name, index=False)
record_ids = set(df_others['Record ID'].astype(str))
#%% przetwarzanie mark21 na podstawie excel file

import pandas as pd
from pymarc import MARCReader, MARCWriter, TextWriter, Field

# Wczytaj plik Excel z identyfikowanymi rekordami
df_others = pd.read_excel('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_data.xlsx')

# Zdefiniuj pliki MARC i MRK
marc_file_path = 'D:/Nowa_praca/czech_works/NKC_12_08_2024/filtered_filtered_combined.mrc'
mrk_file_path = 'D:/Nowa_praca/czech_works/NKC_12_08_2024/filtered_filtered_combined.mrk'

# Zdefiniuj ścieżki do nowych plików, gdzie zapisane zostaną wyselekcjonowane rekordy
identified_marc_file = 'D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_records.mrc'
identified_mrk_file = 'D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_records.mrk'
unidentified_marc_file = 'D:/Nowa_praca/czech_works/NKC_12_08_2024/unidentified_records.mrc'
unidentified_mrk_file = 'D:/Nowa_praca/czech_works/NKC_12_08_2024/unidentified_records.mrk'

# Wczytaj rekordy MARC i MRK
with open(marc_file_path, 'rb') as marc_fh, open(mrk_file_path, 'r', encoding='utf-8') as mrk_fh, \
     open(identified_marc_file, 'wb') as identified_marc_fh, open(identified_mrk_file, 'wt', encoding='utf-8') as identified_mrk_fh, \
     open(unidentified_marc_file, 'wb') as unidentified_marc_fh, open(unidentified_mrk_file, 'wt', encoding='utf-8') as unidentified_mrk_fh:
    
    reader_marc = MARCReader(marc_fh)
    writer_identified_marc = MARCWriter(identified_marc_fh)
    writer_unidentified_marc = MARCWriter(unidentified_marc_fh)
    writer_identified_mrk = TextWriter(identified_mrk_fh)
    writer_unidentified_mrk = TextWriter(unidentified_mrk_fh)

    # Pobierz wszystkie Record ID z dataframe df_others
    identified_ids = set(df_others['Record ID'])

    for record in reader_marc:
        record_id = record['001'].value() if record['001'] else None
        
        if record_id in identified_ids:
            # Zapisz rekord do pliku identified (mrc, mrk)
            writer_identified_marc.write(record)
            writer_identified_mrk.write(record)
        else:
            # Oznacz jako "Unidentified" i zapisz do osobnego pliku (mrc, mrk)
            record.remove_fields('695')
            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Unidentified')]))
            writer_unidentified_marc.write(record)
            writer_unidentified_mrk.write(record)

    # Zamknij wszystkich writerów
    writer_identified_marc.close()
    writer_unidentified_marc.close()
    writer_identified_mrk.close()
    writer_unidentified_mrk.close()

print("Zakończono przetwarzanie i zapis rekordów.")

    
#%%update 29.10.20

# def process_new_conditions(df, marc_records, marc_writer):
#     # 1. Zamiana "Unidentified, Non-Literary" na "Lack of Data"
#     for record in marc_records:
#         record_id = record['001'].value() if record['001'] else None
#         if not record_id:
#             continue

#         # Pobierz bieżące wartości z dataframe
#         current_695_value = df.loc[df['Record ID'] == record_id, '695'].values[0] if record_id in df['Record ID'].values else None

#         # Zamiana "Unidentified, Non-Literary" na "Lack of Data"
#         if current_695_value == "Unidentified, Non-Literary":
#             df.loc[df['Record ID'] == record_id, '695'] = 'Lack of Data'
#             # Zmieniamy również w MARC
#             record.remove_fields('695')
#             record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Lack of Data')]))
#             marc_writer.write(record)

#         # 2. Sprawdzenie warunków dla "Literature 2" (gatunki literackie w 245b)
#         field_245b_list = [subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')]
#         literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk", "obraz", "drama", "bás", "verš", "jednán"]

#         # Jeśli podpole 245b zawiera gatunek literacki, oznacz rekord jako "Literature 2"
#         if any(genre in subfield for subfield in field_245b_list for genre in literary_genres):
#             df.loc[df['Record ID'] == record_id, '695'] = 'Literature 2'
#             record.remove_fields('695')
#             record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literature 2')]))
#             marc_writer.write(record)

#         # 2b. Sprawdzenie tytułu i autora (245a i 100) dla "Literature 2"
#         field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a'))
#         field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a'))

#         # Iteracja po innych rekordach MARC w celu znalezienia takiego samego tytułu i autora, który został już oznaczony jako "Literature"
#         same_title_author_lit = any(
#             other_record for other_record in marc_records
#             if other_record['001'].value() != record_id  # Sprawdzamy inne rekordy
#             and 'Literature' in df.loc[df['Record ID'] == other_record['001'].value(), '695'].values  # Inny rekord jako "Literature"
#             and field_245a == ' '.join(subfield for field in other_record.get_fields('245') for subfield in field.get_subfields('a'))
#             and field_100 == ', '.join(subfield for field in other_record.get_fields('100') for subfield in field.get_subfields('a'))
#         )

#         # Jeśli znaleziono taki sam tytuł i autora, oznacz jako "Literature 2"
#         if same_title_author_lit:
#             df.loc[df['Record ID'] == record_id, '695'] = 'Literature 2'
#             record.remove_fields('695')
#             record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literature 2')]))
#             marc_writer.write(record)

#         # 3. Sprawdzenie tytułu i autora dla "Non Literary 2"
#         same_title_author_non_lit = any(
#             other_record for other_record in marc_records
#             if other_record['001'].value() != record_id  # Sprawdzamy inne rekordy
#             and 'Non-Literary' in df.loc[df['Record ID'] == other_record['001'].value(), '695'].values  # Inny rekord jako "Non-Literary"
#             and field_245a == ' '.join(subfield for field in other_record.get_fields('245') for subfield in field.get_subfields('a'))
#             and field_100 == ', '.join(subfield for field in other_record.get_fields('100') for subfield in field.get_subfields('a'))
#         )

#         # Jeśli znaleziono taki sam tytuł i autora, oznacz jako "Non-Literary 2"
#         if same_title_author_non_lit:
#             df.loc[df['Record ID'] == record_id, '695'] = 'Non-Literary 2'
#             record.remove_fields('695')
#             record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Non-Literary 2')]))
#             marc_writer.write(record)

#         # 4. Czech or World Literature 2 na podstawie pola 041h
#         field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h'))

#         # Sprawdzamy, czy rekord oznaczony jako "Literature" i brak mu wartości dla Czech lub World Literature
#         if df.loc[df['Record ID'] == record_id, '695'].values[0] == 'Literature' and not df.loc[df['Record ID'] == record_id, '690'].values.any():
#             if "cze" in field_041h:
#                 df.loc[df['Record ID'] == record_id, '690'] = 'Czech Literature 2'
#                 record.remove_fields('690')
#                 record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Czech Literature 2')]))
#             else:
#                 df.loc[df['Record ID'] == record_id, '690'] = 'World Literature 2'
#                 record.remove_fields('690')
#                 record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='World Literature 2')]))
#             marc_writer.write(record)

#         # 5. Czech Literature na podstawie pola 080 zawierającego 885-*
#         field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')]
#         if any("885-" in val for val in field_080_values):
#             df.loc[df['Record ID'] == record_id, '690'] = 'Czech Literature'
#             record.remove_fields('690')
#             record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Czech Literature')]))
#             marc_writer.write(record)

#     return df


# Przykład użycia


def process_new_conditions(df, marc_records, marc_writer):
    for record in tqdm(marc_records):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue

        # Pobierz bieżące wartości z dataframe
        current_695_value = df.loc[df['Record ID'] == record_id, '695'].values[0] if record_id in df['Record ID'].values else None

        # Zamiana "Unidentified, Non-Literary" na "Lack of Data"
        if current_695_value == "Unidentified, Non-Literary":
            df.loc[df['Record ID'] == record_id, '695'] = 'Lack of Data'
            record.remove_fields('695')
            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Lack of Data')]))

        # 2. Sprawdzenie warunków dla "Literature 2" (gatunki literackie w 245b)
        field_245b_list = [subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')]
        literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk", "obraz", "drama", "bás", "verš", "jednán"]

        if any(genre in subfield for subfield in field_245b_list for genre in literary_genres):
            df.loc[df['Record ID'] == record_id, '695'] = 'Literature 2'
            record.remove_fields('695')
            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literature 2')]))

        # 2b. Sprawdzenie tytułu i autora (245a i 100) dla "Literature 2"
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a'))
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a'))

        same_title_author_lit = any(
            other_record for other_record in marc_records
            if other_record['001'].value() != record_id
            and 'Literature' in df.loc[df['Record ID'] == other_record['001'].value(), '695'].values
            and field_245a == ' '.join(subfield for field in other_record.get_fields('245') for subfield in field.get_subfields('a'))
            and field_100 == ', '.join(subfield for field in other_record.get_fields('100') for subfield in field.get_subfields('a'))
        )

        if same_title_author_lit:
            df.loc[df['Record ID'] == record_id, '695'] = 'Literature 2'
            record.remove_fields('695')
            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literature 2')]))

        # 3. Sprawdzenie tytułu i autora dla "Non Literary 2"
        same_title_author_non_lit = any(
            other_record for other_record in marc_records
            if other_record['001'].value() != record_id
            and 'Non-Literary' in df.loc[df['Record ID'] == other_record['001'].value(), '695'].values
            and field_245a == ' '.join(subfield for field in other_record.get_fields('245') for subfield in field.get_subfields('a'))
            and field_100 == ', '.join(subfield for field in other_record.get_fields('100') for subfield in field.get_subfields('a'))
        )

        if same_title_author_non_lit:
            df.loc[df['Record ID'] == record_id, '695'] = 'Non-Literary 2'
            record.remove_fields('695')
            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Non-Literary 2')]))

        # 4. Czech or World Literature 2 na podstawie pola 041h
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h'))

        if df.loc[df['Record ID'] == record_id, '695'].values[0] == 'Literature' and not df.loc[df['Record ID'] == record_id, '690'].values.any():
            if "cze" in field_041h:
                df.loc[df['Record ID'] == record_id, '690'] = 'Czech Literature 2'
                record.remove_fields('690')
                record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Czech Literature 2')]))
            else:
                df.loc[df['Record ID'] == record_id, '690'] = 'World Literature 2'
                record.remove_fields('690')
                record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='World Literature 2')]))

        # 5. Czech Literature na podstawie pola 080 zawierającego 885-*
        field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')]
        if any("885-" in val for val in field_080_values):
            df.loc[df['Record ID'] == record_id, '690'] = 'Czech Literature'
            record.remove_fields('690')
            record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Czech Literature')]))

        # Zapisz zaktualizowany rekord tylko raz na końcu, po sprawdzeniu wszystkich warunków
        marc_writer.write(record)

    return df


df = pd.read_excel('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_data.xlsx')  # Odczyt pliku Excel
with open('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_records.mrc', 'rb') as marc_fh, open('updated_identified_records.mrc', 'wb') as marc_output_fh:
    marc_records = MARCReader(marc_fh)
    marc_writer = MARCWriter(marc_output_fh)
    
    updated_df = process_new_conditions(df, marc_records, marc_writer)
    marc_writer.close()

# Zapisanie zaktualizowanego dataframe
updated_df.to_excel('updated_identified_table.xlsx', index=False)


def process_new_conditions(df, marc_records, marc_writer):
    # Przygotowanie listy danych z rekordów MARC
    records_data = []
    for record in tqdm(marc_records, desc='Odczytywanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
        # Pobieranie wartości z pól MARC
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')).strip()
        field_245b = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')).strip()
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')).strip()
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h')).strip()
        field_080a = ';'.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')).strip()
        
        # Dodawanie danych do listy
        records_data.append({
            'Record ID': record_id,
            'Title': field_245a,
            'Subtitle': field_245b,
            'Author': field_100,
            'Field_041h': field_041h,
            'Field_080a': field_080a
        })
    
    # Konwersja listy danych na DataFrame i połączenie z oryginalnym df
    records_df = pd.DataFrame(records_data)
    df = pd.merge(df, records_df, on='Record ID', how='left')
    
    # 1. Zamiana "Unidentified, Non-Literary" na "Lack of Data"
    df.loc[df['695'] == "Unidentified, Non-Literary", '695'] = 'Lack of Data'
    
    # 2a. Oznaczanie "Literature 2" na podstawie gatunków literackich w 245b
    # Lista gatunków literackich
    literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk",
                       "obraz", "drama", "bás", "verš", "jednán"]
    # Sprawdzenie, czy pole 'Subtitle' zawiera którykolwiek z gatunków literackich
    df['contains_literary_genre'] = df['Subtitle'].str.lower().apply(
        lambda x: any(genre in x for genre in literary_genres) if pd.notnull(x) else False)
    # Aktualizacja '695' na 'Literature 2' tylko dla rekordów z 'Lack of Data'
    df.loc[(df['contains_literary_genre']) & (df['695'] == 'Lack of Data'), '695'] = 'Literature 2'
    
    # Filtrowanie rekordów z autorem i tworzenie kopii DataFrame
    df_with_author = df[df['Author'].notnull() & (df['Author'] != '')].copy()
    
    # Tworzenie listy par tytuł-autor dla rekordów z autorem
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    
    # Tworzenie zbioru tytułów i autorów dla "Literature"
    literature_titles_authors = set(zip(
        df[df['695'] == 'Literature']['Title'],
        df[df['695'] == 'Literature']['Author']
    ))
    
    # 2b. Sprawdzenie tytułu i autora dla "Literature 2"
    # Aktualizacja '695' na 'Literature 2' dla rekordów z 'Lack of Data', jeśli ich tytuł i autor są w zbiorze literature_titles_authors
    df.loc[
        (df['Record ID'].isin(df_with_author['Record ID'])) &
        (df['695'] == 'Lack of Data') &
        (df_with_author['title_author_tuple'].isin(literature_titles_authors)),
        '695'] = 'Literature 2'
    
    # Tworzenie zbioru tytułów i autorów dla "Non-Literary"
    non_literary_titles_authors = set(zip(
        df[df['695'] == 'Non-Literary']['Title'],
        df[df['695'] == 'Non-Literary']['Author']
    ))
    
    # 3. Sprawdzenie tytułu i autora dla "Non-Literary 2"
    # Aktualizacja '695' na 'Non-Literary 2' dla rekordów z 'Lack of Data', jeśli ich tytuł i autor są w zbiorze non_literary_titles_authors
    df.loc[
        (df['Record ID'].isin(df_with_author['Record ID'])) &
        (df['695'] == 'Lack of Data') &
        (df_with_author['title_author_tuple'].isin(non_literary_titles_authors)),
        '695'] = 'Non-Literary 2'
    
    # 4. Czech or World Literature 2 na podstawie pola 041h
    # Upewniamy się, że 'Field_041h' nie jest pusty
    df.loc[
        (df['695'] == 'Literature') & 
        (df['690'].isnull()) & 
        df['Field_041h'].notnull() & 
        df['Field_041h'].str.contains('cze', case=False, na=False),
        '690'] = 'Czech Literature 2'
    df.loc[
        (df['695'] == 'Literature') & 
        (df['690'].isnull()) & 
        df['Field_041h'].notnull() & 
        ~df['Field_041h'].str.contains('cze', case=False, na=False),
        '690'] = 'World Literature 2'
    
    # 5. Czech Literature na podstawie pola 080 zawierającego 885-*
    df.loc[df['Field_080a'].str.contains('885-', na=False), '690'] = 'Czech Literature'
    
    # Aktualizacja rekordów MARC na podstawie zaktualizowanego df
    for record in tqdm(marc_records, desc='Aktualizowanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
    
        current_row = df[df['Record ID'] == record_id]
        if current_row.empty:
            continue
    
        # Aktualizacja pola 695
        new_695_value = current_row['695'].values[0]
        if pd.notnull(new_695_value):
            # Usuń konkretne stare wartości z pola 695
            old_695_values_to_remove = ['Unidentified', 'Non-Literary', 'Unidentified, Non-Literary', 'Lack of Data']
            # Usuń pola 695 z wartościami, które zastępujemy
            existing_695_fields = record.get_fields('695')
            fields_to_remove = [field for field in existing_695_fields if 'a' in field and field['a'] in old_695_values_to_remove]
            for field in fields_to_remove:
                record.remove_field(field)
            # Sprawdź, czy nowa wartość już nie istnieje w pozostałych polach
            existing_695_values = [field['a'] for field in record.get_fields('695') if 'a' in field]
            if new_695_value not in existing_695_values:
                # Dodaj nową wartość
                record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value=new_695_value)]))
    
        # Aktualizacja pola 690
        new_690_value = current_row['690'].values[0]
        if pd.notnull(new_690_value):
            # Usuń konkretne stare wartości z pola 690
            old_690_values_to_remove = ['Czech Literature', 'World Literature', 'Czech Literature 2', 'World Literature 2']
            # Usuń pola 690 z wartościami, które zastępujemy
            existing_690_fields = record.get_fields('690')
            fields_to_remove = [field for field in existing_690_fields if 'a' in field and field['a'] in old_690_values_to_remove]
            for field in fields_to_remove:
                record.remove_field(field)
            # Sprawdź, czy nowa wartość już nie istnieje w pozostałych polach
            existing_690_values = [field['a'] for field in record.get_fields('690') if 'a' in field]
            if new_690_value not in existing_690_values:
                # Dodaj nową wartość
                record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value=new_690_value)]))
    
        # Zapisz zaktualizowany rekord
        marc_writer.write(record)
    
    return df

import pandas as pd
from tqdm import tqdm
from pymarc import Field, Subfield, MARCReader, MARCWriter

def process_new_conditions(df, marc_records, marc_writer):
    """
    Przetwarza rekordy MARC, aktualizując pola 695 i 690 na podstawie warunków określonych w DataFrame.
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Record ID', '695', '690'.
    - marc_records: lista rekordów MARC do przetworzenia.
    - marc_writer: obiekt do zapisywania zaktualizowanych rekordów MARC.
    
    Zwraca:
    - Zaktualizowany DataFrame i zapisuje rekordy MARC
    """
    
    # Przygotowanie listy danych z rekordów MARC
    records_data = []
    for record in tqdm(marc_records, desc='Odczytywanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
        
        # Pobieranie wartości z pól MARC
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')).strip()
        field_245b = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')).strip()
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')).strip()
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h')).strip()
        field_080a = ';'.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')).strip()


        # Dodawanie danych do listy
        records_data.append({
            'Record ID': record_id,
            'Title': field_245a,
            'Subtitle': field_245b,
            'Author': field_100,
            'Field_041h': field_041h,
            'Field_080a': field_080a
        })
    
    # Konwersja listy danych na DataFrame
    records_df = pd.DataFrame(records_data)
    
    # Rozdzielenie wielokrotnych wartości w kolumnach '695' i '690' na listy
    df['695'] = df['695'].apply(lambda x: [val.strip() for val in x.split(',')] if pd.notnull(x) else [])
    df['690'] = df['690'].apply(lambda x: [val.strip() for val in x.split(',')] if pd.notnull(x) else [])
    
    # Połączenie DataFrame z danymi z rekordów MARC
    df = pd.merge(df, records_df, on='Record ID', how='left')
    
    # 1. Zamiana "Unidentified, Non-Literary" na "Lack of Data"
    df['695'] = df['695'].apply(lambda x: ['Lack of Data'] if "Unidentified, Non-Literary" in x else x)
    
    # 2a. Oznaczanie "Literature 2" na podstawie gatunków literackich w 245b
    literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk",
                       "obraz", "drama", "bás", "verš", "jednán"]
    df['contains_literary_genre'] = df['Subtitle'].str.lower().apply(
        lambda x: any(genre in x for genre in literary_genres) if pd.notnull(x) else False)
    
    # Dodawanie "Literature 2" dla rekordów z "Lack of Data" i zawierających gatunek literacki
    df.loc[
        (df['contains_literary_genre']) & 
        (df['695'].apply(lambda x: 'Lack of Data' in x)),
        '695'
    ] = df['695'].apply(
        lambda x: x + ['Literature 2'] if 'Literature 2' not in x else x
    )
    
    # Filtrowanie rekordów z autorem i tworzenie kopii DataFrame
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    
    # Tworzenie listy par tytuł-autor dla rekordów z autorem
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    
    # Tworzenie zbioru tytułów i autorów dla "Literature"
    literature_titles_authors = set(zip(
        df[df['695'].apply(lambda x: 'Literature' in x)]['Title'],
        df[df['695'].apply(lambda x: 'Literature' in x)]['Author']
    ))
    
    # 2b. Sprawdzenie tytułu i autora dla "Literature 2"
    df.loc[
        (df['Record ID'].isin(df_with_author['Record ID'])) &
        (df_with_author['title_author_tuple'].isin(literature_titles_authors)) &
        (df['695'].apply(lambda x: 'Lack of Data' in x)),
        '695'
    ] = df['695'].apply(
        lambda x: x + ['Literature 2'] if 'Literature 2' not in x else x
    )
    
    # Tworzenie zbioru tytułów i autorów dla "Non-Literary"
    non_literary_titles_authors = set(zip(
        df[df['695'].apply(lambda x: 'Non-Literary' in x)]['Title'],
        df[df['695'].apply(lambda x: 'Non-Literary' in x)]['Author']
    ))
    
    # 3. Sprawdzenie tytułu i autora dla "Non-Literary 2"
    df.loc[
        (df['Record ID'].isin(df_with_author['Record ID'])) &
        (df_with_author['title_author_tuple'].isin(non_literary_titles_authors)) &
        (df['695'].apply(lambda x: 'Lack of Data' in x)),
        '695'
    ] = df['695'].apply(
        lambda x: x + ['Non-Literary 2'] if 'Non-Literary 2' not in x else x
    )
    
    # 4. Czech or World Literature 2 na podstawie pola 041h
    # Dodajemy 'Czech Literature 2' lub 'World Literature 2' tylko jeśli pole 690 jest puste
    df.loc[
        (df['695'].apply(lambda x: 'Literature' in x)) & 
        (df['690'].apply(lambda x: len(x) == 0)) & 
        (df['Field_041h'].str.contains('cze', case=False, na=False)),
        '690'
    ] = df['690'].apply(
        lambda x: x + ['Czech Literature 2'] if 'Czech Literature 2' not in x else x
    )
    
    df.loc[
        (df['695'].apply(lambda x: 'Literature' in x)) & 
        (df['690'].apply(lambda x: len(x) == 0)) & 
        (~df['Field_041h'].str.contains('cze', case=False, na=False)),
        '690'
    ] = df['690'].apply(
        lambda x: x + ['World Literature 2'] if 'World Literature 2' not in x else x
    )
    
    # 5. Czech Literature na podstawie pola 080 zawierającego "885-*"
    # Dodajemy 'Czech Literature' tylko jeśli nie ma już jej w '690'
    df.loc[
        df['Field_080a'].str.contains('885-', na=False) & 
        (~df['690'].apply(lambda x: 'Czech Literature' in x)),
        '690'
    ] = df['690'].apply(
        lambda x: x + ['Czech Literature'] if 'Czech Literature' not in x else x
    )
    
    # Aktualizacja rekordów MARC na podstawie zaktualizowanego df
    df.set_index('Record ID', inplace=True)
    
    # Przetwarzanie rekordów MARC w strumieniu
    for record in tqdm(marc_records, desc='Aktualizowanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            marc_writer.write(record)
            continue

        # Sprawdzenie, czy rekord ma odpowiadający wiersz w DataFrame
        if record_id not in df.index:
            marc_writer.write(record)
            continue

        # Pobranie wiersza z DataFrame
        current_row = df.loc[record_id]
        
        # Pobranie zaktualizowanych list wartości
        new_695_values = current_row['695'] if isinstance(current_row['695'], list) else []
        new_690_values = current_row['690'] if isinstance(current_row['690'], list) else []
        
        # Aktualizacja pola 695
        if new_695_values:
            # Usunięcie wszystkich istniejących pól 695
            existing_695_fields = record.get_fields('695')
            for field in existing_695_fields:
                record.remove_field(field)
            
            # Dodanie nowych pól 695
            for val in new_695_values:
                record.add_field(
                    Field(
                        tag='695',
                        indicators=[' ', ' '],
                        subfields=[Subfield(code='a', value=val)]
                    )
                )
        
        # Aktualizacja pola 690
        if new_690_values:
            # Usunięcie wszystkich istniejących pól 690
            existing_690_fields = record.get_fields('690')
            for field in existing_690_fields:
                record.remove_field(field)
            
            # Dodanie nowych pól 690
            for val in new_690_values:
                record.add_field(
                    Field(
                        tag='690',
                        indicators=[' ', ' '],
                        subfields=[Subfield(code='a', value=val)]
                    )
                )
        
        # Zapisanie zaktualizowanego rekordu
        marc_writer.write(record)
    
    # Usunięcie tymczasowych kolumn
    df.drop(columns=['contains_literary_genre'], inplace=True)
    if 'title_author_tuple' in df_with_author.columns:
        df_with_author.drop(columns=['title_author_tuple'], inplace=True)
    
    return df

import pandas as pd
from tqdm import tqdm
from pymarc import Field, Subfield, MARCReader, MARCWriter

def process_new_conditions(df, marc_records, marc_writer):
    """
    Przetwarza rekordy MARC, aktualizując pola 695 i 690 na podstawie warunków określonych w DataFrame.
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Record ID', '695', '690'.
    - marc_records: lista rekordów MARC do przetworzenia.
    - marc_writer: obiekt do zapisywania zaktualizowanych rekordów MARC.
    
    Zwraca:
    - Zaktualizowany DataFrame i zapisuje rekordy MARC
    """
    # Przygotowanie listy danych z rekordów MARC
    records_data = []
    for record in tqdm(marc_records, desc='Odczytywanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
        
        # Pobieranie wartości z pól MARC
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')).strip()
        field_245b = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')).strip()
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')).strip()
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h')).strip()
        field_080a = ';'.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')).strip()

        # Dodawanie danych do listy
        records_data.append({
            'Record ID': record_id,
            'Title': field_245a,
            'Subtitle': field_245b,
            'Author': field_100,
            'Field_041h': field_041h,
            'Field_080a': field_080a
        })
    
    # Konwersja listy danych na DataFrame
    records_df = pd.DataFrame(records_data)
    
    # Rozdzielenie wielokrotnych wartości w kolumnach '695' i '690' na listy
    df['695'] = df['695'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    df['690'] = df['690'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    
    # Połączenie DataFrame z danymi z rekordów MARC
    df = pd.merge(df, records_df, on='Record ID', how='left')
    
    # 1. Zamiana "Unidentified, Non-Literary" na "Lack of Data"
    df['695'] = df['695'].apply(lambda x: ['Lack of Data' if "Unidentified, Non-Literary" in x else val for val in x])
    
    # 2a. Oznaczanie "Literature 2" na podstawie gatunków literackich w 245b
    literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk",
                       "obraz", "drama", "bás", "verš", "jednán"]
    df['contains_literary_genre'] = df['Subtitle'].str.lower().apply(
        lambda x: any(genre in x for genre in literary_genres) if pd.notnull(x) else False)
    
    # Dodawanie "Literature 2" dla rekordów z "Lack of Data" i zawierających gatunek literacki
    df['695'] = df.apply(
        lambda row: row['695'] + ['Literature 2'] if row['contains_literary_genre'] and 'Lack of Data' in row['695'] and 'Literature 2' not in row['695'] else row['695'],
        axis=1
    )
    
    # 2b. Sprawdzenie tytułu i autora dla "Literature 2"
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    literature_titles_authors = set(zip(
        df[df['695'].apply(lambda x: 'Literature' in x)]['Title'],
        df[df['695'].apply(lambda x: 'Literature' in x)]['Author']
    ))
    df['695'] = df.apply(
        lambda row: row['695'] + ['Literature 2'] if (row['Record ID'] in df_with_author['Record ID'].values and (row['Title'], row['Author']) in literature_titles_authors and 'Lack of Data' in row['695'] and 'Literature 2' not in row['695']) else row['695'],
        axis=1
    )
    
    # 3. Sprawdzenie tytułu i autora dla "Non-Literary 2"
    non_literary_titles_authors = set(zip(
        df[df['695'].apply(lambda x: 'Non-Literary' in x)]['Title'],
        df[df['695'].apply(lambda x: 'Non-Literary' in x)]['Author']
    ))
    df['695'] = df.apply(
        lambda row: row['695'] + ['Non-Literary 2'] if (row['Record ID'] in df_with_author['Record ID'].values and (row['Title'], row['Author']) in non_literary_titles_authors and 'Lack of Data' in row['695'] and 'Non-Literary 2' not in row['695']) else row['695'],
        axis=1
    )
    
    # 4. Czech lub World Literature 2 na podstawie pola 041h
    df['690'] = df.apply(
        lambda row: row['690'] + ['Czech Literature 2'] if 'Literature' in row['695'] and len(row['690']) == 0 and 'cze' in row['Field_041h'].lower() else (
            row['690'] + ['World Literature 2'] if 'Literature' in row['695'] and len(row['690']) == 0 and 'cze' not in row['Field_041h'].lower() else row['690']
        ),
        axis=1
    )
    
    # 5. Czech Literature na podstawie pola 080 zawierającego "885-*"
    df['690'] = df.apply(
        lambda row: row['690'] + ['Czech Literature'] if '885-' in row['Field_080a'] and 'Czech Literature' not in row['690'] else row['690'],
        axis=1
    )
    
    # Aktualizacja rekordów MARC na podstawie zaktualizowanego df
    df.set_index('Record ID', inplace=True)
    
    # Przetwarzanie rekordów MARC w strumieniu
    for record in tqdm(marc_records, desc='Aktualizowanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id or record_id not in df.index:
            marc_writer.write(record)
            continue

        # Pobranie wiersza z DataFrame
        current_row = df.loc[record_id]
        
        # Pobranie zaktualizowanych list wartości
        new_695_values = current_row['695'] if isinstance(current_row['695'], list) else []
        new_690_values = current_row['690'] if isinstance(current_row['690'], list) else []
        
        # Aktualizacja pola 695
        if new_695_values:
            record.remove_fields('695')
            for val in new_695_values:
                record.add_field(
                    Field(
                        tag='695',
                        indicators=[' ', ' '],
                        subfields=[Subfield(code='a', value=val)]
                    )
                )
        
        # Aktualizacja pola 690
        if new_690_values:
            record.remove_fields('690')
            for val in new_690_values:
                record.add_field(
                    Field(
                        tag='690',
                        indicators=[' ', ' '],
                        subfields=[Subfield(code='a', value=val)]
                    )
                )
        
        # Zapisanie zaktualizowanego rekordu
        marc_writer.write(record)
    
    # Usunięcie tymczasowych kolumn
    df.drop(columns=['contains_literary_genre'], inplace=True)
    
    # Konwersja wartości '695' i '690' z powrotem na stringi oddzielone przecinkami
    df['695'] = df['695'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df['690'] = df['690'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return df

import pandas as pd
from tqdm import tqdm
from pymarc import Field, Subfield, MARCReader, MARCWriter


def process_new_conditions(df, marc_records, marc_writer):
    """
    Przetwarza rekordy MARC, aktualizując pola 695 i 690 na podstawie warunków określonych w DataFrame.
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Record ID', '695', '690'.
    - marc_records: lista rekordów MARC do przetworzenia.
    - marc_writer: obiekt do zapisywania zaktualizowanych rekordów MARC.
    
    Zwraca:
    - Zaktualizowany DataFrame i zapisuje rekordy MARC
    """
    # Przygotowanie listy danych z rekordów MARC
    records_data = []
    for record in tqdm(marc_records, desc='Odczytywanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
        
        # Pobieranie wartości z pól MARC
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')).strip()
        field_245b = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')).strip()
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')).strip()
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h')).strip()
        field_080a = ';'.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')).strip()

        # Dodawanie danych do listy
        records_data.append({
            'Record ID': record_id,
            'Title': field_245a,
            'Subtitle': field_245b,
            'Author': field_100,
            'Field_041h': field_041h,
            'Field_080a': field_080a
        })
    
    # Konwersja listy danych na DataFrame
    records_df = pd.DataFrame(records_data)
    
    # Rozdzielenie wielokrotnych wartości w kolumnach '695' i '690' na listy
    df['695'] = df['695'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    df['690'] = df['690'].apply(lambda x: x.split(',') if pd.notnull(x) else [])
    
    # Połączenie DataFrame z danymi z rekordów MARC
    df = pd.merge(df, records_df, on='Record ID', how='left')
    
    # 1. Zamiana "Unidentified, Non-Literary" na "Lack of Data"
    df['695'] = df['695'].apply(lambda x: ['Lack of Data' if "Unidentified, Non-Literary" in x else val for val in x])
    
    # 2a. Oznaczanie "Literature 2" na podstawie gatunków literackich w 245b
    literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk",
                       "obraz", "drama", "bás", "verš", "jednán"]
    df['contains_literary_genre'] = tqdm(df['Subtitle'].str.lower().apply(
        lambda x: any(genre in x for genre in literary_genres) if pd.notnull(x) else False), desc='Sprawdzanie gatunków literackich')
    
    # Dodawanie "Literature 2" dla rekordów z "Lack of Data" i zawierających gatunek literacki
    df['695'] = tqdm(df.apply(
        lambda row: row['695'] + ['Literature 2'] if row['contains_literary_genre'] and 'Lack of Data' in row['695'] and 'Literature 2' not in row['695'] else row['695'],
        axis=1
    ), desc='Dodawanie Literature 2')
    
    # 2b. Sprawdzenie tytułu i autora dla "Literature 2"
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    literature_titles_authors = set(zip(
        df[df['695'].apply(lambda x: 'Literature' in x)]['Title'],
        df[df['695'].apply(lambda x: 'Literature' in x)]['Author']
    ))
    df['695'] = tqdm(df.apply(
        lambda row: row['695'] + ['Literature 2'] if (row['Record ID'] in df_with_author['Record ID'].values and (row['Title'], row['Author']) in literature_titles_authors and 'Lack of Data' in row['695'] and 'Literature 2' not in row['695']) else row['695'],
        axis=1
    ), desc='Dodawanie Literature 2 na podstawie tytułu i autora')
    
    # 3. Sprawdzenie tytułu i autora dla "Non-Literary 2"
    non_literary_titles_authors = set(zip(
        df[df['695'].apply(lambda x: 'Non-Literary' in x)]['Title'],
        df[df['695'].apply(lambda x: 'Non-Literary' in x)]['Author']
    ))
    df['695'] = tqdm(df.apply(
        lambda row: row['695'] + ['Non-Literary 2'] if (row['Record ID'] in df_with_author['Record ID'].values and (row['Title'], row['Author']) in non_literary_titles_authors and 'Lack of Data' in row['695'] and 'Non-Literary 2' not in row['695']) else row['695'],
        axis=1
    ), desc='Dodawanie Non-Literary 2')
    
    # 4. Czech lub World Literature 2 na podstawie pola 041h
    df['690'] = tqdm(df.apply(
        lambda row: row['690'] + ['Czech Literature 2'] if 'Literature' in row['695'] and len(row['690']) == 0 and 'cze' in row['Field_041h'].lower() else (
            row['690'] + ['World Literature 2'] if 'Literature' in row['695'] and len(row['690']) == 0 and 'cze' not in row['Field_041h'].lower() else row['690']
        ),
        axis=1
    ), desc='Dodawanie Czech lub World Literature 2')
    
    # 5. Czech Literature na podstawie pola 080 zawierającego "885-*"
    df['690'] = tqdm(df.apply(
        lambda row: row['690'] + ['Czech Literature'] if '885-' in row['Field_080a'] and 'Czech Literature' not in row['690'] else row['690'],
        axis=1
    ), desc='Dodawanie Czech Literature')
    
    # Aktualizacja rekordów MARC na podstawie zaktualizowanego df
    df.set_index('Record ID', inplace=True)
    
    # Przetwarzanie rekordów MARC w strumieniu
    for record in tqdm(marc_records, desc='Aktualizowanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id or record_id not in df.index:
            marc_writer.write(record)
            continue

        # Pobranie wiersza z DataFrame
        current_row = df.loc[record_id]
        
        # Pobranie zaktualizowanych list wartości
        new_695_values = current_row['695'] if isinstance(current_row['695'], list) else []
        new_690_values = current_row['690'] if isinstance(current_row['690'], list) else []
        
        # Aktualizacja pola 695
        if new_695_values:
            record.remove_fields('695')
            for val in new_695_values:
                record.add_field(
                    Field(
                        tag='695',
                        indicators=[' ', ' '],
                        subfields=[Subfield(code='a', value=val)]
                    )
                )
        
        # Aktualizacja pola 690
        if new_690_values:
            record.remove_fields('690')
            for val in new_690_values:
                record.add_field(
                    Field(
                        tag='690',
                        indicators=[' ', ' '],
                        subfields=[Subfield(code='a', value=val)]
                    )
                )
        
        # Zapisanie zaktualizowanego rekordu
        marc_writer.write(record)
    
    # Usunięcie tymczasowych kolumn
    df.drop(columns=['contains_literary_genre'], inplace=True)
    
    # Konwersja wartości '695' i '690' z powrotem na stringi oddzielone przecinkami
    df['695'] = df['695'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df['690'] = df['690'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return df
def process_new_conditions(df, marc_records, marc_writer):
    """
    Przetwarza rekordy MARC, aktualizując pola 695 i 690 na podstawie warunków określonych w DataFrame.
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Record ID', '695', '690'.
    - marc_records: lista rekordów MARC do przetworzenia.
    - marc_writer: obiekt do zapisywania zaktualizowanych rekordów MARC.
    
    Zwraca:
    - Zaktualizowany DataFrame i zapisuje rekordy MARC
    """
    # Przygotowanie listy danych z rekordów MARC
    records_data = []
    for record in tqdm(marc_records, desc='Odczytywanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
        
        # Pobieranie wartości z pól MARC
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')).strip()
        field_245b = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')).strip()
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')).strip()
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h')).strip()
        field_080a = ';'.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')).strip()

        # Dodawanie danych do listy
        records_data.append({
            'Record ID': record_id,
            'Title': field_245a,
            'Subtitle': field_245b,
            'Author': field_100,
            'Field_041h': field_041h,
            'Field_080a': field_080a
        })
    
    # Konwersja listy danych na DataFrame
    records_df = pd.DataFrame(records_data)
    
    # Rozdzielenie wielokrotnych wartości w kolumnach '695' i '690' na listy
    df['695'] = df['695'].apply(lambda x: [val.strip() for val in x.split(',')] if pd.notnull(x) else [])
    df['690'] = df['690'].apply(lambda x: [val.strip() for val in x.split(',')] if pd.notnull(x) else [])
    
    # Połączenie DataFrame z danymi z rekordów MARC
    df = pd.merge(df, records_df, on='Record ID', how='left')
    
    # 1. Zamiana "Unidentified, Non-Literary" na "Lack of Data"
    df['695'] = df['695'].apply(lambda x: ['Lack of Data' if val == "Unidentified, Non-Literary" else val for val in x])
    
    # 2a. Oznaczanie "Literature 2" na podstawie gatunków literackich w 245b
    literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk",
                       "obraz", "drama", "bás", "verš", "jednán"]
    df['contains_literary_genre'] = tqdm(df['Subtitle'].str.lower().apply(
        lambda x: any(genre in x for genre in literary_genres) if pd.notnull(x) else False), desc='Sprawdzanie gatunków literackich')
    
    # Dodawanie "Literature 2" dla rekordów z "Lack of Data" i zawierających gatunek literacki
    df['695'] = tqdm(df.apply(
        lambda row: row['695'] + ['Literature 2'] if row['contains_literary_genre'] and 'Lack of Data' in row['695'] and 'Literature 2' not in row['695'] else row['695'],
        axis=1
    ), desc='Dodawanie Literature 2')
    
    # 2b. Sprawdzenie tytułu i autora dla "Literature 2"
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    literature_titles_authors = set(zip(
        df[df['695'].apply(lambda x: 'Literature' in x)]['Title'],
        df[df['695'].apply(lambda x: 'Literature' in x)]['Author']
    ))
    df['695'] = tqdm(df.apply(
        lambda row: row['695'] + ['Literature 2'] if (row['Record ID'] in df_with_author['Record ID'].values and (row['Title'], row['Author']) in literature_titles_authors and 'Lack of Data' in row['695'] and 'Literature 2' not in row['695']) else row['695'],
        axis=1
    ), desc='Dodawanie Literature 2 na podstawie tytułu i autora')
    
    # 3. Sprawdzenie tytułu i autora dla "Non-Literary 2"
    non_literary_titles_authors = set(zip(
        df[df['695'].apply(lambda x: 'Non-Literary' in x)]['Title'],
        df[df['695'].apply(lambda x: 'Non-Literary' in x)]['Author']
    ))
    df['695'] = tqdm(df.apply(
        lambda row: row['695'] + ['Non-Literary 2'] if (row['Record ID'] in df_with_author['Record ID'].values and (row['Title'], row['Author']) in non_literary_titles_authors and 'Lack of Data' in row['695'] and 'Non-Literary 2' not in row['695']) else row['695'],
        axis=1
    ), desc='Dodawanie Non-Literary 2')
    
    # 4. Czech lub World Literature 2 na podstawie pola 041h
    df['690'] = tqdm(df.apply(
        lambda row: row['690'] + ['Czech Literature 2'] if 'Literature' in row['695'] and len(row['690']) == 0 and 'cze' in row['Field_041h'].lower() else (
            row['690'] + ['World Literature 2'] if 'Literature' in row['695'] and len(row['690']) == 0 and 'cze' not in row['Field_041h'].lower() else row['690']
        ),
        axis=1
    ), desc='Dodawanie Czech lub World Literature 2')
    
    # 5. Czech Literature na podstawie pola 080 zawierającego "885-*"
    df['690'] = tqdm(df.apply(
        lambda row: row['690'] + ['Czech Literature'] if '885-' in row['Field_080a'] and 'Czech Literature' not in row['690'] else row['690'],
        axis=1
    ), desc='Dodawanie Czech Literature')
    
    # Aktualizacja rekordów MARC na podstawie zaktualizowanego df
    df.set_index('Record ID', inplace=True)
    
    # Przetwarzanie rekordów MARC w strumieniu
    for record in tqdm(marc_records, desc='Aktualizowanie rekordów'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id or record_id not in df.index:
            marc_writer.write(record)
            continue

        # Pobranie wiersza z DataFrame
        current_row = df.loc[record_id]
        
        # Pobranie zaktualizowanych list wartości
        new_695_values = current_row['695'] if isinstance(current_row['695'], list) else []
        new_690_values = current_row['690'] if isinstance(current_row['690'], list) else []
        
        # Aktualizacja pola 695
        if new_695_values:
            record.remove_fields('695')
            for val in new_695_values:
                record.add_field(
                    Field(
                        tag='695',
                        indicators=[' ', ' '],
                        subfields=[Subfield(code='a', value=val)]
                    )
                )
        
        # Aktualizacja pola 690
        if new_690_values:
            record.remove_fields('690')
            for val in new_690_values:
                record.add_field(
                    Field(
                        tag='690',
                        indicators=[' ', ' '],
                        subfields=[Subfield(code='a', value=val)]
                    )
                )
        
        # Zapisanie zaktualizowanego rekordu
        marc_writer.write(record)
    
    # Usunięcie tymczasowych kolumn
    df.drop(columns=['contains_literary_genre'], inplace=True)
    
    # Konwersja wartości '695' i '690' z powrotem na stringi oddzielone przecinkami
    df['695'] = df['695'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df['690'] = df['690'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    
    return df
# Ta funkcja zawiera poprawki, które powinny rozwiązać problemy związane z niewłaściwą zamianą danych oraz złym dopasowaniem warunków.
# Skupiono się na poprawnym łączeniu danych, lepszej obsłudze warunków oraz ujednoliceniu operacji na polach 695 i 690.
# Główna część skryptu
df = pd.read_excel('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_data.xlsx')
with open('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_records.mrc', 'rb') as marc_fh, open('updated_identified_records.mrc', 'wb') as marc_output_fh:
    marc_records = list(MARCReader(marc_fh))
    marc_writer = MARCWriter(marc_output_fh)
    
    updated_df = process_new_conditions(df, marc_records, marc_writer)
    marc_writer.close()

# Zapisanie zaktualizowanego DataFrame
with pd.ExcelWriter('update_identified_data.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    workbook.strings_to_urls = False  # Wyłączenie konwersji ciągów na URL-e
    updated_df.to_excel(writer, index=False)
import pandas as pd
from tqdm import tqdm

def lack_of_data_first_step(df):
    """
    Pierwszy krok przetwarzania rekordów - zamiana "Unidentified, Non-Literary" na "Lack of Data".
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumnę '695'.
    
    Zwraca:
    - Zaktualizowany DataFrame
    """
    # 1. Zamiana "Unidentified, Non-Literary" na "Lack of Data" - bez podziału na listy
    df['695'] = df['695'].replace('Unidentified, Non-Literary', 'Lack of Data')
    
    return df



def add_literature_2(df, records_data):
    """
    Drugi krok przetwarzania rekordów - oznaczanie "Literature 2" na podstawie gatunków literackich w kolumnie 'Subtitle'.
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Subtitle' i '695'.
    - records_data: DataFrame z danymi wyciągniętymi z rekordów MARC, zawierający kolumny 'Record ID', 'Subtitle'.
    
    Zwraca:
    - Zaktualizowany DataFrame
    """
    # Sprawdzenie, czy records_data jest DataFrame i nie jest pusty
    if not isinstance(records_data, pd.DataFrame) or records_data.empty:
        print("Warning: records_data is not a valid DataFrame or is empty. Skipping merge operation.")
        return df
    
    # Połączenie DataFrame z danymi z rekordów MARC na podstawie 'Record ID'
    df = pd.merge(df, records_data[['Record ID', 'Subtitle']], on='Record ID', how='left')
    
    # 2a. Oznaczanie "Literature 2" na podstawie gatunków literackich w 'Subtitle'
    literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk",
                       "obraz", "drama", "bás", "verš", "jednán"]
    
    # Tworzenie maski, aby zidentyfikować rekordy zawierające gatunki literackie
    mask_literary_genre = df['Subtitle'].str.lower().str.contains('|'.join(literary_genres), na=False)
    
    # Tworzenie maski dla rekordów z "Lack of Data" w kolumnie '695'
    mask_lack_of_data = df['695'].str.contains('Lack of Data', na=False)
    
    # Łączenie masek, aby zidentyfikować rekordy do aktualizacji
    final_mask = mask_literary_genre & mask_lack_of_data
    
    # Aktualizacja kolumny '695' dla zidentyfikowanych rekordów - wektorowo
    df.loc[final_mask, '695'] = df.loc[final_mask, '695'] + ', Literature 2'
    
    return df




# Odczyt rekordów MARC i przygotowanie danych do dalszego przetwarzania
def add_literature_2_based_on_title_author(df, records_data):
    """
    Trzeci krok przetwarzania rekordów - oznaczanie "Literature 2" na podstawie tytułu ('Title') i autora ('Author').
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Title', 'Author', '695'.
    - records_data: DataFrame z danymi wyciągniętymi z rekordów MARC, zawierający kolumny 'Record ID', 'Title', 'Author'.
    
    Zwraca:
    - Zaktualizowany DataFrame
    """
    # Połączenie DataFrame z danymi z rekordów MARC na podstawie 'Record ID', uwzględniając 'Title' i 'Author'
    df = pd.merge(df, records_data[['Record ID', 'Title', 'Author']], on='Record ID', how='left')
    
    # Filtrowanie rekordów z wypełnionym polem 'Author'
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    
    # Tworzenie zbioru tytułów i autorów dla rekordów z 'Literature' w kolumnie '695', korzystając już z df_with_author
    df_with_literature = df_with_author[df_with_author['695'].str.contains('Literature', na=False)]
    literature_titles_authors = set(zip(df_with_literature['Title'], df_with_literature['Author']))
    
    # Tworzenie maski dla rekordów z "Lack of Data" w kolumnie '695'
    mask_lack_of_data = df_with_author['695'].str.contains('Lack of Data', na=False)
    
    # Tworzenie maski dla rekordów, których tytuł i autor są w zbiorze literackim
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    mask_title_author_in_literature = df_with_author['title_author_tuple'].isin(literature_titles_authors)
    
    # Łączenie masek, aby zidentyfikować rekordy do aktualizacji
    final_mask = mask_lack_of_data & mask_title_author_in_literature

    # Aktualizacja kolumny '695' dla zidentyfikowanych rekordów - wektorowo
    df_with_author.loc[final_mask, '695'] = df_with_author.loc[final_mask, '695'] + ', Literature 2'

    # Aktualizacja głównego DataFrame
    df.update(df_with_author)

    return df


def add_non_literary_2_based_on_title_author(df, records_data):
    """
    Czwarty krok przetwarzania rekordów - oznaczanie "Non-Literary 2" na podstawie tytułu ('Title') i autora ('Author') dla rekordów z 'Lack of Data'.
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Title', 'Author', '695'.
    - records_data: DataFrame z danymi wyciągniętymi z rekordów MARC, zawierający kolumny 'Record ID', 'Title', 'Author'.
    
    Zwraca:
    - Zaktualizowany DataFrame
    """
    # Połączenie DataFrame z danymi z rekordów MARC na podstawie 'Record ID', uwzględniając 'Title' i 'Author'
    df = pd.merge(df, records_data[['Record ID', 'Title', 'Author']], on='Record ID', how='left')
    
    # Filtrowanie rekordów z wypełnionym polem 'Author'
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    
    # Tworzenie zbioru tytułów i autorów dla rekordów z 'Non-Literary' w kolumnie '695', korzystając już z df_with_author
    df_with_non_literary = df_with_author[df_with_author['695'].str.contains('Non-Literary', na=False)]
    non_literary_titles_authors = set(zip(df_with_non_literary['Title'], df_with_non_literary['Author']))
    
    # Tworzenie maski dla rekordów z "Lack of Data" w kolumnie '695'
    mask_lack_of_data = df_with_author['695'].str.contains('Lack of Data', na=False)
    
    # Tworzenie maski dla rekordów, których tytuł i autor są w zbiorze non-literary
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    mask_title_author_in_non_literary = df_with_author['title_author_tuple'].isin(non_literary_titles_authors)
    
    # Łączenie masek, aby zidentyfikować rekordy do aktualizacji
    final_mask = mask_lack_of_data & mask_title_author_in_non_literary

    # Aktualizacja kolumny '695' dla zidentyfikowanych rekordów - wektorowo
    df_with_author.loc[final_mask, '695'] = df_with_author.loc[final_mask, '695'] + ', Non-Literary 2'

    # Aktualizacja głównego DataFrame
    df.update(df_with_author)

    return df

# Odczyt rekordów MARC i przygotowanie danych do dalszego przetwarzania
records_data = []
with open('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_records.mrc', 'rb') as marc_fh:
    for record in tqdm(MARCReader(marc_fh), desc='Odczytywanie rekordów MARC'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
        
        # Pobieranie wartości z pól MARC
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')).strip()
        field_245b = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')).strip()
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')).strip()
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h')).strip()
        field_080a = ';'.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')).strip()
        
        # Dodawanie danych do listy
        records_data.append({
            'Record ID': record_id,
            'Title': field_245a,
            'Subtitle': field_245b,
            'Author': field_100,
            'Field_041h': field_041h,
            'Field_080a': field_080a
        })

# Konwersja listy danych na DataFrame
records_data = pd.DataFrame(records_data)

# Główna część skryptu - przetwarzanie pliku Excel
df = pd.read_excel('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_data.xlsx')

# Przetwarzanie pierwszego kroku - zamiana "Unidentified, Non-Literary" na "Lack of Data"
df = lack_of_data_first_step(df)
print(df.head())
print(df['695'].head())


# Przetwarzanie drugiego kroku - dodawanie "Literature 2" na podstawie gatunków literackich z rekordów MARC
df = add_literature_2(df, records_data)

# Podgląd kilku pierwszych wierszy kolumny '695' DataFrame po dodaniu "Literature 2"
print("Podgląd danych po dodaniu Literature 2 (kolumna '695'):")
print(df['695'].head())
literature_2_mask = df['695'].str.contains('Literature 2', na=False)
literature_2_count = literature_2_mask.sum()
print("Wiersze z wartością 'Literature 2' w kolumnie '695':")
print(df[literature_2_mask].head(20))
df = add_literature_2_based_on_title_author(df, records_data)

literature_2_mask = df['695'].str.contains('Literature 2', na=False)
literature_2_count = literature_2_mask.sum()
df = add_non_literary_2_based_on_title_author(df, records_data)



def lack_of_data_first_step(df):
    """
    Pierwszy krok przetwarzania rekordów - zamiana "Unidentified, Non-Literary" na "Lack of Data".
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumnę '695'.
    
    Zwraca:
    - Zaktualizowany DataFrame
    """
    # 1. Zamiana "Unidentified, Non-Literary" na "Lack of Data" - bez podziału na listy
    df['695'] = df['695'].replace('Unidentified, Non-Literary', 'Lack of Data')
    
    return df

def add_literature_2(df, records_data):
    """
    Drugi krok przetwarzania rekordów - oznaczanie "Literature 2" na podstawie gatunków literackich w kolumnie 'Subtitle'.
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Subtitle' i '695'.
    - records_data: DataFrame z danymi wyciągniętymi z rekordów MARC, zawierający kolumny 'Record ID', 'Subtitle'.
    
    Zwraca:
    - Zaktualizowany DataFrame
    """
    # Połączenie DataFrame z danymi z rekordów MARC na podstawie 'Record ID'
    df = pd.merge(df, records_data[['Record ID', 'Subtitle']], on='Record ID', how='left')
    
    # 2a. Oznaczanie "Literature 2" na podstawie gatunków literackich w 'Subtitle'
    literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk",
                       "obraz", "drama", "bás", "verš", "jednán"]
    
    # Tworzenie maski, aby zidentyfikować rekordy zawierające gatunki literackie
    mask_literary_genre = df['Subtitle'].str.lower().str.contains('|'.join(literary_genres), na=False)
    
    # Tworzenie maski dla rekordów z "Lack of Data" w kolumnie '695'
    mask_lack_of_data = df['695'].str.contains('Lack of Data', na=False)
    
    # Łączenie masek, aby zidentyfikować rekordy do aktualizacji
    final_mask = mask_literary_genre & mask_lack_of_data
    
    # Aktualizacja kolumny '695' dla zidentyfikowanych rekordów - wektorowo
    df.loc[final_mask, '695'] = df.loc[final_mask, '695'] + ', Literature 2'
    
    # Usunięcie kolumny 'Subtitle', ponieważ nie jest już potrzebna
    df.drop(columns=['Subtitle'], inplace=True)
    
    return df

def add_literature_2_based_on_title_author(df, records_data):
    """
    Trzeci krok przetwarzania rekordów - oznaczanie "Literature 2" na podstawie tytułu ('Title') i autora ('Author').
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Title', 'Author', '695'.
    - records_data: DataFrame z danymi wyciągniętymi z rekordów MARC, zawierający kolumny 'Record ID', 'Title', 'Author'.
    
    Zwraca:
    - Zaktualizowany DataFrame
    """
    # Połączenie DataFrame z danymi z rekordów MARC na podstawie 'Record ID', uwzględniając 'Title' i 'Author'
    df = pd.merge(df, records_data[['Record ID', 'Title', 'Author']], on='Record ID', how='left')
    
    # Filtrowanie rekordów z wypełnionym polem 'Author'
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    
    # Tworzenie zbioru tytułów i autorów dla rekordów z 'Literature' w kolumnie '695', korzystając już z df_with_author
    df_with_literature = df_with_author[df_with_author['695'].str.contains('Literature', na=False)]
    literature_titles_authors = set(zip(df_with_literature['Title'], df_with_literature['Author']))
    
    # Tworzenie maski dla rekordów z "Lack of Data" w kolumnie '695'
    mask_lack_of_data = df_with_author['695'].str.contains('Lack of Data', na=False)
    
    # Tworzenie maski dla rekordów, których tytuł i autor są w zbiorze literackim
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    mask_title_author_in_literature = df_with_author['title_author_tuple'].isin(literature_titles_authors)
    
    # Łączenie masek, aby zidentyfikować rekordy do aktualizacji
    final_mask = mask_lack_of_data & mask_title_author_in_literature

    # Aktualizacja kolumny '695' dla zidentyfikowanych rekordów - wektorowo
    df_with_author.loc[final_mask, '695'] = df_with_author.loc[final_mask, '695'] + ', Literature 2'

    # Aktualizacja głównego DataFrame
    df.update(df_with_author)

    return df

def add_non_literary_2_based_on_title_author(df, records_data):
    """
    Czwarty krok przetwarzania rekordów - oznaczanie "Non-Literary 2" na podstawie tytułu ('Title') i autora ('Author') dla rekordów z 'Lack of Data'.
    
    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny 'Title', 'Author', '695'.
    - records_data: DataFrame z danymi wyciągniętymi z rekordów MARC, zawierający kolumny 'Record ID', 'Title', 'Author'.
    
    Zwraca:
    - Zaktualizowany DataFrame
    """
    # Połączenie DataFrame z danymi z rekordów MARC na podstawie 'Record ID', uwzględniając 'Title' i 'Author'
    df = pd.merge(df, records_data[['Record ID', 'Title', 'Author']], on='Record ID', how='left')
    
    # Filtrowanie rekordów z wypełnionym polem 'Author'
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    
    # Tworzenie zbioru tytułów i autorów dla rekordów z 'Non-Literary' w kolumnie '695'
    df_with_non_literary = df_with_author[df_with_author['695'].str.contains('Non-Literary', na=False)]
    non_literary_titles_authors = set(zip(df_with_non_literary['Title'], df_with_non_literary['Author']))
    
    # Tworzenie maski dla rekordów z "Lack of Data" w kolumnie '695'
    mask_lack_of_data = df_with_author['695'].str.contains('Lack of Data', na=False)
    
    # Tworzenie maski dla rekordów, których tytuł i autor są w zbiorze non-literary
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    mask_title_author_in_non_literary = df_with_author['title_author_tuple'].isin(non_literary_titles_authors)
    
    # Łączenie masek, aby zidentyfikować rekordy do aktualizacji
    final_mask = mask_lack_of_data & mask_title_author_in_non_literary

    # Aktualizacja kolumny '695' dla zidentyfikowanych rekordów - wektorowo
    df_with_author.loc[final_mask, '695'] = df_with_author.loc[final_mask, '695'] + ', Non-Literary 2'

    # Aktualizacja głównego DataFrame
    df.update(df_with_author)

    return df

# Odczyt rekordów MARC i przygotowanie danych do dalszego przetwarzania
records_data = []
with open('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_records.mrc', 'rb') as marc_fh:
    for record in tqdm(MARCReader(marc_fh), desc='Odczytywanie rekordów MARC'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
        
        # Pobieranie wartości z pól MARC
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')).strip()
        field_245b = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')).strip()
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')).strip()
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h')).strip()
        field_080a = ';'.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')).strip()
        
        # Dodawanie danych do listy
        records_data.append({
            'Record ID': record_id,
            'Title': field_245a,
            'Subtitle': field_245b,
            'Author': field_100,
            'Field_041h': field_041h,
            'Field_080a': field_080a
        })

# Konwersja listy danych na DataFrame
records_data = pd.DataFrame(records_data)
df = pd.read_excel('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_data.xlsx')
df = lack_of_data_first_step(df)
print(df.head())
print(df['695'].head())


# Przetwarzanie drugiego kroku - dodawanie "Literature 2" na podstawie gatunków literackich z rekordów MARC
df = add_literature_2(df, records_data)

# Podgląd kilku pierwszych wierszy kolumny '695' DataFrame po dodaniu "Literature 2"

literature_2_mask = df['695'].str.contains('Literature 2', na=False)
literature_2_count = literature_2_mask.sum()
print("Wiersze z wartością 'Literature 2' w kolumnie '695':")
print(df[literature_2_mask].head(20))
df = add_literature_2_based_on_title_author(df, records_data)

literature_2_mask = df['695'].str.contains('Literature 2', na=False)
literature_2_count = literature_2_mask.sum()
df = add_non_literary_2_based_on_title_author(df, records_data)

# Główna część skryptu - przetwarzanie pliku Excel oraz rekordów MARC
df = pd.read_excel('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_data.xlsx')

# Przetwarzanie pierwszego kroku - zamiana "Unidentified, Non-Literary" na "Lack of Data"
df = lack_of_data_first_step(df)

# Przetwarzanie drugiego kroku - dodawanie "Literature 2" na podstawie gatunków literackich z rekordów MARC
df = add_literature_2(df, records_data)

# Przetwarzanie trzeciego kroku - dodawanie "Literature 2" na podstawie tytułu i autora
df = add_literature_2_based_on_title_author(df, records_data)

# Przetwarzanie czwartego kroku - dodawanie "Non-Literary 2" na podstawie tytułu i autora
df = add_non_literary_2_based_on_title_author(df, records_data)



#%% Najlepsze 29.09.2024


import pandas as pd
from tqdm import tqdm
from pymarc import Field, Subfield, MARCReader, MARCWriter

def lack_of_data_first_step(df):
    # Zamiana "Unidentified, Non-Literary" na "Lack of Data"
    df['695'] = df['695'].replace('Unidentified, Non-Literary', 'Lack of Data')
    return df

def add_literature_2(df):
    literary_genres = ["poesie", "poezie", "próz", "román", "novel", "novella", "povídk", "obraz", "drama", "bás", "verš", "jednán"]
    mask_literary_genre = df['Subtitle'].str.lower().str.contains('|'.join(literary_genres), na=False)
    mask_lack_of_data = df['695'].str.contains('Lack of Data', na=False)
    final_mask = mask_literary_genre & mask_lack_of_data
    #df.loc[final_mask, '695'] = (df.loc[final_mask, '695'] + ', Literature 2').str.strip(', ')
    df.loc[final_mask, '695'] = 'Literature 2'
    return df

def add_literature_2_based_on_title_author(df):
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    df_with_literature = df_with_author[df_with_author['695'].str.contains('Literature', na=False)]
    literature_titles_authors = set(zip(df_with_literature['Title'], df_with_literature['Author']))
    mask_lack_of_data = df_with_author['695'].str.contains('Lack of Data', na=False)
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    mask_title_author_in_literature = df_with_author['title_author_tuple'].isin(literature_titles_authors)
    final_mask = mask_lack_of_data & mask_title_author_in_literature
    #df_with_author.loc[final_mask, '695'] = (df_with_author.loc[final_mask, '695'] + ', Literature 2').str.strip(', ')
    df_with_author.loc[final_mask, '695'] = 'Literature 2'
    df.update(df_with_author)
    return df

def add_non_literary_2_based_on_title_author(df):
    df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
    df_with_non_literary = df_with_author[df_with_author['695'].str.contains('Non-Literary', na=False)]
    non_literary_titles_authors = set(zip(df_with_non_literary['Title'], df_with_non_literary['Author']))
    mask_lack_of_data = df_with_author['695'].str.contains('Lack of Data', na=False)
    df_with_author['title_author_tuple'] = list(zip(df_with_author['Title'], df_with_author['Author']))
    mask_title_author_in_non_literary = df_with_author['title_author_tuple'].isin(non_literary_titles_authors)
    final_mask = mask_lack_of_data & mask_title_author_in_non_literary
    #df_with_author.loc[final_mask, '695'] = (df_with_author.loc[final_mask, '695'] + ', Non-Literary 2').str.strip(', ')
    df_with_author.loc[final_mask, '695'] = 'Non-Literary 2'
    df.update(df_with_author)
    
    return df
def add_czech_or_world_literature_2(df):
    """
    Piąty krok przetwarzania rekordów - oznaczanie 'Czech Literature 2' lub 'World Literature 2'
    na podstawie wartości w polu '041h' oraz wcześniejszej klasyfikacji 'Literature'.

    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny '695', '690', 'Field_041h'.

    Zwraca:
    - Zaktualizowany DataFrame
    """
    # Tworzenie maski dla rekordów oznaczonych jako 'Literature' w kolumnie '695'
    mask_literature = df['695'].str.contains('Literature', na=False)

    # Tworzenie maski dla rekordów, które nie mają jeszcze wartości w kolumnie '690' (uwzględniając NaN)
    mask_missing_690 = df['690'].isna() | (df['690'].str.strip() == '')

    # Tworzenie maski dla rekordów, które w polu 'Field_041h' zawierają 'cze'
    mask_czech = df['Field_041h'].str.lower().str.contains('cze', na=False)

    # Tworzenie maski dla rekordów, które w polu 'Field_041h' zawierają języki inne niż 'cze'
    mask_non_czech = df['Field_041h'].notna() & ~mask_czech

    # Aktualizacja kolumny '690' dla rekordów oznaczonych jako Czech Literature 2
    df.loc[mask_literature & mask_missing_690 & mask_czech, '690'] = df.loc[mask_literature & mask_missing_690 & mask_czech, '690'].fillna('') + ', Czech Literature 2'

    # Aktualizacja kolumny '690' dla rekordów oznaczonych jako World Literature 2
    df.loc[mask_literature & mask_missing_690 & mask_non_czech, '690'] = df.loc[mask_literature & mask_missing_690 & mask_non_czech, '690'].fillna('') + ', World Literature 2'

    # Usunięcie zbędnych przecinków i spacji z początku i końca wartości w kolumnie '690'
    df['690'] = df['690'].str.strip(', ')

    return df
def add_czech_literature(df):
    """
    Ostatni krok przetwarzania rekordów - oznaczanie 'Czech Literature' dla rekordów, które mają '885-' w polu '080'
    oraz nie są już oznaczone jako 'Czech Literature' w kolumnie '690'.

    Parametry:
    - df: pandas DataFrame zawierający dane do aktualizacji. Powinien zawierać kolumny '690', 'Field_080a'.

    Zwraca:
    - Zaktualizowany DataFrame
    """
    # Tworzenie maski dla rekordów, które zawierają '885-' w polu 'Field_080a'
    mask_885 = df['Field_080a'].str.contains('885-', na=False)
    
    # Tworzenie maski dla rekordów, które nie zawierają 'Czech Literature' w kolumnie '690'
    mask_no_czech_literature = ~df['690'].str.contains('Czech Literature', na=False, case=False)

    # Łączenie masek, aby zidentyfikować rekordy do aktualizacji
    final_mask = mask_885 & mask_no_czech_literature

    # Aktualizacja kolumny '690' dla zidentyfikowanych rekordów - dodawanie 'Czech Literature'
    df.loc[final_mask, '690'] = (
        df.loc[final_mask, '690'].fillna('') + ', Czech Literature 2'
    ).str.strip(', ')
    df['690'] = df['690'].str.strip(', ')

    return df


# Odczyt rekordów MARC
records_data = []
with open('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_records.mrc', 'rb') as marc_fh:
    for record in tqdm(MARCReader(marc_fh), desc='Odczytywanie rekordów MARC'):
        record_id = record['001'].value() if record['001'] else None
        if not record_id:
            continue
        field_245a = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')).strip()
        field_245b = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields('b')).strip()
        field_100 = ', '.join(subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')).strip()
        field_041h = ', '.join(subfield for field in record.get_fields('041') for subfield in field.get_subfields('h')).strip()
        field_080a = ';'.join(subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')).strip()
        records_data.append({
            'Record ID': record_id,
            'Title': field_245a,
            'Subtitle': field_245b,
            'Author': field_100,
            'Field_041h': field_041h,
            'Field_080a': field_080a
        })

records_data = pd.DataFrame(records_data)

# Odczyt danych z Excela i połączenie z records_data
df = pd.read_excel('D:/Nowa_praca/czech_works/NKC_12_08_2024/identified_data.xlsx')
df = pd.merge(df, records_data, on='Record ID', how='left')
df = lack_of_data_first_step(df)
df = add_literature_2(df)
df = add_literature_2_based_on_title_author(df)
literature_2_mask = df['695'].str.contains('Literature 2', na=False)
literature_2_count = literature_2_mask.sum()
df = add_non_literary_2_based_on_title_author(df)
NONliterature_2_mask = df['695'].str.contains('Non-Literary 2', na=False).sum()
df = add_czech_or_world_literature_2(df)
czech_2_count = df['690'].str.contains('Czech Literature 2', na=False).sum()
world_2_count = df['690'].str.contains('World Literature 2', na=False).sum()
df = add_czech_literature(df)
czech_3_count = df['690'].str.contains('Czech Literature 2', na=False).sum()


with pd.ExcelWriter('update_identified_data.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    workbook.strings_to_urls = False  # Wyłączenie konwersji ciągów na URL-e
    df.to_excel(writer, index=False)
    
def find_authors_in_both_czech_and_world_literature(df):
    """
    Wyszukuje autorów, którzy mają co najmniej jedno dzieło w kategoriach "Czech Literature" lub "Czech Literature 2",
    oraz jedno w kategoriach "World Literature" lub "World Literature 2".

    Parametry:
    - df: pandas DataFrame zawierający dane do przetwarzania, powinien zawierać kolumny 'Author' i '690'.

    Zwraca:
    - Listę autorów spełniających te kryteria.
    """
    # Tworzenie masek dla kategorii "Czech Literature" i "World Literature"
    mask_czech = df['690'].str.contains('Czech Literature|Czech Literature 2', na=False)
    mask_world = df['690'].str.contains('World Literature|World Literature 2', na=False)
    
    # Tworzenie zestawu autorów, którzy mają dzieła w każdej z tych kategorii
    authors_in_czech = set(df[mask_czech]['Author'].dropna().unique())
    authors_in_world = set(df[mask_world]['Author'].dropna().unique())
    
    # Znalezienie autorów obecnych w obu zestawach
    authors_in_both_categories = authors_in_czech.intersection(authors_in_world)
    
    # Zamiana na listę i zwrócenie wyników
    return list(authors_in_both_categories)

# Przykład użycia:
authors_with_errors = find_authors_in_both_czech_and_world_literature(df)
print("Autorzy z dziełami zarówno w Czech Literature, jak i World Literature:", authors_with_errors)
authors_with_errors = list(set(authors_with_errors))  # opcjonalne, usunięcie duplikatów
authors_df = pd.DataFrame(authors_with_errors, columns=['Author'])
with pd.ExcelWriter('authors_with_errors.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    workbook.strings_to_urls = False  # Wyłączenie konwersji ciągów na URL-e
    authors_df.to_excel(writer, index=False)







#Łaczenie plikow mrc i mrk\
from pymarc import MARCReader, MARCWriter, TextWriter

# Ścieżki do dwóch plików MARC, które chcemy połączyć
input_file_1 = 'D:/Nowa_praca/08.02.2024_marki/fi_fennica_08-02-2024.mrc'
input_file_2 = 'D:/Nowa_praca/08.02.2024_marki/fi_arto__08-02-2024.mrc'

# Ścieżki do wyjściowych plików w formatach MRK i MRC
output_mrk = 'finnish_records.mrk'
output_mrc = 'finnish_records.mrc'

# Funkcja do odczytu i zapisu rekordów bezpośrednio podczas iteracji
def process_marc_files(input_files, output_mrk, output_mrc):
    # Otwórz pliki wyjściowe
    with open(output_mrk, 'w', encoding='utf-8') as mrk_file, open(output_mrc, 'wb') as mrc_file:
        # Inicjalizacja writerów do formatu MRK (tekstowy) i MRC (binarny)
        text_writer = TextWriter(mrk_file)
        marc_writer = MARCWriter(mrc_file)

        # Przetwarzanie każdego pliku MARC
        for file_path in input_files:
            with open(file_path, 'rb') as marc_file:
                reader = MARCReader(marc_file)
                for record in tqdm(reader):
                    # Zapisz rekord bezpośrednio do obu plików
                    text_writer.write(record)
                    marc_writer.write(record)

        # Zamknięcie writerów po zakończeniu
        text_writer.close()
        marc_writer.close()

# Lista plików do połączenia
input_files = [input_file_1, input_file_2]

# Wywołanie funkcji przetwarzającej pliki
process_marc_files(input_files, output_mrk, output_mrc)

print(f'Połączone pliki zostały zapisane jako {output_mrk} i {output_mrc}')









#%% Podejcie iteracyjne vs wektor:
mask_title_author_in_literature = df.apply(lambda row: (row['Title'], row['Author']) in literature_titles_authors, axis=1)
#wektory:
df_with_author = df[df['Author'].notnull() & (df['Author'].str.strip() != '')].copy()
df_with_literature = df_with_author[df_with_author['695'].str.contains('Literature', na=False)]
literature_titles_authors = set(zip(df_with_literature['Title'], df_with_literature['Author']))

