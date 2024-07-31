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

def filter_marcxml(file_path, filter_df, writer_marc, writer_mrk):
    records = []
    processed_count = 0
    matching_count = 0
    literary_author_ids = set()
    literature_authors = []  # Lista do przechowywania autorów literatury
    literature_record_ids = set()

    # Filtracja wartości z pola 150 na podstawie wartości w polu 080
    filter_literature_080 = set(filter_df[filter_df['080'].str.startswith('821', na=False)]['150'].dropna().unique())
    filter_literary_science_080 = set(filter_df[filter_df['080'].str.startswith('82', na=False) & ~filter_df['080'].str.startswith(('820', '821'), na=False)]['150'].dropna().unique())

    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for record in tqdm(reader, desc="Processing records"):
            try:
                if any('0/9-053.2' in field.get_subfields('a') or 'Literatura pro děti a mládež (naučná)' in field.get_subfields('x') for field in record.get_fields('072')):
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
                field_245_values = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields())
                field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                # CRITERION A: 695
                is_literature = any(val in ['25', '26'] for val in field_072_values) or \
                                any(val.startswith('821') for val in field_080_values) or \
                                any(val in filter_literature_080 for val in field_650_values) or \
                                any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' in val for val in field_080_values)

                is_literary_science = '11' in field_072_values and \
                                      (any(val.startswith('82') and not val.startswith(('820', '821')) for val in field_080_values) or \
                                       any(val in filter_literary_science_080 for val in field_650_values)) or \
                                      any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' not in val for val in field_080_values) or \
                                      any(val.startswith('82.') for val in field_080_values)

                has_fields = field_072_values or field_080_values or field_655_values

                if is_literature or is_literary_science or not has_fields:
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
                    if not has_fields:
                        record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Unidentified')]))

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
                is_other_form = ldr_8 not in ['m', 'a', 'b', 's'] or \
                                any(val.startswith('elektron') or val.startswith('stream') for val in field_655_values) or \
                                any(term in field_245_values for term in ['[zvukový záznam]', '[elektronický zdroj]'])
                
                if is_other_form:
                    record.add_field(Field(tag='691', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Other forms')]))
                else:
                    record.add_field(Field(tag='691', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Printed')]))

                if is_literature or is_literary_science or is_czech_literature or is_world_literature or not has_fields:
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
                        '695': ', '.join(filter(None, ['Literature' if is_literature else '', 'Literary Science' if is_literary_science else '', 'Unidentified' if not has_fields else ''])).strip(', '),
                        '690': ', '.join(filter(None, ['Czech Literature' if is_czech_literature else '', 'World Literature' if is_world_literature else ''])).strip(', '),
                        '691': 'Other forms' if is_other_form else 'Printed'
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
    all_literature_authors = []  # Lista do przechowywania autorów literatury
    all_literature_record_ids = set()

    with open(output_marc21_file, 'wb') as marc_fh, open(output_mrk_file, 'wt', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)

        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
            chunk_df, literary_author_ids, literature_authors_df, literature_record_ids = filter_marcxml(chunk_file_mrc, filter_df, writer_marc, writer_mrk)
            all_records.append(chunk_df)
            all_literary_author_ids.update(literary_author_ids)
            all_literature_authors.append(literature_authors_df)
            all_literature_record_ids.update(literature_record_ids)

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
                        field_245_values = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields())
                        field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                        field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                        # Check if the record belongs to non-literary production of literary authors
                        if any(id_ in all_literary_author_ids for id_ in field_100_7_values + field_700_7_values):
                            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Non-Literary')]))
                            ldr_8 = record.leader[7]
                            is_other_form = ldr_8 not in ['m', 'a', 'b', 's'] or \
                                            any(val.startswith('elektron') or val.startswith('stream') for val in field_655_values) or \
                                            any(term in field_245_values for term in ['[zvukový záznam]', '[elektronický zdroj]'])

                            if is_other_form:
                                record.add_field(Field(tag='691', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Other forms')]))
                            else:
                                record.add_field(Field(tag='691', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Printed')]))

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
                                '691': 'Other forms' if is_other_form else 'Printed'
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
combined_df.to_excel('filtered_combined_23-06.xlsx', index=False)
combined_literature_authors_df=combined_literature_authors_df.drop_duplicates()
combined_literature_authors_df.to_excel('literature_authors.xlsx', index=False)
with pd.ExcelWriter('filtered_combined_23-06.xlsx', engine='xlsxwriter') as writer:
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

# Each sheet in the Excel file will contain up to 1,048,575 rows of data from 'combined_df'.

from pymarc import Record, Field, Subfield

record = Record()
record.add_field(
    Field(
        tag = '245',
        indicators = ['0','1'],
        subfields = [
            Subfield(code='a', value='The pragmatic programmer : '),
            Subfield(code='b', value='from journeyman to master /'),
            Subfield(code='c', value='Andrew Hunt, David Thomas.')
        ]))
field_245_values = ' '.join(field.value() for field in record.get_fields('245'))
f=[]
for field in record.get_fields('245'):
    
    print(field.value())
field_100_values = [subfield for field in record.get_fields('245') for subfield in field.get_subfields('a')]
is_literature = any(val in ['The pragmatic programmer : ','lala'] for val in field_100_values)    


def filter_marcxml(file_path, filter_df, writer_marc, writer_mrk):
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
                field_245_values = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields())
                field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                # CRITERION A: 695
                is_literature = any(val in ['25', '26'] for val in field_072_values) or \
                                any(val.startswith('821') for val in field_080_values) or \
                                any(val in filter_literature_080 for val in field_650_values) or \
                                any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' in val for val in field_080_values)

                is_literary_science = ('11' in field_072_values and 
                                      (any(val.startswith('82') and not val.startswith(('820', '821')) for val in field_080_values) or 
                                       any(val in filter_literary_science_080 for val in field_650_values))) or \
                                      any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' not in val for val in field_080_values) or \
                                      any(val.startswith('82.') for val in field_080_values)
                
                has_fields = field_072_values or field_080_values or field_655_values
                classifications = []

                if is_literature:
                    classifications.append('Literature')
                if is_literary_science:
                    classifications.append('Literary Science')
                if not has_fields:
                    classifications.append('Unidentified')

                if is_literature or is_literary_science:
                    literature_record_ids.add(record_id)
                    if is_literature:
                        for id_ in field_100_7_values:
                            literary_author_ids.add(id_)
                        for author, id_ in zip(field_100_values, field_100_7_values):
                            literature_authors.append({'Author': author, 'ID': id_})

                if classifications:
                    for classification in classifications:
                        record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value=classification)]))

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

                if is_literature or is_literary_science or is_czech_literature or is_world_literature or not has_fields:
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
                            'Unidentified' if not has_fields else ''
                        ])).strip(', '),
                        '690': ', '.join(filter(None, [
                            'Czech Literature' if is_czech_literature else '', 
                            'World Literature' if is_world_literature else ''
                        ])).strip(', '),
                        '691': form_691
                    })
                    writer_marc.write(record)
                    writer_mrk.write(record)
            
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
    
    with open(output_marc21_file, 'wb') as marc_fh, open(output_mrk_file, 'wt', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)
        
        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
            chunk_df, literary_author_ids, literature_authors_df, literature_record_ids, unidentified_record_ids = filter_marcxml(chunk_file_mrc, filter_df, writer_marc, writer_mrk)
            all_records.append(chunk_df)
            all_literary_author_ids.update(literary_author_ids)
            all_literature_record_ids.update(literature_record_ids)
            all_literature_authors.append(literature_authors_df)
            all_unidentified_record_ids.update(unidentified_record_ids)
        
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
                            form_691 = 'Printed' if ldr_8 in ['m', 'a', 'b', 's'] else 'Other forms'
                            if any(val.startswith('elektron') or val.startswith('stream') for val in field_655_values) or \
                               any('[zvukový záznam]' in field_245_values or '[elektronický zdroj]' in field_245_values for val in field_245_values):
                                form_691 = 'Other forms'
                            
                            if record_id in all_unidentified_record_ids:
                                existing_record = next((rec for rec in reprocess_records if rec['Record ID'] == record_id and 'Unidentified' in rec['695']), None)
                                if existing_record:
                                    existing_record['695'] = ', '.join(set(existing_record['695'].split(', ') + ['Non-Literary']))
                                    existing_record['691'] = form_691
                            else:
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
                                    '691': form_691  # Dodanie pola 691
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


combined_literature_authors_df.to_excel('literature_authors.xlsx', index=False)
with pd.ExcelWriter('filtered_combined_31-07.xlsx', engine='xlsxwriter') as writer:
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





def filter_marcxml(file_path, filter_df, writer_marc, writer_mrk):
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
                field_245_values = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields())
                field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                # CRITERION A: 695
                is_literature = any(val in ['25', '26'] for val in field_072_values) or \
                                any(val.startswith('821') for val in field_080_values) or \
                                any(val in filter_literature_080 for val in field_650_values) or \
                                any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' in val for val in field_080_values)

                is_literary_science = ('11' in field_072_values and 
                                      (any(val.startswith('82') and not val.startswith(('820', '821')) for val in field_080_values) or 
                                       any(val in filter_literary_science_080 for val in field_650_values))) or \
                                      any(val.startswith(('820', '83', '84', '85', '86', '87', '88', '89')) and '-' not in val for val in field_080_values) or \
                                      any(val.startswith('82.') for val in field_080_values)
                
                has_fields = field_072_values or field_080_values or field_655_values
                classifications = []

                if is_literature:
                    classifications.append('Literature')
                if is_literary_science:
                    classifications.append('Literary Science')
                if not has_fields:
                    classifications.append('Unidentified')
                    unidentified_record_ids.add(record_id)

                if is_literature or is_literary_science:
                    literature_record_ids.add(record_id)
                    if is_literature:
                        for id_ in field_100_7_values:
                            literary_author_ids.add(id_)
                        for author, id_ in zip(field_100_values, field_100_7_values):
                            literature_authors.append({'Author': author, 'ID': id_})

                if classifications:
                    for classification in classifications:
                        record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value=classification)]))
                        # if classification == 'Unidentified':
                        #     unidentified_record_ids.add(record_id)
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

                if is_literature or is_literary_science or is_czech_literature or is_world_literature or not has_fields:
                    matching_count += 1
                    existing_record = next((rec for rec in records if rec['Record ID'] == record_id), None)
                    if existing_record:
                        existing_record['695'] = ', '.join(set(existing_record['695'].split(', ') + classifications))
                    else:
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
                            '695': ', '.join(filter(None, classifications)).strip(', '),
                            '690': ', '.join(filter(None, [
                                'Czech Literature' if is_czech_literature else '', 
                                'World Literature' if is_world_literature else ''
                            ])).strip(', '),
                            '691': form_691
                        })
                    writer_marc.write(record)
                    writer_mrk.write(record)
            
            except Exception as e:
                print(f"Error processing record ID {record_id}: {e}")
    
    literature_authors_df = pd.DataFrame(literature_authors).drop_duplicates()
    return pd.DataFrame(records), literary_author_ids, literature_authors_df, literature_record_ids, unidentified_record_ids


def filter_marcxml(file_path, filter_df, writer_marc, writer_mrk):
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
                field_245_values = ' '.join(subfield for field in record.get_fields('245') for subfield in field.get_subfields())
                field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                # CRITERION A: 695
                is_literature = any(val in ['25', '26'] for val in field_072_values) or \
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

                if not has_fields:
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

                if is_literature or is_literary_science or is_czech_literature or is_world_literature or not has_fields:
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
                            'Unidentified' if not has_fields else ''
                        ])).strip(', '),
                        '690': ', '.join(filter(None, [
                            'Czech Literature' if is_czech_literature else '', 
                            'World Literature' if is_world_literature else ''
                        ])).strip(', '),
                        '691': form_691
                    })
                    writer_marc.write(record)
                    writer_mrk.write(record)
            
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
    
    # Przetwarzanie plików chunków i zapisanie wyników w plikach mrc/mrk
    with open(output_marc21_file, 'wb') as marc_fh, open(output_mrk_file, 'wt', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)
        
        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
            chunk_df, literary_author_ids, literature_authors_df, literature_record_ids, unidentified_record_ids = filter_marcxml(chunk_file_mrc, filter_df, writer_marc, writer_mrk)
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
    return combined_df, combined_literature_authors_df

# Przykładowe użycie

chunk_files_path = 'D:/Nowa_praca/czech_works/chunks_NKC'
chunk_files = [os.path.join(chunk_files_path, f'chunk_{i+1}.mrc') for i in range(1)]
# Przetwarzanie plików chunków i zapisanie wyników do jednego zbiorczego pliku mrc i mrk
combined_df, combined_literature_authors_df = process_chunks(chunk_files, df, 'filtered_combined.mrc', 'filtered_combined.mrk')
combined_literature_authors_df.to_excel('literature_authors.xlsx', index=False)
with pd.ExcelWriter('filtered_combined_31-07.xlsx', engine='xlsxwriter') as writer:
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

#%% Do wypróbowania na wyjeździe:
    
def process_chunks(chunk_files, filter_df, output_marc21_file, output_mrk_file):
    all_records = []
    all_literature_record_ids = set()
    all_literary_author_ids = set()
    all_literature_authors = []  # Lista do przechowywania autorów literatury
    all_unidentified_record_ids = set()
    
    with open(output_marc21_file, 'wb') as marc_fh, open(output_mrk_file, 'wt', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)
        
        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
            chunk_df, literary_author_ids, literature_authors_df, literature_record_ids, unidentified_record_ids = filter_marcxml(chunk_file_mrc, filter_df, writer_marc, writer_mrk)
            all_records.append(chunk_df)
            all_literary_author_ids.update(literary_author_ids)
            all_literature_record_ids.update(literature_record_ids)
            all_literature_authors.append(literature_authors_df)
            all_unidentified_record_ids.update(unidentified_record_ids)
        
        writer_marc.close()
        writer_mrk.close()
    
    combined_df = pd.concat(all_records, ignore_index=True)
    combined_literature_authors_df = pd.concat(all_literature_authors, ignore_index=True).drop_duplicates()
    
    # Create a cache dictionary for combined_df
    combined_df_dict = combined_df.set_index('Record ID').to_dict('index')

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
                            ldr_8 = record.leader[7]
                            form_691 = 'Printed' if ldr_8 in ['m', 'a', 'b', 's'] else 'Other forms'
                            if any(val.startswith('elektron') or val.startswith('stream') for val in field_655_values) or \
                               any('[zvukový záznam]' in field_245_values or '[elektronický zdroj]' in field_245_values for val in field_245_values):
                                form_691 = 'Other forms'
    
                            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Non-Literary')]))
                            writer_marc.write(record)
                            writer_mrk.write(record)

                            if record_id in all_unidentified_record_ids:
                                if record_id in combined_df_dict:
                                    combined_df_dict[record_id]['695'] = combined_df_dict[record_id]['695'] + ', Non-Literary'
                                    combined_df_dict[record_id]['691'] = form_691
                                    continue  # Skip adding this record again, it's already in records
                            
                            else:
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
                                    '691': form_691  # Dodanie pola 691
                                })
                    
                    except Exception as e:
                        print(f"Error reprocessing record ID {record_id}: {e}")
    
    # Convert the dictionary back to DataFrame
    updated_combined_df = pd.DataFrame.from_dict(combined_df_dict, orient='index')
    final_df = pd.DataFrame(reprocess_records)
    combined_df = pd.concat([updated_combined_df, final_df], ignore_index=True)
    return combined_df, combined_literature_authors_df

# Przykładowe użycie

chunk_files_path = 'D:/Nowa_praca/czech_works/chunks_NKC'
chunk_files = [os.path.join(chunk_files_path, f'chunk_{i+1}.mrc') for i in range(6)]
# Przetwarzanie plików chunków i zapisanie wyników do jednego zbiorczego pliku mrc i mrk
combined_df, combined_literature_authors_df = process_chunks(chunk_files, df, 'filtered_combined.mrc', 'filtered_combined.mrk')

