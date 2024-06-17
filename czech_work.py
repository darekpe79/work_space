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

import os
from pymarc import MARCReader, TextWriter, MARCWriter
from tqdm import tqdm
import pandas as pd

def filter_marcxml(file_path, filter_df, output_marc21_file, output_mrk_file):
    records = []
    processed_count = 0
    matching_count = 0
    
    filter_072 = set(filter_df['072'].dropna().unique())
    filter_080 = set(filter_df['080'].dropna().unique())
    filter_150 = set(filter_df['150'].dropna().unique())
    
    writer = MARCWriter(open(output_marc21_file, 'wb'))
    writer_mrk = TextWriter(open(output_mrk_file, 'wt', encoding='utf-8'))
    
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for record in tqdm(reader, desc="Processing records"):
            try:
                processed_count += 1
                record_id = record['001'].value() if record['001'] else None
                field_072_values = [subfield for field in record.get_fields('072') for subfield in field.get_subfields('9') if subfield in filter_072]
                field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a') if subfield in filter_080]
                field_650_values = [subfield for field in record.get_fields('650') for subfield in field.get_subfields('a') if subfield in filter_df['150']]

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

def process_chunks(chunk_files, filter_df, output_dir='filtered_chunks'):
    all_records = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, chunk_file_mrc in enumerate(chunk_files):
        print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
        chunk_output_marc21 = os.path.join(output_dir, f'filtered_{i+1}.mrc')
        chunk_output_mrk = os.path.join(output_dir, f'filtered_{i+1}.mrk')
        chunk_df = filter_marcxml(chunk_file_mrc, filter_df, chunk_output_marc21, chunk_output_mrk)
        all_records.append(chunk_df)
    
    combined_df = pd.concat(all_records, ignore_index=True)
    return combined_df

# Przykładowe użycie


chunk_files = [os.path.join('chunks', f'chunk_{i+1}.mrc') for i in range(6)]  # Zakładamy 6 pliki chunków

combined_df = process_chunks(chunk_files, df)
combined_df.to_excel('filtered_combined.xlsx', index=False)


#second_proper_ver
import os
from pymarc import MARCReader, TextWriter, MARCWriter
from tqdm import tqdm
import pandas as pd

def filter_marcxml(file_path, filter_df, writer_marc, writer_mrk):
    records = []
    processed_count = 0
    matching_count = 0
    
    filter_150 = set(filter_df['150'].dropna().unique())
    
    with open(file_path, 'rb') as fh:
        reader = MARCReader(fh)
        for record in tqdm(reader, desc="Processing records"):
            try:
                processed_count += 1
                record_id = record['001'].value() if record['001'] else None
                field_015_values = [subfield for field in record.get_fields('015') for subfield in field.get_subfields('a')]
                field_072_values = [subfield for field in record.get_fields('072') for subfield in field.get_subfields('9') if subfield in ['25', '26']]
                field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a') if subfield == '808' or subfield.startswith('82')]
                field_650_values = [subfield for field in record.get_fields('650') for subfield in field.get_subfields('a') if subfield in filter_150]
                
                if field_072_values or field_080_values or field_650_values:
                    matching_count += 1
                    records.append({
                        'Record ID': record_id,
                        '015': ', '.join(field_015_values),
                        '072': ', '.join(field_072_values),
                        '080': ', '.join(field_080_values),
                        '650': ', '.join(field_650_values)
                    })
                    writer_marc.write(record)
                    writer_mrk.write(record)
            
            except Exception as e:
                print(f"Error processing record ID {record_id}: {e}")
    
    print(f"Total processed records in {file_path}: {processed_count}")
    print(f"Total matching records in {file_path}: {matching_count}")
    
    return pd.DataFrame(records)

def process_chunks(chunk_files, filter_df, output_marc21_file, output_mrk_file):
    all_records = []
    
    with open(output_marc21_file, 'wb') as marc_fh, open(output_mrk_file, 'wt', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)
        
        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
            chunk_df = filter_marcxml(chunk_file_mrc, filter_df, writer_marc, writer_mrk)
            all_records.append(chunk_df)
        
        writer_marc.close()
        writer_mrk.close()
    
    combined_df = pd.concat(all_records, ignore_index=True)
    return combined_df

# Przykładowe użycie

chunk_files_path = 'D:/Nowa_praca/czech_works/chunks_NKC'
chunk_files = [os.path.join(chunk_files_path, f'chunk_{i+1}.mrc') for i in range(6)]
# Przetwarzanie plików chunków i zapisanie wyników do jednego zbiorczego pliku mrc i mrk
combined_df = process_chunks(chunk_files, df, 'filtered_combined.mrc', 'filtered_combined.mrk')
combined_df.to_excel('filtered_combined.xlsx', index=False)

#%% przetwarzanie pliku

records = []
processed_count = 0




with open('D:/Nowa_praca/czech_works/NKC_new_conditions29_05/filtered_combined.mrc', 'rb') as fh:
    reader = MARCReader(fh)
    for record in tqdm(reader, desc="Processing records"):
        try:
            processed_count += 1
            record_id = record['001'].value() if record['001'] else None
            link = record['998'].value() if record['998'] else None
            field_015_values = [subfield for field in record.get_fields('015') for subfield in field.get_subfields('a')]
            field_072_values = [subfield for field in record.get_fields('072') for subfield in field.get_subfields('9') ]
            field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')]
            field_650_values = [subfield for field in record.get_fields('650') for subfield in field.get_subfields('a') ]
            
            if field_072_values or field_080_values or field_650_values:
                
                records.append({
                    'Record ID': record_id,
                    'Link':link,
                    '015': ', '.join(field_015_values),
                    '072': ', '.join(field_072_values),
                    '080': ', '.join(field_080_values),
                    '650': ', '.join(field_650_values)
                })

        
        except Exception as e:
            print(f"Error processing record ID {record_id}: {e}")
dataframe=pd.DataFrame(records)
dataframe.to_excel('filtered_combined.xlsx', index=False)
print(f"Total processed records in {file_path}: {processed_count}")
with pd.ExcelWriter('filtered_combined.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    workbook.strings_to_urls = False  # Wyłącz konwersję ciągów znaków na URL
    dataframe.to_excel(writer, index=False)
    
#%% nowe warunki
import os
from pymarc import MARCReader, TextWriter, MARCWriter, Field, Subfield
from tqdm import tqdm
import pandas as pd

def filter_marcxml(file_path, filter_df, writer_marc, writer_mrk):
    records = []
    processed_count = 0
    matching_count = 0
    literature_ids = set()
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
                field_100_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')]
                field_245_values = [subfield for field in record.get_fields('245') for subfield in field.get_subfields()]
                field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                # CRITERION A: 695
                is_literature = any(val in ['25', '26'] for val in field_072_values) or \
                                any(val.startswith('821') for val in field_080_values) or \
                                any(val in filter_literature_080 for val in field_650_values)
                
                is_literary_science = '11' in field_072_values and \
                                      (any(val.startswith('82') and not val.startswith('821') for val in field_080_values) or \
                                       any(val in filter_literary_science_080 for val in field_650_values))
                
                if is_literature:
                    record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literature')]))
                    for id_ in field_100_7_values:
                        literature_ids.add(id_)
                    for author, id_ in zip(field_100_values, field_100_7_values):
                        literature_authors.append({'Author': author, 'ID': id_})

                if is_literary_science:
                    record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Literary Science')]))

                # CRITERION B: 690
                is_czech_literature = any(val.startswith('821.162.3') for val in field_080_values) or \
                                      any('česk' in val for val in field_650_values)
                
                is_world_literature = any(val.startswith('821') and not val.startswith('821.162.3') for val in field_080_values)
                
                if is_czech_literature:
                    record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Czech Literature')]))
                if is_world_literature:
                    record.add_field(Field(tag='690', indicators=[' ', ' '], subfields=[Subfield(code='a', value='World Literature')]))

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
                        '100': ', '.join(field_100_values),
                        '245': ', '.join(field_245_values),
                        '695': ', '.join(filter(None, ['Literature' if is_literature else '', 'Literary Science' if is_literary_science else ''])).strip(', '),
                        '690': ', '.join(filter(None, ['Czech Literature' if is_czech_literature else '', 'World Literature' if is_world_literature else ''])).strip(', '),
                        '691': 'Printed' if ldr_8 in ['m', 'a', 'b', 's'] else 'Other forms'
                    })
                    writer_marc.write(record)
                    writer_mrk.write(record)
            
            except Exception as e:
                print(f"Error processing record ID {record_id}: {e}")
    
    literature_authors_df = pd.DataFrame(literature_authors).drop_duplicates()
    return pd.DataFrame(records), literature_ids, literature_authors_df

def process_chunks(chunk_files, filter_df, output_marc21_file, output_mrk_file):
    all_records = []
    all_literature_ids = set()
    all_literature_authors = []  # Lista do przechowywania autorów literatury
    
    with open(output_marc21_file, 'wb') as marc_fh, open(output_mrk_file, 'wt', encoding='utf-8') as mrk_fh:
        writer_marc = MARCWriter(marc_fh)
        writer_mrk = TextWriter(mrk_fh)
        
        for i, chunk_file_mrc in enumerate(chunk_files):
            print(f"Przetwarzanie części {i+1} z {len(chunk_files)}")
            chunk_df, literature_ids, literature_authors_df = filter_marcxml(chunk_file_mrc, filter_df, writer_marc, writer_mrk)
            all_records.append(chunk_df)
            all_literature_ids.update(literature_ids)
            all_literature_authors.append(literature_authors_df)
        
        writer_marc.close()
        writer_mrk.close()
    
    combined_df = pd.concat(all_records, ignore_index=True)
    combined_literature_authors_df = pd.concat(all_literature_authors, ignore_index=True)
    
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
                        link = record['998'].value() if record['998'] else None
                        
                        field_015_values = [subfield for field in record.get_fields('015') for subfield in field.get_subfields('a')]
                        field_072_values = [subfield for field in record.get_fields('072') for subfield in field.get_subfields('9')]
                        field_080_values = [subfield for field in record.get_fields('080') for subfield in field.get_subfields('a')]
                        field_650_values = [subfield for field in record.get_fields('650') for subfield in field.get_subfields('a')]
                        field_100_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('a')]
                        field_245_values = [subfield for field in record.get_fields('245') for subfield in field.get_subfields()]
                        field_100_7_values = [subfield for field in record.get_fields('100') for subfield in field.get_subfields('7')]
                        field_700_7_values = [subfield for field in record.get_fields('700') for subfield in field.get_subfields('7')]

                        # Check if the record belongs to non-literary production of literary authors
                        if any(id_ in all_literature_ids for id_ in field_100_7_values + field_700_7_values):
                            record.add_field(Field(tag='695', indicators=[' ', ' '], subfields=[Subfield(code='a', value='Non-Literary')]))
                            reprocess_records.append({
                                'Record ID': record_id,
                                'Link': link,
                                '015': ', '.join(field_015_values),
                                '072': ', '.join(field_072_values),
                                '080': ', '.join(field_080_values),
                                '650': ', '.join(field_650_values),
                                '100': ', '.join(field_100_values),
                                '245': ', '.join(field_245_values),
                                '695': 'Non-Literary',
                                '690': '',  # Not applicable in reprocessing
                                '691': ''   # Not applicable in reprocessing
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

with pd.ExcelWriter('filtered_combined_12-06.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    workbook.strings_to_urls = False  # Wyłącz konwersję ciągów znaków na URL
    combined_df.to_excel(writer, index=False)

