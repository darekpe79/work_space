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


