# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:10:11 2024

@author: dariu
"""

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
