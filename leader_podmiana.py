# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:56:50 2025

@author: darek
"""

from pymarc import MARCReader, MARCWriter

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