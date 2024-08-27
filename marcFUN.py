# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:09:55 2024

@author: dariu
"""

from pymarc import Record, Field, MARCWriter, Subfield

# Tworzenie rekordu MARC
record = Record()
record.add_field(
    Field(
        tag='001',
        data='123456'
    )
)
record.add_field(
    Field(
        tag='245',
        indicators=['0', '0'],
        subfields=[
            Subfield(code='a', value='Main title :'),
            Subfield(code='b', value='subtitle /'),
            Subfield(code='c', value='author.')
        ]
    )
)
field_245_values = ' '.join(field.value() for field in record.get_fields('245')) 
# Zapis rekordu do pliku
with open('example.mrc', 'wb') as fh:
    writer = MARCWriter(fh)
    writer.write(record)
    writer.close()

# Wydrukowanie rekordu w formacie MARC21
print(record)
