# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 08:50:51 2023

@author: dariu
"""

import pymarc
import pymarc

import pymarc

def modify_marc_field(record, field_tag, subfield_code, new_value, occurrence=1):
    """
    Modifies the value of a specific subfield within a particular occurrence of a field in a MARC21 record.

    Args:
        record (pymarc.Record): The MARC21 record to be modified.
        field_tag (str): The field tag of the field to be modified.
        subfield_code (str): The subfield code of the subfield to be modified.
        new_value (str): The new value to be set for the specified subfield.
        occurrence (int): The occurrence number of the field to be modified. Defaults to 1. 
        If it's control field just pass None as subfield_code

    Returns:
        pymarc.Record: The modified MARC21 record.
    """
    fields = record.get_fields(field_tag)
    if len(fields) >= occurrence:
        if subfield_code==None:
            
            field = fields[0]
            field.data = new_value
        else:
            field = fields[occurrence - 1]
            subfields = field.subfields
            modified_subfields = []
            for subfield in subfields:
                if subfield.code == subfield_code:
                    modified_subfields.append(pymarc.Subfield(subfield_code, new_value))
                else:
                    modified_subfields.append(subfield)
            field.subfields = modified_subfields
    return record







# Create a MARC21 record
record = pymarc.Record()
field_650_1 = pymarc.Field(
    tag='650',
    indicators=[' ', ' '],
    subfields=[
        pymarc.Subfield('a', 'Topic 1')
    ]
)
field_650_2 = pymarc.Field(
    tag='650',
    indicators=[' ', ' '],
    subfields=[
        pymarc.Subfield('a', 'Topic 2')
    ]
)
record.add_field(field_650_1)
record.add_field(field_650_2)

# Print the original record
print("Original Record:")
print(record)

# Modify the second occurrence of field 650
record = modify_marc_field(record, '650', 'a', 'New Topic', occurrence=2)

# Print the modified record
print("\nModified Record:")
print(record)
# Create a MARC21 record
record = pymarc.Record()
field_001 = pymarc.Field(
    tag='001',
    data='123456789'
)
record.add_field(field_001)


# Modify the value of control field 001
record = modify_marc_field(record, '001', None, '987654321')

# Print the modified record
print("\nModified Record:")
print(record)