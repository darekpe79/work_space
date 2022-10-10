# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 12:11:24 2021

@author: darek
"""
import re
import pandas as pd
import xlsxwriter
from tqdm import tqdm
from itertools import chain
import regex as re
import math
from collections import Counter
from itertools import combinations
# from Google import Create_Service
import pandas as pd
import numpy as np
import pymarc
import os
import io
import difflib
import statistics
import unidecode
from tqdm import tqdm
import gspread as gs
from gspread_dataframe import set_with_dataframe, get_as_dataframe
issnexcel = pd.read_excel (r"F:\Nowa_praca\authority.xlsx")

def df_to_mrc(df, field_delimiter, path_out, txt_error_file):
    mrc_errors = []
    df = df.replace(r'^\s*$', np.nan, regex=True)
    outputfile = open(path_out, 'wb')
    errorfile = io.open(txt_error_file, 'wt', encoding='UTF-8')
    list_of_dicts = df.to_dict('records')
    for record in tqdm(list_of_dicts, total=len(list_of_dicts)):
        record = {k: v for k, v in record.items() if pd.notnull(v)}
        try:
            pymarc_record = pymarc.Record(to_unicode=True, force_utf8=True, leader=record['LDR'])
            # record = {k:v for k,v in record.items() if any(a == k for a in ['LDR', 'AVA']) or re.findall('\d{3}', str(k))}
            for k, v in record.items():
                v = str(v).split(field_delimiter)
                if k == 'LDR':
                    pass
                elif k.isnumeric() and int(k) < 10:
                    tag = k
                    data = ''.join(v)
                    marc_field = pymarc.Field(tag=tag, data=data)
                    pymarc_record.add_ordered_field(marc_field)
                else:
                    if len(v) == 1:
                        tag = k
                        record_in_list = re.split('\$(.)', ''.join(v))
                        indicators = list(record_in_list[0])
                        subfields = record_in_list[1:]
                        marc_field = pymarc.Field(tag=tag, indicators=indicators, subfields=subfields)
                        pymarc_record.add_ordered_field(marc_field)
                    else:
                        for element in v:
                            tag = k
                            record_in_list = re.split('\$(.)', ''.join(element))
                            indicators = list(record_in_list[0])
                            subfields = record_in_list[1:]
                            marc_field = pymarc.Field(tag=tag, indicators=indicators, subfields=subfields)
                            pymarc_record.add_ordered_field(marc_field)
            outputfile.write(pymarc_record.as_marc())
        except ValueError as err:
            mrc_errors.append((err, record))
    if len(mrc_errors) > 0:
        for element in mrc_errors:
            errorfile.write(str(element) + '\n\n')
    errorfile.close()
    outputfile.close()
    
df_to_mrc(issnexcel, '❦', 'probacz.mrc', 'txt_error_file.txt')
result = issnexcel.to_json(orient="split")
with open("probka.json", "w") as outfile:
    outfile.write(result)
