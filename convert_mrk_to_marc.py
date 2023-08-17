# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:24:21 2023

@author: dariu
"""
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
from pymarc import Record, Field, Subfield, MARCWriter
import tkinter as tk
import tkinter as tk
from tkinter import filedialog, messagebox



def convert_mrk_to_marc(mrk_file_path, mrc_file_path):
    with open(mrk_file_path, 'r', encoding='utf-8') as file:
        records = file.read().strip().split('\n\n')

    with open(mrc_file_path, 'wb') as marc_file:
        writer = MARCWriter(marc_file)
        for record_data in records:
            record_lines = record_data.split('\n')
            record = Record()
            record.leader = record_lines[0][6:]

            for line in record_lines[1:]:
                tag = line[1:4]

                if tag < '010':
                    # Control fields (001 to 009)
                    value = line[6:]
                    if tag == '008':
                        value = value.replace('\\', ' ')
                    record.add_field(Field(tag=tag, data=value))
                else:
                    indicators = line[6:8].replace('\\', ' ')
                    subfields_raw = line[8:].split('$')[1:]
                    subfields_list = []
                    for subfield in subfields_raw:
                        if not subfield:  # Skip empty subfields
                            continue
                        code = subfield[0]
                        value = subfield[1:]
                        if tag == '080' and code == '1':
                            value = value.replace('\\', '')
                        subfields_list.append(Subfield(code=code, value=value))

                    field = Field(
                        tag=tag,
                        indicators=[indicators[0], indicators[1]],
                        subfields=subfields_list
                    )

                    record.add_field(field)

            writer.write(record)
        writer.close()

def select_mrk_file():
    file_path = filedialog.askopenfilename(filetypes=[("MRK files", "*.mrk"), ("All files", "*.*")])
    mrk_file_path_entry.delete(0, tk.END)
    mrk_file_path_entry.insert(0, file_path)

def select_mrc_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".mrc", filetypes=[("MARC files", "*.mrc"), ("All files", "*.*")])
    mrc_file_path_entry.delete(0, tk.END)
    mrc_file_path_entry.insert(0, file_path)

def start_conversion():
    mrk_file_path = mrk_file_path_entry.get()
    mrc_file_path = mrc_file_path_entry.get()
    convert_mrk_to_marc(mrk_file_path, mrc_file_path)
    messagebox.showinfo('Success', 'Conversion completed successfully!')

root = tk.Tk()
root.title('MRK to MARC Converter')

# MRK File Path
tk.Label(root, text="MRK File Path:").pack()
mrk_file_path_entry = tk.Entry(root, width=50)
mrk_file_path_entry.pack()
tk.Button(root, text="Browse", command=select_mrk_file).pack()

# MRC File Path
tk.Label(root, text="MRC File Path:").pack()
mrc_file_path_entry = tk.Entry(root, width=50)
mrc_file_path_entry.pack()
tk.Button(root, text="Browse", command=select_mrc_file).pack()

# Start Conversion Button
tk.Button(root, text="Start Conversion", command=start_conversion).pack()

root.mainloop()

# Example usage





# Example usage
mrk_file_path = 'C:/Users/dariu/libri_marc_bn_articles_2023-08-07new_viaf_11-08-2023.mrk'
mrc_file_path = 'output.mrc'


convert_mrk_to_marc(mrk_file_path, mrc_file_path)
for my_marc_file in tqdm(['D:/Nowa_praca/08082023-Czarek_BN_update/przerobione-viaf/przyklad_niszczenie_liter2 (2).mrc']):
    writer = TextWriter(open('genre_655.mrk','wt',encoding="utf-8"))
    with open(my_marc_file, 'rb') as data, open(my_marc_file+'genre_655.mrc','wb')as data1:
        reader = MARCReader(data)
        for record in tqdm(reader):
            print(record)
            writer.write(record)    
writer.close() 
            
