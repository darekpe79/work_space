# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:33:38 2023

@author: dariu
"""

from pymarc import Record, Field, Subfield, MARCWriter, MARCReader, TextWriter
import tkinter as tk
from tkinter import filedialog, messagebox

def convert_marc_to_mrk(mrc_file_path, mrk_file_path):
    writer = TextWriter(open(mrk_file_path, 'wt', encoding="utf-8"))
    with open(mrc_file_path, 'rb') as data:
        reader = MARCReader(data)
        for record in reader:
            writer.write(record)
    writer.close()
    messagebox.showinfo('Success', 'Conversion completed successfully!')

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
                        subfields_list.append(Subfield(code=code, value=value))
                    field = Field(
                        tag=tag,
                        indicators=[indicators[0], indicators[1]],
                        subfields=subfields_list
                    )
                    record.add_field(field)

            writer.write(record)
        writer.close()
    messagebox.showinfo('Success', 'Conversion completed successfully!')

def select_file(entry_widget):
    file_path = filedialog.askopenfilename()
    entry_widget.delete(0, tk.END)
    entry_widget.insert(0, file_path)

def start_conversion():
    file1_path = file1_path_entry.get()
    file2_path = file2_path_entry.get()
    conversion_type = conversion_type_var.get()

    if conversion_type == 'MARC to MRK':
        convert_marc_to_mrk(file1_path, file2_path)
    else:  # Assume the conversion type is 'MRK to MARC'
        convert_mrk_to_marc(file1_path, file2_path)

root = tk.Tk()
root.title('MRC and MRK Converter by Dariusz Perliński')

conversion_type_var = tk.StringVar(root)
conversion_type_var.set('MARC to MRK')
conversion_type_menu = tk.OptionMenu(root, conversion_type_var, 'MARC to MRK', 'MRK to MARC')
conversion_type_menu.pack()

tk.Label(root, text="File1 Path:").pack()
file1_path_entry = tk.Entry(root, width=50)
file1_path_entry.pack()
tk.Button(root, text="Browse", command=lambda: select_file(file1_path_entry)).pack()

tk.Label(root, text="File2 Path:").pack()
file2_path_entry = tk.Entry(root, width=50)
file2_path_entry.pack()
tk.Button(root, text="Browse", command=lambda: select_file(file2_path_entry)).pack()

tk.Button(root, text="Start Conversion", command=start_conversion).pack()

root.mainloop()
