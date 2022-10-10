# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:01:18 2022

@author: dariu
"""

records = []
with open(r"C:\Users\dariu\Desktop\praca\wetransfer_ucla0110-mrk_2022-08-10_1019\150.txt", 'r', encoding = 'utf-8') as mrk:
    record = []
    for line_l in mrk.readlines():
        line=line_l[10:]
        print(line)
        if line == '\n':
            pass
        elif line.startswith('LDR') and record: 
            records.append(record)
            record = []
            record.append(line)
        else:
            record.append(line)
    records.append(record)   
recs2table = []
for record in records:
    rec_dict = {}
    for field in record:
        field_stripped=field[:7].strip(' L')
        print(field_stripped)
        if field[:7] in rec_dict.keys():
            rec_dict[field[:7] = '❦'.join([rec_dict[field[:3]], field[6:].strip()])
        else:
            rec_dict[field[1:4]] = field[6:].strip()
    recs2table.append(rec_dict)

    