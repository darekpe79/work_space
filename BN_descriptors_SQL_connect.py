# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:31:43 2022

@author: dariu
"""

import sqlite3
import json
import pandas as pd
from definicje import *
conn = sqlite3.connect('C:/Users/dariu/Downloads/db.sqlite3')
cur = conn.cursor()
#cur.execute("SHOW TABLES")
myresult = cur.fetchall()
sql_query = """SELECT name FROM sqlite_master
    WHERE type='table';"""
mojeposplitowane=pd.read_excel ('C:/Users/dariu/PBL_650_Mapowanie_PBL_BN.xlsx', sheet_name='650')
list6501=mojeposplitowane['exact terms'].to_list()
dicto={}
descriptors={}
nasze_splitted=[]
for term in tqdm(list6501):
    
    if type(term)==int:
        
        continue
    #dicto[term]=[]
    tymczasowa=[]
    tymczasowa_unique=[]
    
    for t in term.split('|'):
        
        
        
        tstrip=t.strip(' .')
        nasze_splitted.append(t)
        tymczasowa.append(t)
        tymczasowa_unique.append(tstrip)
    tymczasowa_unique=unique(tymczasowa_unique)
    
    if len(tymczasowa)>len(tymczasowa_unique):
        
        dicto[term]=tymczasowa_unique
    else:
        dicto[term]=tymczasowa
    
    dicto[term]='|'.join(unique(dicto[term]))
        
        for row in cur.execute('SELECT descriptor FROM descriptors'):
            if row[0].strip(' .')==tstrip:
                descriptors[t]=(row[0],term)
                
for z in cur.execute(sql_query):
    print(z)
 
for row in cur.execute('SELECT descriptor FROM descriptors') :
	print(type(row))



lista=['lala', 'lala|lala','lala|lala21']
dicto1={}
for l in lista:
    dicto1[l]=[]
    for x in l.split('|'):
        dicto1[l].append(x)
    dicto1[l]='|'.join(unique(dicto1[l]))
excel=pd.DataFrame.from_dict(dicto, orient='index')
excel.to_excel('C:/Users/dariu/exact_term_splited_clear.xlsx', sheet_name='650') 
nasz_invented=[]
for nasz in nasze_splitted:
    if nasz not in descriptors:
        nasz_invented.append(nasz)
        
df = pd.DataFrame(nasz_invented)
df.to_excel('invented_by_PBL.xlsx', sheet_name='650') 