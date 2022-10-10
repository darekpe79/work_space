# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:14:40 2022

@author: darek
"""

import sqlite3
import json
from tqdm import tqdm
import requests
from definicje import *
import pandas as pd
import os
from tqdm import tqdm
from googletrans import Translator
from itertools import zip_longest
import regex as re
from time import time
import unicodedata as ud
conn = sqlite3.connect(r"F:\Nowa_praca\Fennica_BN_books_SQL\PBL_proba_hasla1.sqlite")
cur = conn.cursor()

# Make some fresh tables using executescript()
cur.executescript('''
CREATE TABLE IF NOT EXISTS Authors (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name    TEXT,
    viaf  TEXT,
    lata TEXT,
    UNIQUE (name,viaf,lata)
    
);
CREATE TABLE IF NOT EXISTS Book (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    book_name TEXT,
    book_isbn TEXT,
    id_rec TEXT,
    year TEXT,
    UNIQUE (book_name,book_isbn, id_rec, year)
    
   
);

CREATE TABLE IF NOT EXISTS Field_650 (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    subfield_a TEXT,
    link TEXT,

    UNIQUE (subfield_a, link)
    
    
);

CREATE TABLE IF NOT EXISTS Field_655 (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    subfield_a TEXT,
    link TEXT,

    UNIQUE (subfield_a, link)
);    

CREATE TABLE IF NOT EXISTS Field_080 (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    UDC    TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS Field_084 (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    Other_clasification   TEXT UNIQUE
);
    
    
CREATE TABLE IF NOT EXISTS Book_auth (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    book_id INTEGER,
    author_id INTEGER,
    publisher_id INTEGER,
    original_lang_id INTEGER,
    doc_lang_id INTEGER,
    translated_id INTEGER,
    field650_id INTEGER,
    field655_id INTEGER,
    field_080_id INTEGER,
    field_084_id INTEGER, 
    
    UNIQUE (book_id, author_id, publisher_id, original_lang_id,doc_lang_id,translated_id, field650_id, field655_id, field_080_id, field_084_id )
    
    
);

CREATE TABLE IF NOT EXISTS Publisher (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    publisher_name    TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS Original_lang(
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    orig_lang    TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS Doc_lang(
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    doc_lang    TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS Translation(
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    translated    TEXT UNIQUE
);


''')



# DODAĆ ID REKORDU

plik=r"F:\Nowa_praca\pliki_Nikodem_30.03.2022\wetransfer_libri_iteracja7_30-03-2022_2022-03-30_0720\libri_30-03-2022_iter7_final\fennica.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)
url=r'(?<=\$0).*?(?=\$|$)'

patt_authortytulisbn=r'(?<=\$a).*?(?=\$|$)'
patt_data=r'(?<=\$d).*?(?=\$|$)'
patt_viaf_number=r'(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
patt_publisher=r'(?<=\$b).*?(?=\$|$)'
rokpatt=r'(?<=\$c).*?(?=\$|$)'
translation_lang_patt=r'(?<=\$h).*?(?=\$|$)'

for rekord in tqdm(dictrec):
    list_authors=[]
    if '100' not in rekord and '700' not in rekord:
        list_authors=[{'name':'brak', 'data':'brak', 'viaf':'brak'}]
    if '020' not in rekord:
        isbn='brak'
    if '264' not in rekord:
        list_publisher=['brak']
        year='brak'
    if '001' not in rekord:
        id_record='brak'
    if '041' not in rekord:
        jezyk_dane=[{'translacja':'brak_danych','jezyk_dok':'brak_danych', 'jezyk_oryg':'brak_danych'}]
    if '655' not in rekord:
        f655=[{'a655':'brak', 'link655':'brak'}]
    if '650' not in rekord:
        f650=[{'a650':'brak', 'link650':'brak'}]
    if '080' not in rekord:
        f080=['brak']
    if '084' not in rekord:
        f084=['brak']
                    
    for key, val in rekord.items():
        
        if key=='001':
            id_record=val
        elif key=='020':
            isbn=re.search(patt_authortytulisbn, val)
            if isbn:
                isbn=isbn.group(0)
            else:
                isbn='brak'
       
        elif key=='041':
            jezyk_dane=[]
            val=val.split('❦')
            for v in val:
            
                jezyk_dok=re.search(patt_authortytulisbn, v)
                if jezyk_dok:
                    jezyk_dok=jezyk_dok.group(0).strip('/ .,')
                else:
                    jezyk_dok='brak danych'
                if v.startswith('1'):
                    translation='tak'
                    jezyk_oryg=re.search(translation_lang_patt, v)
                    if jezyk_oryg:
                        jezyk_oryg=jezyk_oryg.group(0).strip('/ .,')
                    else:
                        jezyk_oryg='brak danych'
                elif v.startswith('0'):
                    translation='nie'
                    jezyk_oryg='brak'
                else:
                    translation='brak informacji'
                    jezyk_oryg=re.search(translation_lang_patt, v)
                    if jezyk_oryg:
                        jezyk_oryg=jezyk_oryg.group(0).strip('/ .,')
                    else:
                        jezyk_oryg='brak danych'
                jezyk_dane.append({'translacja':translation,'jezyk_dok':jezyk_dok, 'jezyk_oryg':jezyk_oryg} )
# 655 - Index Term - Genre/Form
        elif key=='655':
            f655=[]
            val=val.split('❦')
            for v in val:
                a655=re.search(patt_authortytulisbn, v)
                link655=re.search(url, v)
                if a655 and link655:
                    a655=a655.group(0).strip('/ .,')
                    link655=link655.group(0).strip('/ .,')
                elif a655 and not link655:
                    a655=a655.group(0).strip('/ .,')
                    link655='brak'
                elif link655 and not a655:
                    link655=link655.group(0).strip('/ .,')
                    a655='brak'
                else:
                    link655='brak'
                    a655='brak'
                                             
                f655.append({'a655':a655, 'link655':link655})
# 650 - Subject Added Entry - Topical Term    
        elif key=='650':
            f650=[]
            val=val.split('❦')
            for v in val:
                a650=re.search(patt_authortytulisbn, v)
                link650=re.search(url, v)
                if a650 and link650:
                    a650=a650.group(0).strip('/ .,')
                    link650=link650.group(0).strip('/ .,')
                elif a650 and not link650:
                    a650=a650.group(0).strip('/ .,')
                    link650='brak'
                elif link650 and not a650:
                    link650=link650.group(0).strip('/ .,')
                    a650='brak'
                else:
                    link650='brak'
                    a650='brak'
                                             
                f650.append({'a650':a650, 'link650':link650})
        
# 080 - Universal Decimal Classification Number    

        elif key=='080':
           f080=[]
           val=val.split('❦')
           for v in val:
               a080=re.search(patt_authortytulisbn, v)
               if a080:
                   a080=a080.group(0).strip('/ .,')
               else:
                   a080='brak'
               f080.append(a080)
# 084 084 - Other Classification Number

        elif key=='084':
           f084=[]
           val=val.split('❦')
           for v in val:
               a084=re.search(patt_authortytulisbn, v)
               if a084:
                   a084=a084.group(0).strip('/ .,')
               else:
                   a084='brak'
               f084.append(a084)
            
                
  
            
        elif key=='100':
            author=re.search(patt_authortytulisbn, val)
            if author:
                author=author.group(0)
            else:
                author='brak'
            data=re.search(patt_data, val)
            if data:
                data=data.group(0)
            else:
                data='brak'
            viaf=re.search(patt_viaf_number, val)
            if viaf:
                viaf=viaf.group(0)
            else:
                viaf='brak'
            list_authors.append({'name':author, 'data':data, 'viaf':viaf})
            
            
            
        elif key=='245':
            title=re.search(patt_authortytulisbn, val)
            if title:
                title=title.group(0).strip('/ .,')
            else:
                title='brak'
                
           
        elif key=='264':
            year=re.search(rokpatt, val)
            if year:
                year=year.group(0).strip('/ .,')
            else:
                year='brak'
            val=val.split('❦')
            list_publisher=[]
            
            for v in val:
                
                publisher=re.search(patt_publisher, v)
                if publisher:
                    publisher=publisher.group(0).strip('/ .,')
                    list_publisher.append(publisher)
                    
                    
                    
            if not list_publisher:
                list_publisher=['brak']
            else:
                list_publisher=list(set(list_publisher))
                
            
            
        elif key=='700':
            val=val.split('❦')
            
            for v in val:
            
                author2=re.search(patt_authortytulisbn, v)
                if author2:
                    author2=author2.group(0)
                else:
                    author2='brak'
                    
                data2=re.search(patt_data, v)
                if data2:
                    data2=data2.group(0)
                else:
                    data2='brak'
                viaf2=re.search(patt_viaf_number, v)
                if viaf2:
                    viaf2=viaf2.group(0)
                else: 
                    viaf2='brak'
                list_authors.append({'name':author2, 'data':data2, 'viaf':viaf2})
        
                          
                
                

            
        
           
            
    
    for elem in list_authors:
        author=elem['name'].strip('/ .,')
        viaf_number=elem['viaf'].strip('/ .,')
        data=elem['data'].strip('/ .,')

   
        cur.execute('''INSERT OR IGNORE INTO Authors (name, viaf, lata)
          VALUES ( ?,?,? )''', ( author, viaf_number, data ) )
        cur.execute('SELECT id FROM Authors WHERE name = ? and viaf= ? and lata = ? ', (author, viaf_number, data ))
        author_id = cur.fetchone()[0]

            
            
        
        for pub in list_publisher:
      
            cur.execute('''INSERT OR IGNORE INTO Publisher (publisher_name)
              VALUES ( ? )''', ( pub, ) )
            cur.execute('SELECT id FROM Publisher WHERE publisher_name = ? ', (pub, ))
            publisher_id = cur.fetchone()[0]
          
            cur.execute('''INSERT OR IGNORE INTO Book (book_name, book_isbn, id_rec, year)
              VALUES ( ?,?,?,?)''', ( title, isbn, id_record, year ) )
            cur.execute('SELECT id FROM book WHERE book_name = ? and book_isbn=? and id_rec=? and year=? ', (title, isbn,id_record, year ))
            book_id = cur.fetchone()[0]
            for j in jezyk_dane:
                translation1=j['translacja'] 
                jezyk_dok1=j['jezyk_dok'] 
                jezyk_oryg1=j['jezyk_oryg']
                
            
                cur.execute('''INSERT OR IGNORE INTO Original_lang (orig_lang)
                  VALUES ( ? )''', ( jezyk_oryg1, ) )
                cur.execute('SELECT id FROM Original_lang WHERE orig_lang = ? ', (jezyk_oryg1, ))
                jezyk_oryg_id = cur.fetchone()[0]
                
                cur.execute('''INSERT OR IGNORE INTO Doc_lang (doc_lang)
                  VALUES ( ? )''', ( jezyk_dok1, ) )
                cur.execute('SELECT id FROM Doc_lang WHERE doc_lang = ? ', (jezyk_dok1, ))
                jezyk_dok_id = cur.fetchone()[0]
                
                cur.execute('''INSERT OR IGNORE INTO Translation (translated)
                  VALUES ( ? )''', (translation1, ) )
                cur.execute('SELECT id FROM Translation WHERE translated = ? ', (translation1, ))
                translated_id = cur.fetchone()[0]
                for field655 in f655:
                    a655=field655['a655']
                    link655=field655['link655']
                    cur.execute('''INSERT OR IGNORE INTO Field_655 (subfield_a,link) VALUES (?,?)''', (a655,link655,))
                    cur.execute('SELECT id FROM Field_655 WHERE subfield_a=? and link=? ', (a655,link655, ))
                    field655_id = cur.fetchone()[0]
                    for field650 in f650:
                        a650=field650['a650']
                        link650=field650['link650']
                        cur.execute('''INSERT OR IGNORE INTO Field_650 (subfield_a,link) VALUES(?,?)''', (a650,link650,))
                        cur.execute('SELECT id FROM Field_650 WHERE subfield_a=? and link=? ', (a650,link650, ))
                        field650_id = cur.fetchone()[0]
                        for field080 in f080:
                            cur.execute('''INSERT OR IGNORE INTO Field_080 (UDC)
                              VALUES ( ? )''', ( field080, ) )
                            cur.execute('SELECT id FROM Field_080 WHERE UDC = ? ', (field080, ))
                            field_080_id = cur.fetchone()[0]
                            for field084 in f084:
                                cur.execute('''INSERT OR IGNORE INTO Field_084 (Other_clasification)
                                  VALUES ( ? )''', ( field084, ) )
                                cur.execute('SELECT id FROM Field_084 WHERE Other_clasification = ? ', (field084, ))
                                field_084_id = cur.fetchone()[0]
                            
                        
                    
                
                
                
                                cur.execute('''INSERT OR IGNORE INTO Book_auth (book_id, author_id, publisher_id, original_lang_id, doc_lang_id, translated_id,field650_id, field655_id,field_080_id, field_084_id )
                                  VALUES ( ?,?,?,?,?,?,?,?,?,? )''', ( book_id, author_id, publisher_id, jezyk_oryg_id, jezyk_dok_id, translated_id,field650_id, field655_id,field_080_id, field_084_id,) )
                #cur.execute('SELECT id FROM book WHERE book_name = ? ', (title, ))
                
                                conn.commit()  
                
                

               


                 #if switch1 == True   

        #switch1==False
'''                
    switch=False
    switch2=False
    switch3=False
    switch4=False
    for key, val in rekord.items():
        if key=='245' or key=='100' or key=='260' or key=='700' or key=='020':
            
             val2=val.split('❦')
             listavalue=[]
             
             
             count=1

             for value in val2:
                 count+=1
                # print(key,value)
                 #print(count)
                 
                 if key=='245':
                     #title=''
                     titlelist=re.findall(patt_authortytulisbn, value)
                     if titlelist:
                         title=titlelist[0].strip('/ .,')
                         switch=True

                         
                 if key=='100' or key=='700':
                     authorlist=re.findall(patt_authortytulisbn, value)
                     if authorlist:
                         author=authorlist[0].strip('/ .,')
                         switch2=True
                         
                         
                         
                         
                        
                         
                     viaf_numberlist=re.findall(patt_viaf_number, value)
                     if viaf_numberlist:
                         viaf_number=viaf_numberlist[0].strip('/ .,')
                         switch3=True

                    
                     
                     datalist=re.findall(patt_data, value)
                     if datalist:
                         data=datalist[0].strip('/ .,')
                         switch4=True

                 if key=='260':
                     
                     publisherlist=re.findall(patt_publisher, value)
                     #print(publisherlist)
                     if publisherlist:
                         publisher=publisherlist[0].strip('/ .,')


                 if switch==True:
                    title=title
                    #print(title)
                 else:
                     title='brak'
                 if switch2==True:
                     
                     author=author
                 else:
                     author='brak'
                 if switch3==True:
                     viaf_number=viaf_number
                 else:
                     viaf_number='brak'
                 if switch4==True:
                     data=data
                 else:
                     data='brak'
                 
                     
                
                 print(author)
                 print(viaf_number)
   #Viaf nie jest unique===brak!!!!!  

'''
