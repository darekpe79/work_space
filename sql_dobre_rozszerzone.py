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
conn = sqlite3.connect('fennica_proba.sqlite')
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
CREATE TABLE IF NOT EXISTS Book_auth (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    book_id INTEGER,
    author_id INTEGER,
    publisher_id INTEGER,
    original_lang_id INTEGER,
    doc_lang_id INTEGER,
    translated_id INTEGER,
    UNIQUE (book_id, author_id, publisher_id, original_lang_id,doc_lang_id,translated_id )
    
    
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

plik=r"F:\Nowa_praca\fennica\MARK 10.03.2022Ujednolicone_OSOBY\SQL.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)

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
    if '260' not in rekord:
        list_publisher=['brak']
        year='brak'
    if '001' not in rekord:
        id_record='brak'
                    
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
                
           
        elif key=='260':
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
                
                
                cur.execute('''INSERT OR IGNORE INTO Book_auth (book_id, author_id, publisher_id, original_lang_id, doc_lang_id, translated_id)
                  VALUES ( ?,?,?,?,?,? )''', ( book_id, author_id, publisher_id, jezyk_oryg_id, jezyk_dok_id, translated_id ) )
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
