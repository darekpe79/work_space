# wyszukiwanie rekordow, ktore w polu 080 lub 084 maja ukd/dewey lub ykl zaczynajace sie od 8, nastepnie sprawdzanie czy w polu 650 sa hasla YSO zwiazane  z literatura (literatura wraz z podgrupami)
import pandas as pd
import os
from tqdm import tqdm
from definicje import *

#%%
os.listdir()
        
        
#%%
records = []
for i in range (11):
    path = r"F:\Nowa_praca\fennica\msplit000000{:02d}.mrk".format(i)
    with open(path, 'r', encoding = 'utf-8') as mrk:
        record = []
        for line in mrk.readlines():
            if line != '\n':
                record.append(line)
            else:
                records.append(record)
                record = []
#%%
recs2table =list_of_dict_from_list_of_lists(records)
#%% Wyszukujemy rekordów z polami 080 i 084 dzielimy jesli danych pol jest wiecej i sprawdzamy czy zaczynaja sie od'8' czyli literatura to samo dla 080 i 084
records8084=[]
for record in recs2table:
    if '080' in record.keys() and '084' in record.keys():
        if any([field[4:].startswith('8') for field in record['080'].split('❦')]) or any([field[4:].startswith('8') for field in record['084'].split('❦')]):
            records8084.append(record)
            
    elif '080' in record.keys():
        if any([field[4:].startswith('8') for field in record['080'].split('❦')]):
            
            records8084.append(record)

    elif '084' in record.keys():
        if any([field[4:].startswith('8') for field in record['084'].split('❦')]):
            records8084.append(record)
            
#%% Zapisujemy rekordy do pliku mark
#with open ('fennica080084.mrk', 'w') as writefile:
file1 = open("fennica080084.mrk", "w", encoding='utf-8') 
to_file ("fennica080084.mrk", records8084)
#%% zaciagamy liste hasel bez szczegolow
hasla=pd.read_excel(r"F:\Nowa_praca\fennica\650literatura_bez_szczegolow.xlsx")
listahasel=[]
#cols=list(hasla.columns.values)
for nazwa in list(hasla.columns.values):
    #listahasel.append(list(hasla[nazwa]))
    listahasel+=list(hasla[nazwa])
#%% sprawdzamy czy haslo z listy znajduje sie w rekordzie
records650=[]
for record in tqdm(recs2table):

    if '650' in record.keys():
        if any ([str(haslo).lower() in str(record['650']).lower() for haslo in listahasel ]):
            records650.append(record)
            

#%% zapisanie do pliku mrk rekordow z listy records650                

to_file ("fennica650_ogolne.mrk", records650)

#%% sprawdzamy czy haslo kirjallisuus znajduje sie w rekordzie
kirjallisuus=[]
literaturalista=['kirjallisuus', 'litteratur', 'Literature']
for record in tqdm(recs2table):

    if '650' in record.keys():
        if any ([str(haslo).lower() in str(record['650']).lower() for haslo in literaturalista ]):
            kirjallisuus.append(record)
            

#%% zapisanie do pliku mrk rekordow z listy records650                

to_file ("fennica650only_LITERATURE.mrk", kirjallisuus)
                
                
#%% zaciagamy tylko liste hasel szczegolowych
haslaszczegol=pd.read_excel(r"F:\Nowa_praca\fennica\hasla_tylko_szczegolowe.xlsx")
listahaselszczegol=[]
#cols=list(hasla.columns.values)
for nazwa in list(haslaszczegol.columns.values):
    #listahasel.append(list(hasla[nazwa]))
    listahaselszczegol+=list(haslaszczegol[nazwa])
#%% sprawdzamy czy haslo z listy znajduje sie w rekordzie
records650haslaszczegol=[]
for record in tqdm(recs2table):

    if '650' in record.keys():
        if any ([str(haslo).lower() in str(record['650']).lower() for haslo in listahaselszczegol ]):
            records650haslaszczegol.append(record)
            

#%% zapisanie do pliku mrk rekordow z listy records650                
to_file ("fennica650_tylko_szczegolowe.mrk", records650haslaszczegol)  
#%% Zaciagamy listę hasel YSO87
haslaYSO87=pd.read_excel(r"F:\Nowa_praca\fennica\YSO_87_Literature_hasla.xlsx")
listahaselYSO87=[]
#cols=list(hasla.columns.values)
for nazwa in list(haslaYSO87.columns.values):
    #listahasel.append(list(hasla[nazwa]))
    listahaselYSO87+=list(haslaYSO87[nazwa])
#%% sprawdzamy czy haslo z listy znajduje sie w rekordzie
recordsYSO87=[]
for record in tqdm(recs2table):

    if '650' in record.keys():
        if any ([str(haslo).lower() in str(record['650']).lower() for haslo in listahaselYSO87 ]):
            recordsYSO87.append(record)
            

#%% zapisanie do pliku mrk rekordow z listy records650                

to_file ("fennica_YSO87.mrk", recordsYSO87)

     
#%% Wczytuję pliki do list        
fennica650_ogolne = mark_to_list (r"F:\Nowa_praca\fennica\fennica650_ogolne.mrk")
fennica650only_LITERATURE = mark_to_list(r"F:\Nowa_praca\fennica\fennica650only_LITERATURE.mrk")
fennica650_tylko_szczegolowe = mark_to_list(r"F:\Nowa_praca\fennica\fennica650_tylko_szczegolowe.mrk")
fennica_YSO87 = mark_to_list(r"F:\Nowa_praca\fennica\fennica_YSO87.mrk")
fennica08084=mark_to_list(r"F:/Nowa_praca/fennica/fennica080084.mrk")
#%% robię wykluczenia
Unique_fennica650_ogolne=[e for e in tqdm(fennica650_ogolne) if e not in fennica08084]
Unique_fennica650only_LITERATURE=[e for e in tqdm(fennica650only_LITERATURE) if e not in fennica08084 and e not in Unique_fennica650_ogolne]

Unique_fennica650_tylko_szczegolowe=[e for e in tqdm(fennica650_tylko_szczegolowe) if e not in fennica08084 and e not in Unique_fennica650_ogolne and e not in Unique_fennica650only_LITERATURE]

Unique_fennica_YSO87=[e for e in tqdm(fennica_YSO87) if e not in fennica08084 and e not in Unique_fennica650_ogolne and e not in Unique_fennica650only_LITERATURE and e not in Unique_fennica650_tylko_szczegolowe]
#%% wykluczone listy zapisuje do slownikow
DUnique_fennica650_ogolne=list_of_dict_from_list_of_lists(Unique_fennica650_ogolne)
DUnique_fennica650only_LITERATURE=list_of_dict_from_list_of_lists(Unique_fennica650only_LITERATURE)
DUnique_fennica650_tylko_szczegolowe=list_of_dict_from_list_of_lists(Unique_fennica650_tylko_szczegolowe)
DUnique_fennica_YSO87=list_of_dict_from_list_of_lists(Unique_fennica_YSO87)
to_file ('Unique_fennica650_ogolne.mrk', DUnique_fennica650_ogolne)
to_file ('Unique_fennica650only_LITERATURE.mrk', DUnique_fennica650only_LITERATURE)
to_file ('Unique_fennica650_tylko_szczegolowe.mrk', DUnique_fennica650_tylko_szczegolowe)
to_file ('Unique_fennica_YSO87.mrk', DUnique_fennica_YSO87)

    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
