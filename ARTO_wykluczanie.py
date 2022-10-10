from definicje import *
import pandas as pd
import os
from tqdm import tqdm
plik=r'/home/darek/Nowa_praca /fennica/rekordyarto11'
lista=mark_to_list(plik+'.mrk')
dictrec=list_of_dict_from_list_of_lists(lista)
monography=LDR_monography(dictrec)
article=LDR_article(dictrec)
#%% zabawa liczbami
# Sprawdzam wszystkie ósemki:
records8084=[]
for record in tqdm(article):
    if '080' in record.keys() and '084' in record.keys():
        if any([field[4:].startswith('8') for field in record['080'].split('❦')]) or any([field[4:].startswith('8') for field in record['084'].split('❦')]):
            records8084.append(record)
            
    elif '080' in record.keys():
        if any([field[4:].startswith('8') for field in record['080'].split('❦')]):
            
            records8084.append(record)

    elif '084' in record.keys():
        if any([field[4:].startswith('8') for field in record['084'].split('❦')]):
            records8084.append(record)
            
# sprawdzam YKL 
ykl=[]
yklpole80=[]
for record in records8084:
    if '084'in record.keys() and '080' in record.keys():
        
        if 'ykl' in record['084'] or 'ykl' in record['080']:
            ykl.append(record)
    elif '080' in record.keys():
        if 'ykl' in record['080']:
            ykl.append(record)
            yklpole80.append(record)
    elif '084' in record.keys():
        if 'ykl' in record['084']:
            ykl.append(record)
            
# szukam dobrych YKL (w polu 080 nie ma YKL)
          
dobreYKL=[]
ykl87_89=[]
for yklrecord in ykl:
    if '084' in yklrecord.keys():
        if any([field[4:].startswith(("80", "81", "82", "83", "84", "85", "86")) for field in yklrecord['084'].split('❦')]):
            dobreYKL.append(yklrecord)
        else:
            ykl87_89.append(yklrecord)
# z rekordów 8084 wykluczam te ze zlymi YKLami:
tylko_dobreYKL=[record for record in records8084 if record not in ykl87_89]

#%%Sprawdzam czy mam jakies 810,  (zeby je wykluczyc przy wyszukiwaniu 81, jesli brak moge od razu wyszukac 81, jesli brak jade dalej. Wszystko to w rekordach bez YKL (tam 81 jest dobre i porządane)
rekordy_bez_YKL=[e for e in tqdm(tylko_dobreYKL) if e not in ykl ] 
dobre810=[]

for otherrecord in rekordy_bez_YKL:
    if '080' in otherrecord.keys() and '084' in otherrecord.keys():
        if any([field[4:].startswith('81') for field in otherrecord['080'].split('❦')]) or any([field[4:].startswith('81') for field in otherrecord['084'].split('❦')]):
            dobre810.append(otherrecord)
            
    elif '080' in otherrecord.keys():
        if any([field[4:].startswith('81') for field in otherrecord['080'].split('❦')]):
            
            dobre810.append(otherrecord)

    elif '084' in otherrecord.keys():
        if any([field[4:].startswith('81') for field in otherrecord['084'].split('❦')]):
            dobre810.append(otherrecord)
#jesli są 810 wywalam je (z listy bez YKL- zeby nie usunac dobrych 810) i przeszukuje pozostale pod katem 81
rekordy_bez_810= [e for e in rekordy_bez_YKL if e not in dobre810]
zle81=[]

for otherrecord in rekordy_bez_810:
    if '080' in otherrecord.keys() and '084' in otherrecord.keys():
        if any([field[4:].startswith('81') for field in otherrecord['080'].split('❦')]) or any([field[4:].startswith('81') for field in otherrecord['084'].split('❦')]):
            zle81.append(otherrecord)
            
    elif '080' in otherrecord.keys():
        if any([field[4:].startswith('81') for field in otherrecord['080'].split('❦')]):
            
            zle81.append(otherrecord)

    elif '084' in otherrecord.keys():
        if any([field[4:].startswith('81') for field in otherrecord['084'].split('❦')]):
            zle81.append(otherrecord)

ostateczna_lista_po_numerkach=[e for e in tylko_dobreYKL if e not in zle81 ]

#%% zabawa z hasłami
hasla=pd.read_excel(r"/home/darek/Nowa_praca /fennica/Hasła_po_selekcji_wszystkie.xlsx")
listahasel=hasla['hasła'].to_list()
record_dobry_po_hasle=[]
for record in tqdm(article):

    if '650' in record.keys():
        if any ([str(haslo).lower() in str(record['650']).lower() for haslo in listahasel ]):
            record_dobry_po_hasle.append(record)
# wykluczam czy te po numerkach sa w tych po hasle
unique_rekord_dobry_po_hasle=[e for e in tqdm(record_dobry_po_hasle) if e not in ostateczna_lista_po_numerkach]
ostateczna_lista=unique_rekord_dobry_po_hasle+ostateczna_lista_po_numerkach
to_file(plik+'odsiane.mrk',ostateczna_lista)