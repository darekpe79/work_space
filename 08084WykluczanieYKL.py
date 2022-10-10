import pandas as pd
import os
from tqdm import tqdm
from definicje import *
fennica08084=mark_to_list(r"F:/Nowa_praca/fennica/fennica080084.mrk")
Dict_fennica08084=list_of_dict_from_list_of_lists(fennica08084)

Dict_fennica08084M=LDR_monography(Dict_fennica08084)
##Szukam rekordów z yklami:
ykl=[]
yklpole80=[]
for record in Dict_fennica08084M:
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
            
#YKl w polu 080 wystapil 5 razy (rekordy do wywalenia-złe) wiec pomijam, jeli którekolwiek pole zaczyna sie od tego co chce, to biore, jeli w ogóle nie ma tych co chce, pomijam:            
dobreYKL=[]
ykl87_89=[]
for yklrecord in ykl:
    if '084' in yklrecord.keys():
        if any([field[4:].startswith(("80", "81", "82", "83", "84", "85", "86")) for field in yklrecord['084'].split('❦')]):
            dobreYKL.append(yklrecord)
        else:
            ykl87_89.append(yklrecord)
 # wykluczam złe YKLe z calej listy rekordow i rekordy dobre zapisuje do pliku      
recordy_dobre=[e for e in tqdm(Dict_fennica08084M) if e not in ykl87_89 and e not in yklpole80 ]        

to_file('08084_bez_blednych_YKL.mrk', recordy_dobre)
#wykluczam wszystkie rekordy z YKL z tych dobrych do dalszej analizy (UKD, DEWEY) i poszukiwań potencjalych smieci
rekordy_bez_YKL=[e for e in tqdm(recordy_dobre) if e not in ykl ]        
        
# 81 w UKD to jezykoznawstwo, ale 810 w DEWEY to American literature in English więc odszukam wszystkie 810        
        
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
# nie ma wątpliwych rekordow brak i 810 i 81  
        
        
        
        
        
        