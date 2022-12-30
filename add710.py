# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:07:56 2022

@author: darek
"""
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
import regex as re
from datetime import date

today = date.today()

# dd/mm/YY
d1 = today.strftime("%d-%m-%Y")
field26x=pd.read_excel('E:/Python/wszystko_bez_710_matcher710-26x.xlsx', sheet_name='publisher',dtype=str)
dictionary26x=field26x.to_dict('records')
field710=pd.read_excel('E:/Python/710_bezdupli_bez_Fin_po_ISNI_Publishers+instytucje.xlsx', sheet_name='publisher',dtype=str)
dictionary710=field710.to_dict('records')
fin11=pd.read_excel('E:/Python/fin_po_isni_do viafowania710.xlsx', sheet_name='Arkusz1',dtype=str)
dictionaryfin11=fin11.to_dict('records')
concat26x_710=(pd.concat([field26x,field710]))
dictionaryconcat=concat26x_710.to_dict('records')
patterna=r'(?<=\$a).*?(?=\$|$)' 
#daty
patternb='(?<=\$b).*?(?=\$|$)'
patternviaf='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
patfin11=r'(?<=\(FIN11\)).*?(?=$| |\$)'

paths=['E:/Python/do_prob.mrk']


#daty
pattern4='(?<=\$v).*?(?=\$|$)'
#pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
slownik={'Wojciech Sumliński Reporter':"12345"}



#val100=[]
for plik in paths:
    record=list_of_dict_from_file(plik)
    nametopath=plik.split('/')[-1].split('.')[0]+'_'
    
    for rec in tqdm(record):
        if '710' not in rec:
            fieldstocheck=[]
            ujednolicone_slownik={}
            if '260' or '264' in rec:
                
                
                field260=rec.get('260',['nicniema'])
                field264=rec.get('264',['nicniema'])
                fieldstocheck.extend(field260)
                fieldstocheck.extend(field264)
            for field in fieldstocheck:
                name = re.findall(patternb, field)
                if name:
                    for n in name:
                        name1=n.strip(', . :').casefold()
                        #print(name1)
                        for records in dictionary26x:
                            if name1 in records.values():
                         #      print(records)
                               
                               ujednolicone=records['Oryginal_710_po_VIAF_i_z_ujednolicone_bez_710']
                               if not isinstance(ujednolicone, float):
                                   ujednolicone=records['Oryginal_710_po_VIAF_i_z_ujednolicone_bez_710'].strip(', . :')
                               else:
                                   ujednolicone=records['Oryginal_260_264'].strip(', . :')
                            
                               ujednolicone_slownik[records['VIAF']]=ujednolicone
            if ujednolicone_slownik:
                
                rec['710']=[]
                for viaf,names in ujednolicone_slownik.items():
                    #print(viaf,names)
                    
                    rec['710'].append(r'2\$a'+names+r'$1http://viaf.org/viaf/'+viaf)


                                   
        else:      
                
            
                       
            for key, val in rec.items():
                
                
            #     '''badam czy są pola 260, 264 i 710 i porównam ich długosc, następnie należy zbadać czego ewentualnie brakuje
            #       (po słowniku- brać oryginał z 26x i szukać w ujednoliconym 710)jesli np. sa 3 260 a 2 710 w slowniku te dwa znajde i wiem które są 710, to wiem którego nie ma
            #     i jesli mam ujednolicone brakujace 260 to moge stworzyc nowe 710 w innym wypadku (jesli znajde, ale nie wszystko, lub nic, to nie nic nie zrobię z automatu, bo mogłoby się zdublować'''
            #     if {"710", "260"} <= rec.keys():
            #         #print(key, val)
                    
    
                    
            #         if key=='260':
            #             for v in val:
                    
            #                 sub260_b=re.findall(patternb, v)
            #                 len260=len(sub260_b)
            #                 #print(len260)
            #         if key=='710':
            #             len_val=len(val)
            # print(len_val)             #print(len_val)
                   
            # if len260 and len_val:
            #             print('ok')
            # if len260!=len_val:
            #     print(val,'BBBBBBBBBBBBBB', sub260_b)
                        
                        
                        
                    
                    
                
                if key=='710':
                    #print(rec)
                    
                    #new_val=[]
                    for v in val: 
                    
                        #print(v)
                        name = re.findall(patterna, v)
                        #print(name)
                        fin11finder=re.findall(patfin11, v)
                        
                        if fin11finder:
                            for records in dictionaryfin11:
                                if fin11finder[0] in records.values():
                                    #print(records['viaf'])
                                    index=[i for i, e in enumerate(val) if e == v]
                                    #print(index)
                                    for i in index:
                                     #   print(val[i])
                                        valstrip=val[i]
                                        new_val=val[i]+r'$1http://viaf.org/viaf/'+records['viaf']
                                        val[i]=new_val.replace(name[0], name[0].strip(', . :'))
                            
                        elif name:
                            name1=name[0].strip(', . :').casefold()
                            for records in dictionary710:
                                if name1 in records.values():
                                    #print(records['VIAF'])
                                    index=[i for i, e in enumerate(val) if e == v]
                                    #print(index)
                                    for i in index:
                                        #print(val[i])
                                        
                                        new_val=val[i]+r'$1http://viaf.org/viaf/'+records['VIAF']
                                        val[i]=new_val.replace(name[0], name[0].strip(', . :'))
                                        
        
                                
                            
                    
                    
                    
                    
                    
    to_file2(nametopath+d1+'.mrk',record)                    
                    
                
                    
                    
                    
 
                
                
                
                
                
                
                
                
                

                

                
                

        



