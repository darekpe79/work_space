# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:48:03 2021

@author: darek
"""

from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm


from itertools import zip_longest
import regex as re
import requests


#%% Proba na jednym pliku
plik=r"F:\Nowa_praca\fennica\MARK 10.03.2022Ujednolicone_OSOBY\calosc_arto_dobre_995.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)

        
            
        


pattern=r'(?<=\\\$).*?(?=$)'
string="\\\$aKansalliskirjasto"
## inne podescie ten sam efekt: 
rekordy=[]
for listy in tqdm(dictrec):
    nowe={}
    for k,v in listy.items():
        nowe[k]=v
        if k=='995':
            val2=v.split('❦')
            listavalue=[]
            for vi in val2:
                #print(vi)
                splited=vi.split('$')

                
                newlist=[]
                for elem in splited:
                    if elem.startswith('aKansalliskirjasto'):
                        
                        newlist.append(elem)
                        
                        
                    elif elem.startswith('a'):
                        elem=elem.replace(elem[0],'b')
                        newlist.append(elem)
                    else:
                        newlist.append(elem)
                new2='$'.join(newlist)
                    
                    

                listavalue.append(new2)
            
        #new=v.replace("$0(VIAF)", "$1http://viaf.org/viaf/")
        
            nowe[k]='❦'.join(listavalue)
            
    rekordy.append(nowe)
to_file('calosc_arto_dobre_995.mrk',rekordy)


#%%
strings=json.dumps(rekordy)
string_replaced=strings.replace("$0viaf", "$1http://viaf.org/viaf/")

mydict=json.loads(string_replaced)        
to_file('pbl_marc_books_2021-8-4_ISSN_ALL_good995_ALL_VIAF.mrk',mydict)

#%%


field='995'
subfield='a'
position=1
pattern=r'(?<=\${}).*?(?=\$|$)'.format(subfield)
string="PLiteracka"
rekordy=[]
for listy in tqdm(dictrec):
    nowe={}
    for k,v in listy.items():
        nowe[k]=v
        if k==field:
            #print(v)
            podpole995_lista=re.findall(pattern, v)
            try:
                podpole995=podpole995_lista[position-1]
                #print(podpole995)
                podpola=v.split('$'+subfield)
                len_podpola=len(podpola)
                if len_podpola==position+1:
                    
                    lastsubfield=podpola[-1]
                    last_list=lastsubfield.split('$')
                    last_string_to_change=last_list[0]
                    newstring=last_string_to_change.replace(podpole995,string )
                    last_list[0]=newstring
                    newstring='$'.join(last_list)
                    podpola[-1]=newstring
                    newstring=('$'+subfield).join(podpola)
                    nowe[k]=newstring
                    newstring=''
                
                else:
                    find_sub=podpola[position]
                    #print(find_sub)
                    
                    newstring=find_sub.replace(podpole995,string )
                    podpola[position]=newstring
                    newstring=('$'+subfield).join(podpola)
                    print(newstring)
                    nowe[k]=newstring
                    #print(podpola)
                    newstring=''
            except:
                None
            

                       
            
            #new2=new+'$b'+a995
            
        #new=v.replace("$0(VIAF)", "$1http://viaf.org/viaf/")
        
            
            
    rekordy.append(nowe)

    


def change_subffield_value(dictrec,field,subfield,string,position):
    
    pattern=r'(?<=\${}).*?(?=\$|$)'.format(subfield)
    
    rekordy=[]
    for listy in tqdm(dictrec):
        nowe={}
        for k,v in listy.items():
            nowe[k]=v
            if k==field:
                #print(v)
                podpole995_lista=re.findall(pattern, v)
                try:
                    podpole995=podpole995_lista[position-1]
                    #print(podpole995)
                    podpola=v.split('$'+subfield)
                    len_podpola=len(podpola)
                    if len_podpola==position+1:
                        
                        lastsubfield=podpola[-1]
                        last_list=lastsubfield.split('$')
                        last_string_to_change=last_list[0]
                        newstring=last_string_to_change.replace(podpole995,string )
                        last_list[0]=newstring
                        newstring='$'.join(last_list)
                        podpola[-1]=newstring
                        newstring=('$'+subfield).join(podpola)
                        nowe[k]=newstring
                        newstring=''
                    
                    else:
                        find_sub=podpola[position]
                        #print(find_sub)
                        
                        newstring=find_sub.replace(podpole995,string )
                        podpola[position]=newstring
                        newstring=('$'+subfield).join(podpola)
                        print(newstring)
                        nowe[k]=newstring
                        #print(podpola)
                        newstring=''
                except:
                    None
                
    
                           
                
                #new2=new+'$b'+a995
                
            #new=v.replace("$0(VIAF)", "$1http://viaf.org/viaf/")
            
                
                
        rekordy.append(nowe)
    return rekordy
string="PLiteracka"                
x=change_subffield_value(dictrec,'995','a',string,1)

