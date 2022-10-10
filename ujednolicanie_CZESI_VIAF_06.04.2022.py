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
from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
from time import time
from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()


files=["F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_05.04.2022/cz_chapters.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_05.04.2022/cz_articles0.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_05.04.2022/cz_articles1.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_05.04.2022/cz_articles2.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_05.04.2022/cz_articles3.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_05.04.2022/cz_articles4.mrk",
"F:/Nowa_praca/NOWI CZESI_FENNICAPBL_BN_05.04.2022/cz_books.mrk"]
pattern4=r'(?<=\$7).*?(?=\$|$)'
id_czech=[]
for plik in files:

    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)

    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key=='700' or key=='100' or key=='600':
    
                v=val.split('❦')
                for vi in v:
                    id_c=re.findall(pattern4, vi)
                    
                    if id_c:
                        id_c=id_c[0]
                        if id_c not in id_czech:
                            id_czech.append(id_c)
                    else:
                        id_c='brak'
                    
                    


df = pd.DataFrame(id_czech,columns =['id_czech'])
df.to_excel("id_czech.xlsx", sheet_name='Sheet_name_1') 
#wydobycie nazw osobowych:
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$d).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
original_names=[]
name_date_list=[]
viafs_list=[]
for names in tqdm(val100):
#    original_names.append(names)
    
    name = re.findall(pattern3, names)
    dates = re.findall(pattern4, names)
    viaf=re.findall(pattern5, names)
    #print  (dates)
    
    if name:
        name=name[0]
    else:
        name='brak'
    
    if dates:
        
        datesstr=re.sub('\)|\(','',dates[0])
        datesstr=datesstr.strip('.')
        
    else:
        datesstr=''
    if viaf:
        viaf=viaf[0]
    else:
        viaf='brak'
    #print(datesstr)
    name_date=name.strip('.')+' '+datesstr
    name_date_list.append(name_date)
    viafs_list.append(viaf)
    
df2 = pd.DataFrame (list(zip(original_names,name_date_list,viafs_list)), columns =['100','nazwisko data','viaf' ]) 
df2.to_excel("Brakujace_books_PBN_dobre_995_do_ujednolicania_z brak.xlsx", sheet_name='Sheet_name_1')
df2 = df2[df2.viaf != 'brak']
df2.to_excel("Brakujace_books_PBN_dobre_995_do_ujednolicania.xlsx", sheet_name='Sheet_name_1')
lista_viaf=df2.viaf.tolist()

    
mylistViafwithoutDuplicArticlesPBL = list( dict.fromkeys(lista_viaf) )
#all_data=pd.read_excel(r"C:\Users\darek\ujednolicanie\wszystkoBN+PBL.xlsx")
#listaallBN=all_data.viaf.tolist()


#dopsrawdzenia=[i for i in mylistViafwithoutDuplicArticlesPBL if i not in listaallBN]
#probe=[i for i in mylistViafwithoutDuplicarticles if i  in mylistViafwithoutDuplicBooks]
nazwisko_data_lista=df2['nazwisko data'].tolist()
viaf_nazwa = dict(zip(lista_viaf,nazwisko_data_lista))
viaf_nazwa_df=pd.DataFrame.from_dict(viaf_nazwa, orient='index')
viaf_nazwa_df.to_excel("Przed_ujednolicanko_calosc_arto_dobre_995.xlsx", sheet_name='Sheet_name_1') 

#%%
mylistViafwithoutDuplicArticlesBN=pd.read_excel(r"C:\Users\darek\ujednolicanko_blad_Czesi_wszystko.xlsx",sheet_name=0)
dopsrawdzenia=mylistViafwithoutDuplicArticlesBN['id_czech'].tolist()
def lang_detect(texts):
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    # russian
    if re.search("[\u0400-\u0500]+", texts):
        return "ru"
    return None
zle_viafy=[]
nowe_viaf_nazwy={}
for viaf in tqdm(dopsrawdzenia):

    
    
    
    pattern_daty=r'(([\d?]+-[\d?]+)|(\d+-)|(\d+\?-)|(\d+))(?=$|\.|\)| )'
    query='http://viaf.org/viaf/sourceID/NKC%7C{}/viaf.json'.format(viaf)
    
    
    try:
        r = requests.get(query)
        r.encoding = 'utf-8'
        response = r.json()
        viafy=response['viafID']
        warianty=response['mainHeadings']['data']

        
        wszystko=[]
        if type(warianty) is list:
            
            
            for wariant in warianty:
 
                nazwa=wariant['text']
                if 'orcid' in nazwa.lower() or ad.is_arabic(nazwa) or ad.is_cyrillic(nazwa) or ad.is_hebrew(nazwa) or ad.is_greek(nazwa) or lang_detect(nazwa) :
                    continue
                #print(nazwa)
                zrodla=wariant['sources']['s']
    
                if type(zrodla) is list:
                    liczba_zrodel=len(zrodla)
                else:
                    liczba_zrodel=1
                        
                
                daty = re.findall(pattern_daty, nazwa)
                #print(daty)
                
                if daty:
                    for index,grupa in enumerate(daty[0][1:]):
                        if grupa:
                            priorytet=index
                            break
                            
                else: 
                    priorytet=5
                    
                jeden_wariant=[nazwa,priorytet,liczba_zrodel] 
                wszystko.append(jeden_wariant)
            best_option=wszystko[0]
                
                
            for el in wszystko:
                
               if el[1]<best_option[1]:
                   
                   best_option=el
               elif el[1]==best_option[1]:
                   
                   if el[2]>best_option[2]:
                       best_option=el
                           
        else:
            best_option=[warianty['text']]
              
        nowe_viaf_nazwy[viaf]=[viafy,best_option[0]]
    
    except KeyboardInterrupt as e:
        raise e
    except:
        zle_viafy.append(viaf)
        
excel=pd.DataFrame.from_dict(nowe_viaf_nazwy, orient='index')
excel.to_excel("ujednolicanko_Czesi5.xlsx", sheet_name='Sheet_name_1') 
df = pd.DataFrame(zle_viafy)
df.to_excel("ujednolicanko_blad_Czesi5.xlsx", sheet_name='Sheet_name_1') 


#%% mapowanie tabel bez viaf plus viafy    
        
dataframe_viafy=pd.read_excel(r"F:\Nowa_praca\libri\Iteracja 2021-07\do Viafoawnia PBL_ze 100_opdowiedzi\viafs_respond_ratioPBL_caloscbezspacji_TYLKO_JEDYNKI.xlsx") 
listaViafs=dataframe_viafy['viaf'].tolist()
lista_names=dataframe_viafy['name'].tolist()
imie_viaf = dict(zip(lista_names, listaViafs))       
df_wszyscy_autorzy = pd.read_excel(r"F:\Nowa_praca\libri\Iteracja 2021-07\do Viafoawnia PBL_ze 100_opdowiedzi\całPBL_dopracy-Viafy_jedna odp_i_bez_viaf.xlsx")
df_wszyscy_autorzy['Viaf2'] = df_wszyscy_autorzy['nazwisko data'].map(imie_viaf)
df_wszyscy_autorzy.to_excel("całyPBL700_zviaf.xlsx", sheet_name='Sheet_name_1') 
df_wszyscy_autorzy_NOTnan=df_wszyscy_autorzy.dropna()
#df_wszyscy_autorzy_nan=df_wszyscy_autorzy[df_wszyscy_autorzy['Viaf'].isna()]

df_wszyscy_autorzy_NOTnan.to_excel("700_pbl_marc_books_700_2021-08-05BEZ_DUPLIKATOW_TYlko_VIAFY.xlsx", sheet_name='Sheet_name_1') 

                            
                    
                    
                            
                            
                            
                
                
                
                
                
                
                
                
                
                
                
                
                

                

                
                

        



#print(json.dumps(response, indent=4))