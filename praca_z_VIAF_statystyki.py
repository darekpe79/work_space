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

from itertools import zip_longest
import regex as re
import requests
from time import time
#from alphabet_detector import AlphabetDetector
#ad = AlphabetDetector()
from concurrent.futures import ThreadPoolExecutor
import threading

paths=["D:/Nowa_praca/marki_04_01_2023/cz_articles1_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles2_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles3_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles4_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_books_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_chapters_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/fennica_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/pbl_articles_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/pbl_books_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/arto_2022-09-02_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/bn_articles_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/bn_books_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/bn_chapters_2022-08-26_30-12-2022.mrk",
"D:/Nowa_praca/marki_04_01_2023/cz_articles0_2022-08-26_30-12-2022.mrk"]
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$d).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
zViaf={}
bezviaf={}
viafs_list=[]

#val100=[]
for plik in paths:
  #  lista=mark_to_list(plik)
  #  dictrec=list_of_dict_from_list_of_lists(lista)
    dictrec=list_of_dict_from_file(plik) 

    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key=='700' or key=='100' or key=='600':
    
                
                for names in val:
                    
                    #val100.append(vi)

                    
                    
                    name = re.findall(pattern3, names)

                    viaf=re.findall(pattern5, names)

                    
                    if name:
                        name=name[0]
                    else:
                        name='brak'
                    


                    if viaf:
                        viaf=viaf[0]
                    else:
                        viaf='brak'
                    if viaf=='brak':
                        if name not in bezviaf:
                            bezviaf[name]=1
                        else:
                            bezviaf[name]+=1
                    else:
                        if name not in zViaf:
                            zViaf[name]=[viaf,1]
                        else:
                            zViaf[name][1]+=1
                            
viaf_nazwa_df=pd.DataFrame.from_dict(zViaf, orient='index') 
bez_viaf_nazwa_df=pd.DataFrame.from_dict(bezviaf, orient='index')

viaf_nazwa_df.to_excel("wszystko_z_VIAF.xlsx", sheet_name='Sheet_name_1')
bez_viaf_nazwa_df.to_excel("wszystko_bez_VIAF.xlsx", engine='xlsxwriter')

dictionary={}    
sets={'1', '2', '3'}
sets2={1, 2, 4}

viaf_nazwa_df=pd.DataFrame(sets)
viaf_nazwa_df.columns = ['columnName']   
viaf_nazwa_df2=pd.DataFrame(sets2) 
viaf_nazwa_df2.columns = ['columnName2'] 
d1 = pd.DataFrame(columns=['col1','col2'])  
d1['col1']=sets         
                            
                        
        

                        
                        


                    
                    
                    
                    
                    
                    

#%%
df = pd.DataFrame(val100,columns =['100_700_600'])
df.to_excel("calosc_Brakujace_books_PBN_dobre_995_do_ujednolicania.xlsx", sheet_name='Sheet_name_1') 
#wydobycie nazw osobowych:
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$d).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
original_names=[]
name_date_list=[]
viafs_list=[]
for names in tqdm(val100):
    original_names.append(names)
    
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
df2=df2.drop_duplicates()
df2.to_excel("caly_PBL.xlsx", sheet_name='Sheet_name_1')
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

def uniform(viaf):
    
    

    pattern_daty=r'(([\d?]+-[\d?]+)|(\d+-)|(\d+\?-)|(\d+))(?=$|\.|\)| )'
    query='https://www.viaf.org/viaf/{}/viaf.json'.format(viaf)
    
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
              

        nowe_viaf_nazwy=(viaf,viafy,best_option[0])
    
    except:
        nowe_viaf_nazwy=(viaf,'zly','zly')
    return nowe_viaf_nazwy

mylistViafwithoutDuplicArticlesBN=pd.read_excel(r"F:\Nowa_praca\libri\Iteracja 2021-07\ujednolicanie-zderzenie_tabel-Czarek-ja__BŁĘDY\caly_PBL.xlsx", sheet_name=2)
dopsrawdzenia=mylistViafwithoutDuplicArticlesBN['viaf'].tolist()

with ThreadPoolExecutor(max_workers=50) as executor:
    results=list(tqdm(executor.map(uniform,dopsrawdzenia),total=len(dopsrawdzenia)))

#threading.active_count()
lock=threading.Lock()
with lock:
    x+=result


excel=pd.DataFrame(results)
excel.to_excel("PBL_brakujacy2.xlsx", sheet_name='Sheet_name_1') 
df = pd.DataFrame(zle_viafy)
df.to_excel("ujednolicanko_blad_arto_fennica_po_weryf.xlsx", sheet_name='Sheet_name_1') 


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

                            
#%% Ludki nowe 05.12.2022
paths=["D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_chapters_2022-09-02.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles0_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles1_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles2_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles3_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_articles4_2022-08-26.mrk",
"D:/Nowa_praca/marki_02.09.2022/marki_02.09.2022/cz_books_2022-08-26.mrk"]
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$d).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
zViaf={}
bezviaf={}
viafs_list=[]

allnames={}
for plik in paths:
    dictrec=list_of_dict_from_file(plik)
    

    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key=='700' or key=='100' or key=='600':
    
                
                for v in val:
                    
                    #val100.append(vi)

                    
                    date = re.findall(pattern4, v)
                    name = re.findall(pattern3, v)

                    viaf=re.findall(pattern5, v)

                    
                    if name:
                        name=name[0]
                    else:
                        name='brak'
                    


                    if viaf:
                        viaf=viaf[0]
                    else:
                        viaf='brak'
                    if date:
                        date=date[0]
                    else: 
                        date='brak'
                    
                    if name+' '+date+' '+viaf not in allnames:
                            allnames[name+' '+date+' '+viaf]=[1,name,date,viaf]
                    else:
                        allnames[name+' '+date+' '+viaf][0]+=1
                        
viaf_nazwa_df=pd.DataFrame.from_dict(allnames, orient='index') 


viaf_nazwa_df.to_excel("NAMES_ALL_Cze.xlsx", engine='xlsxwriter')    
                    
                            
                            
                            
                
                
                
                
                
                
                
                
                
                
                
                
                

                

                
                

        



#print(json.dumps(response, indent=4))