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
#from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
from time import time
from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()
from concurrent.futures import ThreadPoolExecutor
import threading
import xlsxwriter


paths=["F:/Nowa_praca/24.05.2022Marki/arto.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_articles.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_books.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_chapters.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles0.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles1.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles2.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles3.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_articles4.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_books.mrk",
"F:/Nowa_praca/24.05.2022Marki/cz_chapters.mrk",
"F:/Nowa_praca/24.05.2022Marki/fennica.mrk",
"F:/Nowa_praca/24.05.2022Marki/PBL_articles.mrk",
"F:/Nowa_praca/24.05.2022Marki/PBL_books.mrk"]

paths2=["F:/Nowa_praca/24.05.2022Marki/PBL_books.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_articles.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_books.mrk",
"F:/Nowa_praca/24.05.2022Marki/BN_chapters.mrk",
"F:/Nowa_praca/24.05.2022Marki/PBL_articles.mrk"]

pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$v).*?(?=\$|$)'
#pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'


output={'600v':{},'610v':{},'611v':{},'630v':{},'648v':{},'650v':{},'651v':{},'655a':{},'655v':{}}

#val100=[]
for plik in paths2:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    

    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key in ('600','610','611','630','648','650','651','655'):
                v=val.split('❦')
                for names in v:
                    if key=='655':
                        podpole_a_marc=re.findall(pattern3, names)
                        for element in podpole_a_marc:
                            if element not in output[key+'a']:
                                output[key+'a'][element]=1
                            else:
                                output[key+'a'][element]+=1
                                
                        

                        
                    else:
                        podpole_v_marc=re.findall(pattern4, names)
                        for element in podpole_v_marc:
                            if element not in output[key+'v']:
                                output[key+'v'][element]=1
                            else:
                                output[key+'v'][element]+=1
output2=[]
for key, value in output.items():
    for k, v in value.items():
        output2.append([key,k,v])
        
    

df2=pd.DataFrame(output2,columns=['field','genre','counter']) 
df2.to_excel("statsy_genre_form_Czech.xlsx", sheet_name='Sheet_name_1', index=False)
#%% 
pos006={'a':'Language material',
'c' : 'Notated music',
'd' : 'Manuscript notated music',
'e' : 'Cartographic materia',
'f' : 'Manuscript cartographic material',
'g' : 'Projected medium',
'i' : 'Nonmusical sound recording',
'j' : 'Musical sound recording',
'k' : 'Two-dimensional nonprojectable graphic',
'm' : 'Computer file',
'o' : 'Kit',
'p' : 'Mixed materials',
'r' : 'Three-dimensional artifact or naturally occurring object',
't' : 'Manuscript language material'}
pos007={'a' : 'Monographic component part',
'b' : 'Serial component part',
'c' : 'Collection',
'd' : 'Subunit',
'i' : 'Integrating resource',
'm' : 'Monograph/Item',
's' : 'Serial'}
output={}
output2={}
for plik in paths2:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)

    for element in dictrec: 
        for key, value in element.items():
            if key=='LDR':
                letters=value[6]+value[7]
                if letters not in output2:
                    output2[letters]=1
                else:
                    output2[letters]+=1
                    
                
                
                search_pos=pos006[value[6]]+', '+pos007[value[7].lower()]
                
                
                if search_pos not in output:
                    output[search_pos]=1
                    
                else:
                    output[search_pos]+=1
excel=pd.DataFrame.from_dict(output, orient='index')
excel2=pd.DataFrame.from_dict(output2, orient='index')
with pd.ExcelWriter('FORMAT_LDR_Polish.xlsx', engine='xlsxwriter') as writer:
    excel.to_excel(writer, sheet_name='format1')
    excel2.to_excel(writer, sheet_name='format2')

#%% wyciaganie 084

yso_dict={'82':['\\\\$aLyrical poetry$leng','\\\\$aLiryka$lpol','\\\\$aLyrická poezie$lcze','\\\\$aRunot$lfin'], 
          '84':['\\\\$aFiction$leng','\\\\$aEpika$lpol', '\\\\$aProza$lcze','\\\\$aKertomakirjallisuus$lfin'],
          '83':['\\\\$aDrama$leng','\\\\$aDramat$lpol', '\\\\$aDrama$lcze','\\\\$aNäytelmät$lfin']}

pattern3=r'(?<=\$a).*?(?=\$|$)' 
nieobrobione=[]
output=[]
rekordy=[]
for plik in [r"F:\Nowa_praca\24.05.2022Marki\obrobka84.mrk"]:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    
    
    for rekord in tqdm(dictrec):
        
        #if '084' not in rekord:
            #nieobrobione.append(rekord)
        
        neeew=[]
        nowe={}
        for key, val in rekord.items():
            nowe[key]=val
            rekord_num=rekord['001']
            
            if key=='084':
                #output[rekord_num]=[]
                #print(val)
                v=val.split('❦')
                #new_value_list=[]
                for number in v:
                    if len(number)>=1:
                        value_084=re.findall(pattern3, number)
                        #output[rekord_num].append(value_084[0])
                        numbers=value_084[0][:2]
                        if numbers in yso_dict:
                            print(yso_dict[numbers])
                 #           new_value_list.extend(yso_dict[numbers])
                            neeew.extend(yso_dict[numbers])
                            
                #if new_value_list: 
                                                   
                #    nowe['380']='❦'.join(new_value_list)
                    
       # rekordy.append(nowe)            
                    
                #else:
                #    nieobrobione.append(rekord)
        if neeew:
            if '380' in nowe:
                output.append(rekord)
                print(nowe['380'])
                nowe['380']=nowe['380']+'❦'+'❦'.join(neeew)
            else:
                nowe['380']='❦'.join(neeew)
        else:
            nieobrobione.append(rekord)
            
        rekordy.append(nowe)   

to_file ('obrobka84.mrk', rekordy)   
to_file ('bez_380_arto.mrk', nieobrobione)          

excel=pd.DataFrame.from_dict(yso_dict, orient='index')
excel.to_excel('084.xlsx', sheet_name='format1')
output2={}
output2['rekord_num']=[]
output2['rekord_num'].append('f')
excel=pd.DataFrame.from_dict(output2, orient='index')

dicti={'k':'l'}
dicti['c']='d'
eng
pol
fin
cze                        
#proza, drama, lyrická poezie, sekundární literatura
 #%%                    
def unique(list1):
  
    
    unique_list = []
      
    
    for x in list1:
        
        if x not in unique_list:
            unique_list.append(x)
    return unique_list
BN_dict={'poezja':['\\\\$aLyrical poetry$leng','\\\\$aLiryka$lpol','\\\\$aLyrická poezie$lcze','\\\\$aRunot$lfin'],
         'wiersze':['\\\\$aLyrical poetry$leng','\\\\$aLiryka$lpol','\\\\$aLyrická poezie$lcze','\\\\$aRunot$lfin'],
          'powieść':['\\\\$aFiction$leng','\\\\$aEpika$lpol', '\\\\$aProza$lcze','\\\\$aKertomakirjallisuus$lfin'],
          'proza':['\\\\$aFiction$leng','\\\\$aEpika$lpol', '\\\\$aProza$lcze','\\\\$aKertomakirjallisuus$lfin'],   
          'dramat (rodzaj)':['\\\\$aDrama$leng','\\\\$aDramat$lpol', '\\\\$aDrama$lcze','\\\\$aNäytelmät$lfin']}

pattern3=r'(?<=\$a).*?(?=\$|$| )' 
pattern4=r'(?<=\$a)Literatura podróżnicza(?=\$|$| )|(?<=\$a).*?(?=\$|$| )'
nieobrobione=[]
output=[]
rekordy=[]
for plik in [r"F:\Nowa_praca\24.05.2022Marki\cos.mrk"]:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    
    
    for rekord in tqdm(dictrec):
        new_value_list=[]

        
        
        nowe={}
        for key, val in rekord.items():
            nowe[key]=val
            

            
            if key=='380':
                
                #print(val)
                v=val.split('❦')
                
                for number in v:
                    if len(number)>=1:
                        
                        value_655=re.findall(pattern3, number)
                        
                        
                        if value_655:
      
                            szukane=value_655[0].lower().strip('. ')
                            if szukane in BN_dict:
                                #print(yso_dict[numbers])
                                new_value_list.extend(BN_dict[szukane])
                            
        if new_value_list:
            if '380' in nowe:
                output.append(rekord)
                print(nowe['380'])
                list_set=unique(new_value_list+[nowe['380']])
                #list_set = list(set(new_value_list+[nowe['380']])) 
                nowe['380']='❦'.join(list_set)
            else:
                list_set = list(set(new_value_list))                                  
                nowe['380']='❦'.join(list_set)
        else:
            nieobrobione.append(rekord)
            
                
        rekordy.append(nowe)   

to_file ('probaBN.mrk', rekordy)   
to_file ('probabez_380_BN_articles.mrk', nieobrobione)                       
                    
                
pattern3=r'(?<=\$a)Literatura podróżnicza(?=\$|$| )|(?<=\$a).*?(?=\$|$| )'
val=re.findall(pattern3,number)               
 
            
            
       

form_set=set()
genre_set=set()
subject_set=set()                  
for plik in [r"F:\Nowa_praca\24.05.2022Marki\PBL_books.mrk"]:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    

    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key in ('655'):
                
    
                v=val.split('❦')
                for names in v:
                    form_set.add((names))
s = list(form_set)
df=pd.DataFrame(s)
df.to_excel('pbl655.xlsx', sheet_name='format_df1')

#%%    
format_set=set()
genre_set=set()
subject_set=set()                  
for plik in paths:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    

    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            if key in ('LDR', '008', '007','086', '111','711', '245', '502'):
                
    
                v=val.split('❦')
                for names in v:
                    format_set.add((key,names))
            elif key in ('600','610','611','630','648','650','651','655'):
                v=val.split('❦')
                for names in v:
                    if key=='650':
                        genre_set.add((key, names))
                        subject_set.add((key,names))
                    else:
                        genre_set.add((key,names))
                    
                    
                    
                    

                            
format_df=pd.DataFrame(format_set) 
genre_df=pd.DataFrame(genre_set) 
subject_df=pd.DataFrame(subject_set) 
df_1 = format_df.iloc[0:1000000,:]
df_2 = format_df.iloc[1000000:2000000,:]
df_3= format_df.iloc[2000000:,:]

with pd.ExcelWriter('wszystko.xlsx', engine='xlsxwriter') as writer:
    df_1.to_excel(writer, sheet_name='format_df1')
    df_2.to_excel(writer, sheet_name='format_df2')
    df_3.to_excel(writer, sheet_name='format_df3')
    genre_df.to_excel(writer, sheet_name='genre_df')
    subject_df.to_excel(writer, sheet_name='subject_df')
#%%
genre = pd.read_excel (r"C:\Users\darek\format,form,genre,subject\wszystko_genre_format_subject.xlsx", sheet_name=3)
genre_field=dict(zip(genre.genre.tolist(),genre.field.tolist()))
genre_from_facet = pd.read_excel (r"C:\Users\darek\format,form,genre,subject\wszystko_genre_format_subject.xlsx", sheet_name=5)
genre_list_facet=genre_from_facet.genre_facet.tolist()

dictionary={}

for k,v in genre_field.items():
    for element in genre_list_facet:
        if element in k:
            dictionary[v]=tuple
            
dictionary={}
for x in 'lalalaefwerf':
                
    
    


#%%
dictionary={}    
sets={'1', '2', '3','4'}
sets2={1, 2, 4}

viaf_nazwa_df=pd.DataFrame(sets)
viaf_nazwa_df.columns = ['columnName']   
viaf_nazwa_df2=pd.DataFrame(sets2) 
viaf_nazwa_df2.columns = ['columnName2'] 

frames = [viaf_nazwa_df, viaf_nazwa_df2]
  
result = pd.concat(frames).fillna('')

d1 = pd.DataFrame(zip_longest(sets2,sets),columns=['col1','col2'])  
d1['col1']=sets         
                            
                        
        
x = set()

x.add((4, '4a'))
                        
                        


                    
                    
                    
                    
                    
                    

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

                            
                    
                    
                            
                            
                            
                
                
                
                
                
                
                
                
                
                
                
                
                

                

                
                

        



#print(json.dumps(response, indent=4))