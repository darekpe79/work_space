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
from pprint import pprint
import pprint
from time import time




# Zaciąganie odpowiedzi Vaif do słownika

dictionary={}
folderpath = r"F:\Nowa_praca\libri\Iteracja 2021-07\noweViafy-100odpowiedzi_PBL"
os.listdir(folderpath)
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]

for path in tqdm([r"F:\Nowa_praca\baza_biogramy_viaf77.json"]):
    with open(path, 'r', encoding='utf-8') as jsonFile:
    

        jsonObject = json.load(jsonFile)
        for k,v in jsonObject.items():
            dictionary[k]=v
            


#%% Tworzenie wygodnego słownika do pracy - viaf id i warianty nazwisk
#porównanie go z kartoteką wzorcową

    
viaf_Dict={}    
for name, records in tqdm(dictionary.items()):
    viaf_Dict[name]={}
    if records:
        for record in records:
            Viaf=record['record']['recordData']['viafID']
            Data=record['record']['recordData']['mainHeadings']['data']
            names=[]
            if type(Data)==list:
                for element in Data:
                    names.append(element['text'])
            else:
                names.append(Data['text'])
            viaf_Dict[name][Viaf]=names
#%%  Wybieranie wartoci po długosci dicta:
nazwaviaf={} 
inne={}   
for name, variant in viaf_Dict.items():
    if len(variant)>=1:
        nazwaviaf[name]=variant
    else:
        inne[name]=variant
        
        
        
        
          

isni_viaf={}
for isni, viafs in tqdm(viaf_Dict.items()):
    if viafs:
        for viaf, names in viafs.items():
            isni_viaf[isni]=viaf
            #print(isni, viaf)
df = pd.read_excel (r"F:\Nowa_praca\fennica\statystyki\kartoteka_wzorcowa — kopia.xlsx")  
df1_identifier_list=[]
for isni, viaf in tqdm(isni_viaf.items()):
    #df1=df['identyfikator'].str.findall(isni)
    #print(df1)
    df1 = df[df['identyfikator'].str.contains(isni)]
    
    #print(df1)
    
    df1['viaf']=viaf
    df1_identifier_list.append(df1)

    
df_identyfikatory_viaf = pd.concat(df1_identifier_list) 
df_identyfikatory_viaf.to_excel('identyfikatory_viaf_wszyscy_finto.xlsx', index=False)      
#%% Nadawanie viafów naszym autorom po identyfikatorze fin11/asteriId
df_wszyscy_autorzy_i_pola = pd.read_excel (r"F:\Nowa_praca\fennica\statystyki\pole600 viafowanie_Arto\xxfield600_ArtoNowe_2ID_najnowsze.xlsx")    


df_identyfikatory_viaf = pd.read_excel (r'F:\Nowa_praca\fennica\statystyki\identyfikatory_viaf_wszyscy_finto.xlsx')  
Asteri_Id_pattern='(?<=ID: ).*?(?=\')'
listaidentyfikatorów=df_identyfikatory_viaf['identyfikator'].tolist()
asteri_list=[]
for asteris in listaidentyfikatorów:
    asteri = re.findall(Asteri_Id_pattern, asteris)
    asteri_list.append(asteri[0])
listaviafow=df_identyfikatory_viaf['viaf'].tolist()
asteri_viaf_dict = dict(zip(asteri_list, listaviafow))
fennica_z_viaf=df_wszyscy_autorzy_i_pola[df_wszyscy_autorzy_i_pola['identyfikator_FIN11'].str.contains('|'.join(asteri_viaf_dict.keys()))]
fennica_z_viaf['viafs']=[asteri_viaf_dict[asteri.strip(',')] for asteri in fennica_z_viaf['identyfikator_FIN11']]

fennica_bez_viaf=df_wszyscy_autorzy_i_pola[~df_wszyscy_autorzy_i_pola.isin(fennica_z_viaf)].dropna(how = 'all')



        

fennica_z_viaf.to_excel('identyfikatory_viaf_nasi_pewni_poIdentyfikatorze_arto_600.xlsx', index=False)    

fennica_bez_viaf.to_excel('wszyscy_autorzy_i_pola_z2ID_bez_pewnych_po_ID_arto_600.xlsx', index=False)

#%%analiza nazwa_pseudo, warianty nazw - rozbicie i nadanie viafow wszystkim po wariantach


nazwa_i_pseudo=df_identyfikatory_viaf['nazwa_i_pseudonim'].tolist()

dict_viaf_nazwa_pseudo = dict(zip(nazwa_i_pseudo,listaviafow))

dictViaf_nazwa_pseudo_rozdzielone={}
for key, value in dict_viaf_nazwa_pseudo.items():
    key1=key.strip("[]'")
    print(key1)
    nazwa1=key1.split('❦')
    for nazwa in nazwa1:
        dictViaf_nazwa_pseudo_rozdzielone[nazwa]=value 
#analiza wariantów nazw i nadanie wiafów  

      
warianty_nazwy_lista=df_identyfikatory_viaf['wariant_nazwy'].tolist()
dict_warianty_nazwy = dict(zip(warianty_nazwy_lista,listaviafow))   
dictViaf_warianty_nazwy_rozdzielone={}
for key, value in dict_warianty_nazwy.items():
    key1=key.strip("[]'")
    print(key1)
    nazwa1=key1.split('❦')
    for nazwa in nazwa1:
        dictViaf_warianty_nazwy_rozdzielone[nazwa]=value   
dic2 = dict(dictViaf_nazwa_pseudo_rozdzielone, **dictViaf_warianty_nazwy_rozdzielone)
warianty_pseudo_excel=pd.DataFrame.from_dict(dic2, orient = 'index')
warianty_pseudo_excel.to_excel('warianty_pseudo_nazwy.xlsx', index=True) 

#%% ładowanie gotowych exceli i sprawdzanie pseudo i nazwy z kartoteki wzrocowej 


df_wszyscy_autorzy = pd.read_excel(r"F:\Nowa_praca\fennica\statystyki\pole600 viafowanie_fennica\wszyscy_autorzy_i_pola_z2ID_bez_pewnych_po_ID_i_po_pseudonazwa_po_nazwisko_data_pole_i_respond_ratio_x2600_Fennica.xlsx")
warianty_pseudo_nazwy = pd.read_excel(r"F:\Nowa_praca\fennica\statystyki\pole600 viafowanie_Arto\viafs_respond1_ratio1_pole_600_arto_bezprzecinkow_bezspacji.xlsx")
warianty_pseudo_nazwy_lista=warianty_pseudo_nazwy['name'].tolist()

Viaf_lista=warianty_pseudo_nazwy['Viaf'].tolist()
dic2=dict(zip(warianty_pseudo_nazwy_lista, Viaf_lista))


mask = df_wszyscy_autorzy['nazwiskodata2'].apply(lambda x: any(item for item in warianty_pseudo_nazwy_lista if item == str(x)))
df3 = df_wszyscy_autorzy[mask]
df4=df_wszyscy_autorzy[~mask]
df3['viafs']=[dic2[name] for name in df3['nazwiskodata2']]
df3.to_excel('wszyscy_autorzy_i_pola_z2ID_respond1_trzeci_POLE 600_bezprzecinkow_spacji_Fennica.xlsx', index=False)    

df4.to_excel('wszyscy_autorzy_i_pola_z2ID_bez_pewnych_po_ID_i_po_pseudonazwa_po_nazwisko_data_pole_i_respond_ratio_x3600_Fennica.xlsx', index=False)


# INNY SPOSÓB:
wszyscy_autorzy_viaf_z_700_tys_list=[]

for nazwa, viaf in tqdm(dic2.items()):
    
    if [df_wszyscy_autorzy_i_pola_700_tys[df_wszyscy_autorzy_i_pola_700_tys['nazwisko data 2'] == nazwa]]:
        wszyscy_autorzy_viaf_z_700_tys=df_wszyscy_autorzy_i_pola_700_tys[df_wszyscy_autorzy_i_pola_700_tys['nazwisko data 2'] == nazwa]
        wszyscy_autorzy_viaf_z_700_tys['viaf']=viaf
        wszyscy_autorzy_viaf_z_700_tys['nazwa/pseudo']=nazwa
        wszyscy_autorzy_viaf_z_700_tys_list.append(wszyscy_autorzy_viaf_z_700_tys)
wszyscy_autorzy_i_pola_z2ID_bez_duplikatów_z_viaf=pd.concat(wszyscy_autorzy_viaf_z_700_tys_list)
wszyscy_autorzy_i_pola_z2ID_bez_duplikatów_z_viaf.to_excel('wszyscy_autorzy_i_pola_z2ID_bez_duplikatów_z_viaf2.xlsx', index=False) 



#%% dodanie wiafów z ratio 1,  datą potem pojedynczych:
df_wszyscy_autorzy= pd.read_excel(r"F:\Nowa_praca\fennica\statystyki\wszyscy_autorzy_i_pola_z2ID_bez_duplikatów_bez_przecinków.xlsx")
nazwisko_data = pd.read_excel(r"F:\Nowa_praca\fennica\statystyki\viafs_pojedynczy_bez_przecinkow.xlsx")    
nazwisko_data_lista=nazwisko_data['name'].tolist()
Viaf_lista=nazwisko_data['Viaf'].tolist()    
dic2=dict(zip(nazwisko_data_lista, Viaf_lista)) 
  
wszyscy_autorzy_Viaf=[]
#sprawdzam=[e for e in nazwisko_data_lista2 if e in nazwisko_data_lista]

for nazwa, viaf in tqdm(dic2.items()):
    
    if [df_wszyscy_autorzy[df_wszyscy_autorzy['nazwisko data 2'] == nazwa]]:
            
        wszyscy_autorzy=df_wszyscy_autorzy[df_wszyscy_autorzy['nazwisko data 2'] == nazwa]
        wszyscy_autorzy['viaf']=viaf
        wszyscy_autorzy['nazwa/pseudo']=nazwa
        wszyscy_autorzy_Viaf.append(wszyscy_autorzy)
wszyscy_autorzy_i_pola_z2ID_bez_duplikatów_z_viaf=pd.concat(wszyscy_autorzy_Viaf)
wszyscy_autorzy_i_pola_z2ID_bez_duplikatów_z_viaf.to_excel('Ratio1_viafs_pojedynczy.xlsx', index=False) 





   
#%% dopasowywanie viafow do nazwy RATIO :            
matchesdict={}
for name, viafs in tqdm(nazwaviaf.items()):
    matchesdict[name]=[]
    if not (re.search(r'\d', name)):
    
        if viafs:
            for viaf, names in viafs.items():
                #wrocić do name jutro (02.02.2022)
                namestrip=name.strip()
                
                matches = get_close_matches(namestrip, [onename.strip('. ') for onename in names],n=1, cutoff=0.9)
                #print(name, matches, viaf)
                
                
                if matches:
                    
                    
                    ratio=matcher(name, matches[0] )
                    #print(ratio)
                    matchesdict[name].append([viaf, matches+[ratio],names])
    
                    
        if not matchesdict[name] or len(matchesdict[name])>1:
            del matchesdict[name]

            
starttime=time()
df = pd.DataFrame(columns =['name', 'viaf', 'close_match', 'ratio','names'])  
rows=[]
for key, value in matchesdict.items():

    
    
    for element in value:
        row={'name':key, 'viaf':element[0], 'close_match':element[1][0], 'ratio':element[1][1],'names':element[2]}
        rows.append(row)
df=df.append(rows, ignore_index=True)
endtime=time()
print(endtime-starttime)

                    



df.to_excel('viafs_respond_ratioPBL_caloscbezspacji.xlsx', index=False)            
#%% Zaczytanie pliku viafs i selekcja osób oraz viafów:
df = pd.read_excel (r"F:\Nowa_praca\fennica\statystyki\viafs_ratio1.xlsx")
listaViafs=df['viaf'].tolist()
unikatowe_viaf=[]
duplikaty_viaf=[]
for Viaf in listaViafs:
    if Viaf not in unikatowe_viaf:
        unikatowe_viaf.append(Viaf)
    else:
        duplikaty_viaf.append(Viaf)
pojedyncze_viafy=[pojedviaf for pojedviaf in unikatowe_viaf if pojedviaf not in duplikaty_viaf]
listaimion=df['name'].tolist()
imie_data=[]
unikatowe=[]
duplikaty=[]
for imie in tqdm(listaimion):
    
    if (re.search(r'\d', imie)):
        imie_data.append(imie)
    if imie not in unikatowe:
        unikatowe.append(imie)
    else:
        duplikaty.append(imie)
        
pojedyncze=[unique for unique in tqdm(unikatowe) if unique not in duplikaty] 
df_pojedyncze_list=[]
for jedno in pojedyncze:
    
    df1=df[(df['name'].isin([jedno]))]
    
    df_pojedyncze_list.append(df1)
df_pojedyncze = pd.concat(df_pojedyncze_list)        

df_imie_data_list=[]
for im_dat in imie_data:
    df1=df[(df['name'].isin([im_dat]))]
    
    df_imie_data_list.append(df1)
df_imie_data = pd.concat(df_imie_data_list)     
df_pojedyncze.to_excel('viafs_pojedynczy.xlsx', index=False)  
df_imie_data.to_excel('viafs_imie_data.xlsx', index=False)  
#%%tworzenie klastrów podobieństw

def cluster_strings(strings, similarity_level):
    clusters = {}
    output = {}
    for string in tqdm((x.strip() for x in strings)):
        if string in clusters.values():
            clusters[string].append(string)
        else:
            match = get_close_matches(string, clusters.keys(), cutoff=similarity_level)
            if match:
                clusters[match[0]].append(string)
            else:
                clusters[string] = [string]
    for i, (key, value) in enumerate(clusters.items()):
        output[i] = value
    return output
autor_klaster=cluster_strings(name_list, 0.65)
    

    
    
  
           

#%%            
with open(r"F:\Nowa_praca\fennica\statystyki\clusters.json", 'w', encoding='utf-8') as jfile:
    json.dump(autor_klaster, jfile, ensure_ascii=False, indent=4)        
            


#%%
with open(r"F:\Nowa_praca\fennica\statystyki\Nowy folder\bazaviaf1.json",'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
name_list=[]
duplicate=[]
for k,v in jsonObject.items():
    if k not in name_list:
        name_list.append(k)
    else:
        duplicate.append(k)
    