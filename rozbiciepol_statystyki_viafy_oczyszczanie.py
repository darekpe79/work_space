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
translator = Translator()

#%% Proba na jednym pliku
plik=r"F:\Nowa_praca\libri\Iteracja 2021-07\pbl_marc_articles2021-8-4!.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)
val336=[]
for rekord in dictrec:
    for key, val in rekord.items():
        for field in val.split('â¦'):
            if key=='336':
                val336.append(val)
                
#%% ladowanie wszystkich plikow
folderpath = r'F:\Nowa_praca\fennica\odsiane_z_viaf_100_700_600arto' 
os.listdir(folderpath)
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
all_files =[]
all_files1=[]
for path in filepaths:
    with open(path, 'r', encoding='utf-8') as f:
        file = f.readlines()
        all_files.append(file)
        all_files1.extend(file)
records=[]
record=[]        
for line in all_files1:
    if line != '\n':
        record.append(line)
    else:
        records.append(record)
        record = []
        
dictrec=list_of_dict_from_list_of_lists(records)


#%%Szukanie typów, których nie chcę 
dysk_audio=[]
muzyka=[]
obrazek=[]
zapis_nutowy=[]
program_komputerowy=[]
dwuwymiarowy_obraz=[]
for myDict in tqdm(dictrec):
    for key, value in myDict.items():
            if key=='336' and 'teksti' not in value:
                if 'CD-äänilevy' in value:
                    dysk_audio.append(myDict)
                elif 'esitetty musiikki' in value or 'musiikki (esitetty)' in value or 'ääni' in value:
                    muzyka.append(myDict)
                elif 'Kuva' in value or 'kuva' in value:
                    obrazek.append(myDict)
                elif 'Musiikki (notaatio)' in value:
                    zapis_nutowy.append(myDict)
                elif 'tietokoneohjelma' in value:
                    program_komputerowy.append(myDict)
                elif 'kaksiulotteinen liikkuva kuva' in value:
                    dwuwymiarowy_obraz.append(myDict)
                    
#if any([e in value for e in [string1, string2...]])                    
                    
dobre=[record for record in dictrec if record not in program_komputerowy and record not in muzyka and record not in obrazek and record not in zapis_nutowy and record not in dwuwymiarowy_obraz ]            

dobre2=[record for record in dictrec if record not in program_komputerowy+muzyka+obrazek+zapis_nutowy+dwuwymiarowy_obraz ] 

#%% statystyki pol
pola={}
for rekord in dictrec:
    for key, val in rekord.items():
        if key not in pola:
            pola[key]=0
        pola[key]+=1
pola=pd.DataFrame.from_dict(pola, orient='index')
pola.to_excel("ilosc_pol_fennica.xlsx", sheet_name='Sheet_name_1') 

#%% wartości poszczegolnych pol
val773=[]
val7732=[]
for rekord in dictrec:
    
    for key, val in rekord.items():
            if key=='773':
                val773.append(val.split('❦'))
                
            for field in val.split('❦'):
                if key=='773':
                    val7732.append(val)
#vallist=[]
#for val in val336:
#    for val2 in val:
#        vallist.append(val2)
    

                
df = pd.DataFrame((val773), columns =['773']) 

    
df.to_excel("PBL_field773_jedna_kolumnaLibri.xlsx", sheet_name='Sheet_name_1') 
#/(?<=a).*?(?=[$])
#(?<=b).*?(?=[$])

#%%Wydobycie autorów z pola 100 i uporządkowanie

#Znalezienie inicjałów:
inicialy=r'(^\p{Lu}\. ?\p{Lu}\.|^\p{Lu}\.) ?$'    

pattern=r'(?<=\$a).*?(?=[\$\.])'
pattern2=r"(?<=\$a).*?(?=[\$\]])"
# własciwy bez nawiasow kwadratowych:
#czyszczenie imion (wydobycie podpola a):    

pattern3=r'(?<=\$a).*?(?=\$|$)'
#do dat urodzenia
pattern4='(?<=\$d).*?(?=\$|$)'#dziala
#replace ., na końcu oprócz inicjału:
pattern5=r'(?<!\p{Lu})[\.,]$'#dziala
#wyciągnięcie identyfikatora fińskiego FIN11 i fi_AsteriN:
pattern6='(?<=FIN11\)).*?(?=\$|$)'#dziala
pattern8='(?<=FI-ASTERI-N\)).*?(?=\$|$)'
#re.sub(pattern4, '', string)
#usunięcie kropki, przecinka po dacie:
pattern7=r'(?<=\d)([\.,])$'
val100=[]
for rekord in dictrec:
    for key, val in rekord.items():
        if key=='100':
            val100.append(val.split('❦'))
lista_imion=[]
purenames=[]
borndates=[] 
nameborndate=[]
error=[] 
original_names=[]
idFennica=[]
idFennica2=[]              
for names in tqdm(vallist):
    
    #for names in lists:
    original_names.append(names)
    ident=re.findall(pattern6, names)
    try:
        identyfikator=ident[0]
    except:
        identyfikator='brak'
    idFennica.append(identyfikator)
    ident2=re.findall(pattern8, names)
    try:
        identyfikator2=ident2[0]
    except:
        identyfikator2='brak'
    idFennica2.append(identyfikator2)
    
    substring = re.findall(pattern3, names)
    lista_imion.append(substring)
    try:
        clear=re.sub(pattern5, '', substring[0]).strip()
    #oczyszczone pole 100 musi mieć więcej niż 4 znaki    
        if len(clear)<=4:
            clear='mało'
            
    except:
        error.append(names)
        clear='błąd'
        
        
    #print(clear)
    purenames.append(clear)
    dates = re.findall(pattern4, names)
    
    borndates.append(dates)
    if dates:
        
        datesstr=re.sub(pattern7,'',dates[0])
        
    else:
        datesstr=''
    #print(datesstr)
    if clear=='mało':
        nameyear=''
    else:
        nameyear=clear.strip()+' '+datesstr
    #print(nameyear)
    nameborndate.append(nameyear.strip())
        
        
        
        
                
df = pd.DataFrame (list(zip(original_names, purenames,borndates, nameborndate, idFennica, idFennica2 )), columns =['100', 'oczyszczone', 'data', 'nazwisko data', 'identyfikator_FIN11', 'identyfikator_FI-ASTERI-N' ]) 
df.to_excel("xxfield600_ArtoNowe_2ID_najnowsze.xlsx", sheet_name='Sheet_name_1') 

#%% łączenie autorów w jeden duzy plik i usuwanie duplikatów (setem)
folder=r'F:\Nowa_praca\fennica\statystyki\field_100_700'
files = os.listdir(folder)
df = pd.DataFrame()
for file in files:
    print(file)
    df = df.append(pd.read_excel(folder+'\\'+file), ignore_index=True) 
df.to_excel(r"F:\Nowa_praca\fennica\statystyki\wszyscy_autorzy_i_pola_z2ID.xlsx", sheet_name='Sheet_name_1')    
Nazwy_set=set(df['nazwisko data'])
import numpy as np

Nazwy_set.remove(np.nan)
Nazwy_set_list=list(Nazwy_set)


# tworzenie pliku z nazwami autorów


with open (r"F:\Nowa_praca\fennica\statystyki\autorzy_Lista.txt", 'w', encoding='utf-8') as listaautorow:
    for element in Nazwy_set_list:
        listaautorow.write(element+'\n')
        



#%% ładowanie autorów i wyciąganie Viafow
inicialy=r'(^\p{Lu}\. ?\p{Lu}\.|^\p{Lu}\.) ?$'
list_authors=[]      
with open (r"F:\Nowa_praca\fennica\statystyki\autorzy_Lista.txt", 'r', encoding='utf-8') as listaautorow:
    
    for autor in listaautorow:
        
        clear=re.sub(inicialy, '', autor.strip() )
        
        
        if clear:
            list_authors.append(clear)
        

#Zaciąganie viafow autorów do jsona 
list_authors

count=0
name_viaf={}
blad={}
search_querylist=[]
for name in tqdm(list_authors):
    
    count+=1 
    search_query = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name.strip(), number = 1)
    search_querylist.append(search_query)
    try:
        r = requests.get(search_query)
        r.encoding = 'utf-8'
        response = r.json()
    except Exception as error:
        blad[name]=error
        name_viaf[name] = []
        
        continue
        
    number_of_records = int(response['searchRetrieveResponse']['numberOfRecords'])
    if number_of_records > 10:
        for elem in range(number_of_records)[11:100:10]:
            search = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search=name.strip(), number = elem)
            try:
                r = requests.get(search)
                r.encoding = 'utf-8'        
                response['searchRetrieveResponse']['records'] = response['searchRetrieveResponse']['records'] + r.json()['searchRetrieveResponse']['records']
            except:
                continue
    if number_of_records == 0:
        name_viaf[name] = []
    else: 
        name_viaf[name] = response['searchRetrieveResponse']['records']
    if count%100==0 or count==len(list_authors):
        
        with open(r"F:\Nowa_praca\fennica\statystyki\VIAF_responses_2\baza_biogramy_viaf"+str(count)+".json", 'w', encoding='utf-8') as jfile:
            json.dump(name_viaf, jfile, ensure_ascii=False, indent=4)
        name_viaf={}

with open(r"F:\Nowa_praca\fennica\statystyki\blad.txt",'w', encoding='utf-8') as data:
    data.write(str(blad))




#%%
import re
s = r"\\$3CD-äänilevy$aesitetty musiikki$bprm$2rdacontent"

pattern = '(?<=\$3).*?(?=[$])'
pattern2='(?<=\$b)(.*?)(?=\$)'
pattern3='(?<=\$a).*?(?=[$])'

substring = re.findall(pattern, s)

print(substring)
#%% Pole 336 tłumaczenie (typy dokumnetów)
all_data=pd.read_excel(r"F:\Nowa_praca\fennica\statystyki\field336_split_arto.xlsx")
print (all_data)
lista=all_data.values.tolist()
dotlumaczenia=[]
przetlumaczone=[]
for x in lista:
    string=x[0]
    pattern = '(?<=\$3).*?(?=[$])'
    pattern2='(?<=\$b)(.*?)(?=\$)'
    pattern3='(?<=\$a).*?(?=[$])'

    substring = re.findall(pattern, string)
    substring2=re.findall(pattern2, string)
    substring3=re.findall(pattern3, string)
    if substring:
        #print(substring)
        substring=substring[0]
        dotlumaczenia.append(substring)
        #print(substring)
        result = translator.translate(substring, src='fi', dest='pl')
        #print(result.text)
        przetlumaczone.append(result.text)
    #if substring2:
     #   print (substring2)
      #  substring=substring2[0]
       # dotlumaczenia.append(substring)
        #result = translator.translate(substring, src='fi', dest='pl')
        #print(result.text)
        #przetlumaczone.append(result.text)
    if substring3:
        substring=substring3[0]
        dotlumaczenia.append(substring)
        #print(substring)
        result = translator.translate(substring, src='fi', dest='pl')
        #print(result.text)
        przetlumaczone.append(result.text)
df = pd.DataFrame(list(zip(dotlumaczenia, przetlumaczone)),
               columns =['Oryginal', 'Tlumaczenie'])
df.to_excel("tlumaczenie.xlsx", sheet_name='Sheet_name_1')        

#%% Zaciąganie viafów jedna odpowiedź nie więcej
all_data=pd.read_excel(r"F:\Nowa_praca\fennica\statystyki\pole600 viafowanie_fennica\wszyscy_autorzy_i_pola_z2ID_bez_pewnych_po_ID_i_po_pseudonazwa_po_nazwisko_data_pole_i_respond_ratio_x2600_Fennica.xlsx")

list_authors=all_data['nazwisko data']
unique=[]
duplikat=[]
for auth in tqdm(list_authors):
    if auth not in unique:
        unique.append(auth)
    else:
        duplikat.append(auth)
    
cleanedList = [x for x in unique if str(x) != 'nan']
count=0
name_viaf={}
blad={}
search_querylist=[]
for name in tqdm(cleanedList):
    count+=1
    if count%100==0 or count==len(cleanedList):
        
        with open(r"F:\Nowa_praca\fennica\statystyki\VIAF_responses_pole600Fennica\baza_biogramy_viaf"+str(count)+".json", 'w', encoding='utf-8') as jfile:
            json.dump(name_viaf, jfile, ensure_ascii=False, indent=4)
        name_viaf={}
    
    
    
    search_query = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name.strip(), number = 1)
    #search_querylist.append(search_query)
    try:
        r = requests.get(search_query)
        r.encoding = 'utf-8'
        response = r.json()
    except Exception as error:
        blad[name]=error
        name_viaf[name] = []
        
        continue
        
    number_of_records = int(response['searchRetrieveResponse']['numberOfRecords'])
    if number_of_records > 1 or number_of_records == 0:
        continue

    else: 
        name_viaf[name] = response['searchRetrieveResponse']['records']
        
with open(r"F:\Nowa_praca\fennica\statystyki\VIAF_responses_pole600Fennica\baza_biogramy_viaf_ostatni.json", 'w', encoding='utf-8') as jfile:
    json.dump(name_viaf, jfile, ensure_ascii=False, indent=4)

with open(r"F:\Nowa_praca\fennica\statystyki\blad.txt",'w', encoding='utf-8') as data:
    data.write(str(blad))
    
#%% Wyciąganie ISSN
val773=pd.read_excel(r"C:\Users\darek\PBL_field773_jedna_kolumnaLibri.xlsx")
listval773=val773['773'].tolist()
patternISSN=r'(?<=\$x)\d{4}-[a-zA-Z0-9_.-]{4}'
patterntitle=r'(?<=\$t).*?(?=\$|$)'   
issns=[]
tytul=[]
name=[]
for names in tqdm(listval773):
    for names2 in names.split('❦'):
        name.append(names2)
        title=re.findall(patterntitle, names2)   
        issn=re.findall(patternISSN, names2)
        try:
            tytuly=title[0].rstrip('. - /')
        except:
            tytuly='brak'
        try:
            issny=issn[0]
        except:
            issny='brak'
        issns.append(issny)
        tytul.append(tytuly)
df = pd.DataFrame(list(zip(tytul,issns, name)),
               columns =['tytul','issn', '773'])
df.to_excel("PBL_773_ISSNrozbiteLibri.xlsx", sheet_name='Sheet_name_1') 
new=df.loc[df['issn'] == 'brak']
new.to_excel("PBL_773_ISSNrozbiteLibri_tylko_brak.xlsx", sheet_name='Sheet_name_1')  
new2 = new.drop_duplicates(subset = ["tytul"])
new2.to_excel("PBL_773_ISSNrozbiteLibri_tylko_brak_bez_duplikatow.xlsx", sheet_name='Sheet_name_1')

#%% biblioteka narodowa rozbicie Pól Bn.mark VIAFY
plik=r"F:\Nowa_praca\libri\Iteracja 2021-07\oviafowane02.02.2022BN\libri_marc_bn_articles_2021-08-05!100_700_600z_VIAF_i_bez_viaf_good995_issns.mrk"
lista=mark_to_list(plik)
dictrec=list_of_dict_from_list_of_lists(lista)
val100=[]
probal=[]
for rekord in tqdm(dictrec):
    for key, val in rekord.items():
        if key=='700' or key=='100' or key=='600':

            v=val.split('❦')
            for vi in v:
                val100.append(vi)


df = pd.DataFrame(val100,columns =['100_700_600'])
df.to_excel("sprawdzanie_marc_articles_2021_2021-08-05.xlsx", sheet_name='Sheet_name_1') 
#wydobycie nazw osobowych:
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$d).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$)'
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
    
df = pd.DataFrame (list(zip(original_names,name_date_list,viafs_list)), columns =['100','nazwisko data','viaf' ]) 
df.to_excel("sprawdzanie_marc_articles_ALLgood995_2021-08-05.xlsx", sheet_name='Sheet_name_1')

df2=df.drop_duplicates()  
df2.to_excel("100_700_600bn_marc_articles_ALLgood995_2021-08-05BEZ_DUPLIKATOW2.xlsx", sheet_name='Sheet_name_1')   
###
lista_do_szukania_excel=pd.read_excel(r"C:\Users\darek\Caly_PBL_DO_ODPYTANIA.xlsx")
lista_do_szukania=lista_do_szukania_excel['name_date'].tolist()
unique=[]
duplikat=[]
for auth in tqdm(lista_do_szukania):
    if auth not in unique:
        unique.append(auth)
    else:
        duplikat.append(auth)

count=0
name_viaf={}
blad={}
search_querylist=[]
for name in tqdm(unique[0]):
    
    count+=1 
    search_query = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name.strip(), number = 1)
    search_querylist.append(search_query)
    try:
        r = requests.get(search_query)
        r.encoding = 'utf-8'
        response = r.json()
    except Exception as error:
        blad[name]=error
        name_viaf[name] = []
        
        continue
        
    number_of_records = int(response['searchRetrieveResponse']['numberOfRecords'])
    if number_of_records > 10:
        for elem in range(number_of_records)[11:100:10]:
            search = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search=name.strip(), number = elem)
            try:
                r = requests.get(search)
                r.encoding = 'utf-8'        
                response['searchRetrieveResponse']['records'] = response['searchRetrieveResponse']['records'] + r.json()['searchRetrieveResponse']['records']
            except:
                continue
    if number_of_records == 0:
        name_viaf[name] = []
    else: 
        name_viaf[name] = response['searchRetrieveResponse']['records']
    if count%100==0 or count==len(unique):
        
        with open(r"F:\Nowa_praca\libri\Iteracja 2021-07\noweViafy-100odpowiedzi_PBL\baza_biogramy_viaf"+str(count)+".json", 'w', encoding='utf-8') as jfile:
            json.dump(name_viaf, jfile, ensure_ascii=False, indent=4)
        name_viaf={}

with open(r"F:\Nowa_praca\libri\Iteracja 2021-07\noweViafy-100odpowiedzi_PBL\blad.txt",'w', encoding='utf-8') as data:
    data.write(str(blad))
    
# Zaciąganie odpowiedzi Viaf do słownika

dictionary={}
folderpath = r"F:\Nowa_praca\libri\Iteracja 2021-07\viafy_BN_articles_600"
os.listdir(folderpath)
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]

for path in tqdm(filepaths):
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
                    
                    
                    ratio=matcher(namestrip, matches[0] )
                    #print(ratio)
                    matchesdict[name].append([viaf, matches+[ratio],names, namestrip])
    
                    
        if not matchesdict[name] or len(matchesdict[name])>1:
            del matchesdict[name]

            
starttime=time()
df = pd.DataFrame(columns =['name', 'viaf', 'close_match', 'ratio','names','namestrip'])  
rows=[]
for key, value in matchesdict.items():

    
    
    for element in value:
        row={'name':key, 'viaf':element[0], 'close_match':element[1][0], 'ratio':element[1][1],'names':element[2],'namestrip':element[3]}
        rows.append(row)
df=df.append(rows, ignore_index=True)
endtime=time()
print(endtime-starttime)

                    



df.to_excel('viafs_respond_ratioPBL_caloscbezspacji.xlsx', index=False)         
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



#%% Statystyki 650

hasla={}
id_number={}
error=[]
pattern='(?<=7\$a).*?(?=\$|$)'
pattern2='(?<=fin\$0).*?(?=\$|$)'
for filename in tqdm(os.listdir(r"F:\Nowa_praca\fennica\nowe_Fennica_arto_19_01_2022")):
    if filename.endswith(".mrk"):
        
        path=os.path.join(r"F:\Nowa_praca\fennica\nowe_Fennica_arto_19_01_2022", filename)
        #print(path)
    
    
        #print(file)
        lista=mark_to_list(path)
        dictrec=list_of_dict_from_list_of_lists(lista)
    
      
    
        for record in tqdm(dictrec):
         
            if '655' in record.keys():
                value=record['655'].split("❦")
                
                for v in value:
                    #print(v)
                    try:
                        haslo = re.findall(pattern, v)[0]
                         
                        if haslo not in hasla:
                            hasla[haslo]=0
                        hasla[haslo]+=1
                         
                        id_yso=re.findall(pattern2, v)[0]
                        if id_yso not in id_number:
                            id_number[id_yso]=[haslo, 0]
                        id_number[id_yso][1]+=1
                    except IndexError:
                        error.append(record)

hasla=pd.DataFrame.from_dict(hasla, orient='index')
id_number=pd.DataFrame.from_dict(id_number, orient='index')
id_number.to_excel('655_id_number_nowe.xlsx')
#%% statystki pole 336
hasla={}
id_number={}
error=[]
pattern='(?<=\$a).*?(?=\$|$)'
pattern2='(?<=\/yso\/).*?(?=\$|$)'
"F:\Nowa_praca\fennica\nowe_Fennica_arto_19_01_2022"
    
      
    
        for record in tqdm(dictrec):
         
            if '336' in record.keys():
                value=record['336'].split("❦")
                
                for v in value:
                    #print(v)
                    try:
                        haslo = re.findall(pattern, v)[0]
                         
                        if haslo not in hasla:
                            hasla[haslo]=0
                        hasla[haslo]+=1
                         
                        #id_yso=re.findall(pattern2, v)[0]
                        #if id_yso not in id_number:
                        #    id_number[id_yso]=[haslo, 0]
                        #id_number[id_yso][1]+=1
                    except IndexError:
                        error.append(record)
hasla=pd.DataFrame.from_dict(hasla, orient='index')
#id_number=pd.DataFrame.from_dict(id_number, orient='index')
hasla.to_excel('336_stats_nowe.xlsx',engine='xlsxwriter')

ldr={}
for filename in tqdm(os.listdir(r"F:\Nowa_praca\fennica\nowe_Fennica_arto_19_01_2022")):
    if filename.endswith(".mrk"):
        
        path=os.path.join(r"F:\Nowa_praca\fennica\nowe_Fennica_arto_19_01_2022", filename)
        #print(path)
    
    
        #print(file)
        lista=mark_to_list(path)
        dictrec=list_of_dict_from_list_of_lists(lista)

        
        #list_6=[]
        for record in dictrec:
            if "LDR" in record.keys():
                ldr_6=record['LDR'][6]
                if ldr_6 not in ldr:
                    ldr[ldr_6]=0
                ldr[ldr_6]+=1
                #list_6.append(record['LDR'][6])
    
#unique=list(set(list_6))
                    
                                                
                                                
                                                
                
        
        