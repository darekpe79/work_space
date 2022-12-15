import requests
import json
from definicje import *
import pandas as pd
import os
from tqdm import tqdm
#from googletrans import Translator
from itertools import zip_longest
import regex as re
import _pickle as pickle
all_data=pd.read_excel(r"F:\Nowa_praca\fennica\statystyki\wszyscy autorzy i daty.xlsx")
listaautor=all_data['nazwisko data'].tolist()
listaautor.index('Tickell, Jerrard ')
count=0
name_viaf={}
blad={}
search_querylist=[]
for name in tqdm(listaautor):
    count+=1 
    search_query = "http://www.viaf.org//viaf/search?query=local.corporateNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name.strip(), number = 1)
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
        for elem in range(number_of_records)[11::10]:
            search = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search=name.strip(), number = elem)
            r = requests.get(search_query)
            r.encoding = 'utf-8'
            response['searchRetrieveResponse']['records'] = response['searchRetrieveResponse']['records'] + r.json()['searchRetrieveResponse']['records']
           
    if number_of_records == 0:
        name_viaf[name] = []
    else: 
        name_viaf[name] = response['searchRetrieveResponse']['records']
    if count%100==0 or count==len(listaautor):
        
        with open(r"F:\Nowa_praca\fennica\statystyki\Nowy folder\baza_biogramy_viaf"+str(count)+".json", 'w', encoding='utf-8') as jfile:
            json.dump(name_viaf, jfile, ensure_ascii=False, indent=4)
        name_viaf={}
with open(r"F:\Nowa_praca\fennica\statystyki\Nowy folder\blad2.txt",'w', encoding='utf-8') as data:
    data.write(str(blad))


#%%

def keywithmaxval(d):
    max_keys = [key for key, value in d.items() if value == max(d.values())]

    return max_keys
name_viaf={}
#name='Drukarnia i Księgarnia Świętego Wojciecha. Wydawnictwo Święty Wojciech'
all_data=pd.read_excel(r"D:/Nowa_praca/publishers_work/fin_publishers.xlsx")
publisher=list(dict.fromkeys(all_data['publisher'].tolist()))
output={}
bad_output={}
for name in tqdm(publisher):
    if name=='brak':
        continue
    
    
    else:
        name1=name.replace("\"", "").replace("'", "").strip(', . :[]').casefold()
        search_query = "http://www.viaf.org//viaf/search?query=local.corporateNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name1, number = 1)
        
        try:
            r = requests.get(search_query)
            r.encoding = 'utf-8'
            response = r.json()
            number_of_records = int(response['searchRetrieveResponse']['numberOfRecords'])
            #if number_of_records==1:
            
            response1=response['searchRetrieveResponse']['records']
            name_to_check_bad={}
            checker={}
            proba={}
            for elem in response1:
                viaf = elem['record']['recordData']['viafID']
                headings = elem['record']['recordData']['mainHeadings']['data']
                if isinstance(headings, list):
                    for head in headings:
                        checkname=head['text'].replace("\"", "").replace("'", "").strip(', . :').casefold()
                        s = SequenceMatcher(None, name1, checkname).ratio()
                        print(s)
                        
                        if s>0.90:
                            
                            checker[viaf+' '+checkname]=s
                            name_to_check_bad[name]=s
                            proba[viaf+' '+checkname]=[name1,viaf, checkname,number_of_records,s,name]
                            #output2[viaf+' '+ checkname]=[viaf,name1, checkname,number_of_records,s,name]

                            
                else:
                    checkname=headings['text'].replace("\"", "").replace("'", "").strip(', . :').casefold()
                    s = SequenceMatcher(None, name1, checkname).ratio()
                    print(s)
                    if s>0.90:
                        name_to_check_bad[name]=s
                        checker[viaf+' '+checkname]=s
                        
                        proba[viaf+' '+checkname]=[name1,viaf, checkname,number_of_records,s,name]
                        #output2[viaf+' '+checkname]=[viaf, name1, checkname,number_of_records,s,name]
                        
            bestKeys=keywithmaxval(checker)
            for best in bestKeys:
                output[best]=proba[best]
            if name not in name_to_check_bad:
                bad_output[name]=number_of_records
                
                    
                
                
        except Exception as error:
            
            name_viaf[name]=error
            
            
with open('institution_dict.json', 'w', encoding='utf-8') as jfile:
    json.dump(output, jfile, ensure_ascii=False, indent=4)
excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("Bn_publisher.xlsx", sheet_name='publisher') 
lis=[]
nowy={}
prob={}   
#prob['lala']=1

prob['lll']=1
def keywithmaxval(d):
    max_keys = [key for key, value in d.items() if value == max(d.values())]


     
    return max_keys
l=keywithmaxval(prob)
nowy[l]=prob[l] 

lis.append(prob)
excel=pd.DataFrame.from_dict(prob, orient='index') 
t='Gießen'.casefold()

#%%

viafid=[]
for autor in listaautor[:10]:
    url = "http://www.viaf.org/viaf/AutoSuggest?query="+autor
    data = requests.get(url).json()
    try:
        viafid.append(data['result'][0]['viafid'])
    except:
        viafid.append('brak')
        
#print (viafid)
autorViafid=[]
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

name='Shaw, George Bernard'
search_query = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name.strip(), number = 1)
dane = requests.get(search_query).json()['searchRetrieveResponse']
records=dane['records']
slownik={}
for record in records:
    viaf=record['record']['recordData']['viafID']
    data=record['record']['recordData']['mainHeadings']['data']
    
    if type(data) is list:
        for d in data:
            
            name=d['text']
            slownik[name]=[]
            
            
            source=d['sources']
            if type(source) is list:
                slownik[name].append(source[0]['s'])
            else:
                slownik[name].append(source['s'])
            slownik[name].append(viaf)
            
    else:
        name=data['text']
        source=data['sources']
        slownik[name]=[]
        if type(source) is list:
            slownik[name].append(source[0]['s'])
        else:
            slownik[name].append(source['s'])
        
        slownik[name].append(viaf)
        
        
# DO DOKONCZENIA PO URLOPIE:::::::::::::::        
    
pattern_daty=r'(([\d?]+-[\d?]+)|(\d+-)|(\d+\?-)|(\d+))(?=$|\.|\)| )'

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