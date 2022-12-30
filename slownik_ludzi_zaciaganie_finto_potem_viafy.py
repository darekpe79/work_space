import requests
import json
import pandas as pd
from tqdm import tqdm
import regex as re
liczby=[]
for x in range (1,214970):
    x=str(x)
    y=x.zfill(9)
    
    liczby.append(y)
nazwy_i_pseudonimy=[]
identyfikatory=[]
warianty_nazwy=[]
for x in tqdm(liczby):  #214970
    URL=r'https://finto.fi/rest/v1/finaf/data?uri=http%3A%2F%2Furn.fi%2FURN%3ANBN%3Afi%3Aau%3Afinaf%3A'+x+'&format=application/ld%2Bjson'

    try:
        response = requests.get(url = URL).json()

    
    #print(json.dumps(response['graph'], indent=2))
        graph=response['graph']
    except:
        continue
    if graph[0]['uri']=='http://rdaregistry.info/Elements/a/P50094':
    
        
    
        identyfikator=[]
        nazwa_i_pseudo=[]
        zmienna=False
        zmenna1=False
        zmienna2=False
        wariant_nazwy=[]
        for content in graph:
            
    
    
            
            
           
            if 'http://rdaregistry.info/Elements/a/P50094' in content:
                
                #print(content['http://rdaregistry.info/Elements/a/P50094'])
                ident=content['http://rdaregistry.info/Elements/a/P50094']
                identyfikator.append(ident)
                zmienna1=True    
            if 'prefLabel' in content:
                
                
                nazwa=content['prefLabel']
                
                nazwa_i_pseudo.append(nazwa['value'])
                zmienna=True
            if 'altLabel' in content:
                wariant=content['altLabel']
                
                if type(wariant)==list:
                    for warianty in wariant:
                        warianty_nazw=warianty['value']
                
                        wariant_nazwy.append(warianty_nazw)
                else:
                    wariant_nazwy.append(wariant['value'])
                    
                zmienna2=True
    
        if len(wariant_nazwy)>1:
            
            wariant_nazwy_str='❦'.join(wariant_nazwy) 
            wariant_nazwy=[]
            wariant_nazwy.append(wariant_nazwy_str)                    
        
        if len(nazwa_i_pseudo)>1:
            
            nazwa_i_pseudo_str='❦'.join(nazwa_i_pseudo) 
            nazwa_i_pseudo=[]
            nazwa_i_pseudo.append(nazwa_i_pseudo_str)
        if zmienna==True:
            zmienna=False
        else:
            nazwa_i_pseudo.append('brak')
        if zmienna1==True:
            zmienna1=False
        else:
            identyfikator.append('brak')
            
        if zmienna2==True:
            zmienna2=False
        else:
            wariant_nazwy.append('brak')
        nazwy_i_pseudonimy.append(nazwa_i_pseudo)
        identyfikatory.append(identyfikator)
        warianty_nazwy.append(wariant_nazwy)
df = pd.DataFrame (list(zip(nazwy_i_pseudonimy, identyfikatory,warianty_nazwy)), columns =['nazwa_i_pseudonim', 'identyfikator', 'wariant_nazwy']) 
df.to_excel("kartoteka_wzorcowa.xlsx", sheet_name='Sheet_name_1')     

#%%
Asteri_Id_pattern='(?<=ID: ).*?(?=\')'
ISNI_Pattern='(?<=isni\/).*?(?=\')'

all_data=pd.read_excel(r"F:\Nowa_praca\fennica\statystyki\kartoteka_wzorcowa — kopia.xlsx")
lista=all_data['identyfikator'].tolist()
isnilist=[]
for ident in lista:
    isni = re.findall(ISNI_Pattern, ident)
    if isni:
        print(isni[0])
        isnilist.append(isni[0])
isnilist.index('0000000121478925')    
count=0
name_viaf={}
blad={}
search_querylist=[]
for name in tqdm(isnilist):
    
    count+=1 
    search_query = "http://www.viaf.org//viaf/search?query=local.personalNames+=+%27{search}%27&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name.strip(), number = 1)
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
            search = "http://www.viaf.org//viaf/search?query=local.personalNames+=+'{search}'&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search=name.strip(), number = elem)
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
    if count%100==0 or count==len(isnilist):
        
        with open(r"F:\Nowa_praca\fennica\statystyki\VIAF_responses_2\baza_biogramy_viaf"+str(count)+".json", 'w', encoding='utf-8') as jfile:
            json.dump(name_viaf, jfile, ensure_ascii=False, indent=4)
        name_viaf={}

with open(r"F:\Nowa_praca\fennica\statystyki\blad.txt",'w', encoding='utf-8') as data:
    data.write(str(blad))


