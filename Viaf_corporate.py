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
from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()


#%% MOJE KORPO Z ALGORYTMEM

def keywithmaxval(d):
    max_keys = [key for key, value in d.items() if value == max(d.values())]

    return max_keys

#name='Państwowe Wydawnictwo Naukowe,'
all_data=pd.read_excel(r'D:/Nowa_praca/publishers_work/fin_publishers_with_fin11_isni_viaf_dziwne.xlsx',sheet_name='do_spr')
publisher=list(dict.fromkeys(all_data['nazwa Marc 710'].tolist()))
output={}
bad_output={}
name_viaf={}
for name in tqdm(publisher):
    if name=='brak':
        continue
    
    
    else:
        name1=name.replace("\"", "").replace("'", "").strip(', . :[]').casefold()
        search_query = "http://www.viaf.org//viaf/search?query=local.corporateNames%20all%20%22{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name1, number = 1)
        
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
            

with open('pozostale_710_fin_bezFIN11_bad.json', 'w', encoding='utf-8') as jfile:
    json.dump(bad_output, jfile, ensure_ascii=False, indent=4)
excel=pd.DataFrame.from_dict(bad_output, orient='index') 
excel.to_excel("pozostale_710_fin_bezFIN11_bad.xlsx", sheet_name='publisher') 


#### Sprawdzam wyniki czy zgadza się 710-260-264
from difflib import SequenceMatcher      
all_data=pd.read_excel(r'D:/Nowa_praca/publishers_work/do_ujednolicania_viafowania/wszystko_bez_710_do_Viafowania_potem_ujednolicania_bez_duplikatów.xlsx')
rslt_df = all_data[(all_data['fin11'] != 'brak')]
publisher=list(dict.fromkeys(all_data['Oryginal_710_po_VIAF'].tolist()))
all_data['matcher'] = all_data.apply(lambda x : SequenceMatcher(None, x['Oryginal_260_264'], x['Oryginal_710_po_VIAF_i_z_ujednolicone_bez_710']).ratio() if not any ([isinstance(x['Oryginal_260_264'], float),isinstance(x['Oryginal_710_po_VIAF_i_z_ujednolicone_bez_710'], float)]) else 'brak',axis=1)
all_data.to_excel("wszystko_bez_710_matcher710-26x.xlsx", sheet_name='publisher') 

##dociągam brakujące ujednolicone:
    
rslt_df = all_data[(all_data['z ujednolicone_bez_710(plik)'] == 'brak')]
publisher=list(dict.fromkeys(rslt_df['VIAF'].tolist()))
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

def keywithmaxval(d):
    max_keys = [key for key, value in d.items() if value == max(d.values())]

    return max_keys
output={}

for name in tqdm(publisher):
    if name=='brak':
        continue
    
    
    else:
        
        
       # name = '139876367'

        search_query = "http://www.viaf.org//viaf/search?query=local.corporateNames%20all%20%22{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name, number = 1)
        
        try:
            r = requests.get(search_query)
            r.encoding = 'utf-8'
            response = r.json()
            number_of_records = int(response['searchRetrieveResponse']['numberOfRecords'])
            if number_of_records>=1:
            
                response1=response['searchRetrieveResponse']['records']
                name_to_check_bad={}
                checker={}
                proba={}
                for elem in response1:
                    viaf = elem['record']['recordData']['viafID']
                    if viaf==name:
     
                        headings = elem['record']['recordData']['mainHeadings']['data']
                        
                        if isinstance(headings, list):
                            for head in headings:
                                sources=head['sources']['s']
                                text=head['text']
                                if ad.is_arabic(text) or ad.is_cyrillic(text) or ad.is_hebrew(text) or ad.is_greek(text) or lang_detect(text) :
                                    continue
                             
                                
                               
                                if isinstance(sources, list):
                                  
                                    
                                    checker[text]=len(sources)
                                    proba[text]=[name,sources]
                                    
                                    print(sources)
                                else:
                                    checker[text]=1
                                    proba[text]=[name,sources]
                        else:
                            sources=headings['sources']['s']
                            text=headings['text']
                            checker[text]=1
                            proba[text]=[name,sources]
                        
                                 
                            
                bestKeys=keywithmaxval(checker)
                for best in bestKeys:
                    output[best]=proba[best]

                    
                        
                    
                
        except Exception as error:
            
            name_viaf[name]=error


excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("ujednolicone_bez_710.xlsx", sheet_name='publisher') 
with open('ujednolicone_bez_710.json', 'w', encoding='utf-8') as jfile:
    json.dump(output, jfile, ensure_ascii=False, indent=4)        
     

#%%FIN11
import requests
import json
import pandas as pd
from tqdm import tqdm
import regex as re
all_data=pd.read_excel(r'D:/Nowa_praca/publishers_work/fin_publishers_with_fin11_isni_viaf.xlsx')
rslt_df = all_data[(all_data['fin11'] != 'brak')]
publisher=list(dict.fromkeys(all_data['fin11'].tolist()))
#publisher=['000005851','000006117']


output={}
for x in tqdm(publisher):
    #x='000002466'
    #214970
    URL=r'https://finto.fi/rest/v1/finaf/data?uri=http%3A%2F%2Furn.fi%2FURN%3ANBN%3Afi%3Aau%3Afinaf%3A'+x+'&format=application/ld%2Bjson'

    try:
        response = requests.get(url = URL).json()

    
    #print(json.dumps(response['graph'], indent=2))
        graph=response['graph']
    except:
        continue
    if graph[0]['uri']=='http://rdaregistry.info/Elements/a/P50006':
        output[x]=[]
    
        
    
        # identyfikator=[]
        # nazwa_i_pseudo=[]
        # zmienna=False
        # zmenna1=False
        # zmienna2=False
        # wariant_nazwy=[]
        for content in graph:
            
    
    
            
            
           
            if 'http://rdaregistry.info/Elements/a/P50006' in content:
                
                #print(content['http://rdaregistry.info/Elements/a/P50094'])
                ident=content['http://rdaregistry.info/Elements/a/P50006']
                if isinstance(ident, list):
                    for i in ident:
                        if 'uri' in i:
                            output[x]=[i['uri']]
                elif 'uri' in ident:
                        output[x]=[i['uri']]
                # zmienna1=True    
            else:
                output[x]=['brak_isni']
                
            if 'prefLabel' in content:
                
                
                nazwa=content['prefLabel']
                output[x].append(nazwa['value'])
                
excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("fin_isni.xlsx", sheet_name='publisher') 


###isni to viaf


rslt_df = excel[(excel[0] != 'brak_isni')]
publisher=list(dict.fromkeys(rslt_df[0].tolist()))
viafs={}
ISNI_Pattern='(?<=isni\/).*?(?=\'|$)'
for name in tqdm(publisher):
    if name=='brak':
        continue
    
    
    else:
        name = re.findall(ISNI_Pattern, name)[0]
        
        
        #name = '147390924'

        search_query = "http://www.viaf.org//viaf/search?query=local.corporateNames%20all%20%22{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name, number = 1)
        
        #try:
        r = requests.get(search_query)
        r.encoding = 'utf-8'
        response = r.json()
        number_of_records = int(response['searchRetrieveResponse']['numberOfRecords'])
        if number_of_records>=1:
        
            response1=response['searchRetrieveResponse']['records']
            name_to_check_bad={}
            checker={}
            proba={}
            for elem in response1:
                viaf = elem['record']['recordData']['viafID']
                viafs[name]=viaf
excel=pd.DataFrame.from_dict(viafs, orient='index') 
excel.to_excel("isni-viaf.xlsx", sheet_name='publisher')                  
    
#%%fin11 ---- Viaf _____UJEDNOLICANIE NAZWA
ISNI_Pattern='(?<=isni\/).*?(?=\'|$)'
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

def keywithmaxval(d):
    max_keys = [key for key, value in d.items() if value == max(d.values())]

    return max_keys
name_viaf={}
#name='Państwowe Wydawnictwo Naukowe,'
all_data=pd.read_excel(r'D:/Nowa_praca/publishers_work/do_ujednolicania/wszystko_z_710_do_ujednolicania_bez_duplikatów.xlsx')
rslt_df = all_data[(all_data['nazwa_z_710'] != 'brak')]
publisher=list(dict.fromkeys(rslt_df['VIAF'].tolist()))
output={}
bad_output={}
for name in tqdm(publisher):
    if name=='brak':
        continue
    
    
    else:
        
        
        #name = '147390924'

        search_query = "http://www.viaf.org//viaf/search?query=local.corporateNames%20all%20%22{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search = name, number = 1)
        
        try:
            r = requests.get(search_query)
            r.encoding = 'utf-8'
            response = r.json()
            number_of_records = int(response['searchRetrieveResponse']['numberOfRecords'])
            if number_of_records>=1:
            
                response1=response['searchRetrieveResponse']['records']
                name_to_check_bad={}
                checker={}
                proba={}
                for elem in response1:
                    viaf = elem['record']['recordData']['viafID']
     
                    headings = elem['record']['recordData']['mainHeadings']['data']
                    
                    if isinstance(headings, list):
                        for head in headings:
                            sources=head['sources']['s']
                            text=head['text']
                            if ad.is_arabic(text) or ad.is_cyrillic(text) or ad.is_hebrew(text) or ad.is_greek(text) or lang_detect(text) :
                                continue
                         
                            
                           
                            if isinstance(sources, list):
                              
                                
                                checker[text]=len(sources)
                                proba[text]=[name,sources]
                                
                                print(sources)
                            else:
                                checker[text]=1
                                proba[text]=[name,sources]
                    else:
                        sources=headings['sources']['s']
                        text=headings['text']
                        checker[text]=1
                        proba[text]=[name,sources]
                        
                                 
                            
                bestKeys=keywithmaxval(checker)
                for best in bestKeys:
                    output[best]=proba[best]

                    
                        
                    
                
        except Exception as error:
            
            name_viaf[name]=error


excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("ujednolicone_wszystkie.xlsx", sheet_name='publisher') 
with open('ujednolicone_wszystkie.json', 'w', encoding='utf-8') as jfile:
    json.dump(output, jfile, ensure_ascii=False, indent=4)        


with open('D:/Nowa_praca/publishers_work/do_ujednolicania/ujednolicone_wszystkie.json', encoding='utf-8') as fh:
    data = json.load(fh)

print(data)

        



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