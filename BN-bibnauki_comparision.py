# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from tqdm import tqdm
import ijson
import json
import jellyfish
with open ('C:/Users/dariu/Downloads/results_final.json', 'r', encoding='utf-8') as f:
    nikodem=json.load(f)['records']
files=["C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000001.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000002.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000003.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000004.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000005.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000006.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000007.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000008.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000009.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000010.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000011.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000012.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000013.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000014.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000015.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000016.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000017.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000018.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000019.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000020.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000021.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000022.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000023.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000024.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000025.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000026.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000027.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000028.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000029.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000030.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000031.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000032.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000033.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000034.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000035.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000036.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000037.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000038.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000039.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000040.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000041.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000042.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000043.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000044.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000045.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000046.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000047.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000048.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000049.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000050.json_results",
"C:/Users/User/Desktop/BN_ARTICLES_JSON_HASHED-20221011T104350Z-001/BN_ARTICLES_JSON_HASHED/msplit00000051.json_results"]
output={}
for file in files:
    with open (file, 'r', encoding='utf-8') as f2:
        patryk=json.load(f2)['bn_records']
    darek={'brak':[]} 
       
    for elem in patryk:
        switch=False
        if elem.get('773'):
            
            for e in elem.get('773'):
                if e.get('x'):
                    switch=True
                    issn=e.get('x').strip(' \'.,:;')
                    if issn not in darek:
                        darek[issn]=[(elem.get('001'), elem.get('245_hashed'),elem.get('650a')) ]
                    else:
                        darek[issn].append((elem.get('001'), elem.get('245_hashed'),elem.get('650a')))
        if not switch:
            
            darek['brak'].append((elem.get('001'), elem.get('245_hashed'),elem.get('650a')))
                    
       
            
    lista=[]   
    for item in tqdm(nikodem):
        #print(item)
        
        #titlebibNauk=item['hashed']
        if 'hashed' not in item:
            lista.append(item)
            
            
        
        identifier=item['identifier']
        
        issnbibNauk=[item.get('issn')[0] if item.get('issn') else None][0]
        if issnbibNauk in darek:

            
            for dp in darek[issnbibNauk]:
                titleBibNar=dp[1]
                idBibNar=dp[0]
                haslo=dp[2]
                if titlebibNauk==titleBibNar:
                    if identifier not in output:
                    
                        output[identifier]=[idBibNar]
                    else:
                        output[identifier].append(idBibNar)
                        
                    

        for dp in darek['brak']:
            titleBibNar=dp[1]
            idBibNar=dp[0]
            haslo=dp[2]
            if titlebibNauk==titleBibNar:
                
                    if identifier not in output:
                    
                        output[identifier]=[idBibNar]
                    else:
                        output[identifier].append(idBibNar)
        
with open ('json_all.json', 'w', encoding='utf-8') as file:
    json.dump(output,file,ensure_ascii=False)           
#jellyfish.levenshtein_distance(titlebibNauk, titleBibNar) <=1
            
#%% Opracowanie resultatow
from tqdm import tqdm
import ijson
import json
import jellyfish
with open ('C:/Users/dariu/Desktop/BNvsBibNauki/json_dopasowane_from150tys.json', 'r', encoding='utf-8') as f:
    dopasowane=json.load(f)
listaBN=["C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000051.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000000marc8.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000001.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000002.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000003.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000004.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000005.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000006.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000007.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000008.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000009.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000010.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000011.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000012.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000013.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000014.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000015.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000016.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000017.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000018.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000019.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000020.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000021.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000022.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000023.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000024.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000025.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000026.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000027.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000028.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000029.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000030.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000031.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000032.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000033.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000034.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000035.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000036.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000037.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000038.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000039.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000040.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000041.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000042.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000043.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000044.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000045.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000046.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000047.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000048.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000049.json",
"C:/Users/dariu/Desktop/BNvsBibNauki/BN_article_json_wszystko/msplit00000050.json"]            


dictionary={}

for adres in listaBN:
    with open (adres, 'r', encoding='utf-8') as f:
        Bnarticle=json.load(f,strict=False)
    
              
    for article in tqdm(Bnarticle):
        fields=article['fields']
        for field in fields:
            for key, value in field.items():
                if key=='001':
                    ident=value
                #     if ident==value:
                #         switch=True
                #if switch:
                if key=='650':
                    
                    #print(value)
                    for k, v in value.items():
                        if k=='subfields':
                            for words in v:
                                
                                for klucz,wartosc in words.items():
                                    if klucz=='a':
                                        #print(wartosc)
                                        if ident in dictionary:
                                            dictionary[ident].append(wartosc)
                                        else:
                                            dictionary[ident]=[wartosc]
with open ('BibNar_id_hasla.json', 'w', encoding='utf-8') as file:
    json.dump(dictionary,file,ensure_ascii=False) 
output={}                                        
for idNauki, idBn in tqdm(dopasowane.items()):
    if len(idBn)<4:
        for ident in idBn: 
            if ident in dictionary:
                if idNauki not in output:
                    output[idNauki]=[(ident,dictionary[ident])]
                else:
                    output[idNauki].append((ident,dictionary[ident]))
with open ('BibNauk_hasla.json', 'w', encoding='utf-8') as file:
    json.dump(output,file,ensure_ascii=False) 
                
                
            
        
 

        
    
