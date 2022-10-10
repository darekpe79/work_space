# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:31:17 2021

@author: darek
"""
import pandas as pd
from random import randint
from time import sleep
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm


periodicals = pd.read_excel (r"F:\Nowa_praca\libri\Iteracja 2021-07\PBL_773_ISSNrozbiteLibri_tylko_brak_bez_duplikatow.xlsx") 
#Key-title    Menneske og miljø i Nord-Troms 
dosprawdzenia=periodicals['tytul'].tolist()
dictionary={}
glowny=[]
wyciagniety=[]
issnlist=[]
for onetitle in tqdm(dosprawdzenia):
    URL='https://portal.issn.org/api/search?search[]=MUST=default='+onetitle
    page=get(URL)
    bs=BeautifulSoup(page.content)
    block=bs.find_all('div', {'class':'item-result-block'})
    #print(block)
    
    lista=[]
    dictionary[onetitle]={}
    for element in block:
        #print(element)
        
        
        
        tytul=element.find('h5', {'class':'item-result-title'})
        wlasciwy=tytul.text
        #print(wlasciwy)
        #wlasciwy=wlasciwy.split('    ')
        #print(wlasciwy)
        wlasciwy=wlasciwy.replace('\nKey-title \xa0', '')#strip('\nKey-title \xa0')
        

        
        
        issns=element.find('div', {'class':'item-result-content-text flex-zero'})
        
        #issns2=issns.find('p')
        #issns=issns2.text
        
            
        
    
    
        
        if issns==None:
            issns=bs.find('div', {'sidebar-accordion-list-selected-item'}).text
            issns=issns.strip('ISN :')
            #print(issns)
            
            
        elif issns is not None:
            issns2=issns.find('p')
            issns=issns2.text
            issns=issns.strip('ISN :')
        else:
            issns='brak'
        
        
        
        dictionary[onetitle][wlasciwy]=issns
        glowny.append(onetitle)
        wyciagniety.append(wlasciwy.strip())
        issnlist.append(issns)
        sleep(randint(1,2))
df3 = pd.DataFrame(list(zip(glowny,wyciagniety, issnlist)),
               columns =['tytul_szukany','tytul_wyciagniety','issn'])

df3.to_excel("PBLsciagnieteISSNYladne2.xlsx", sheet_name='Sheet_name_1')
#%% Wgrać plik, zrobić listy, wyczycić
periodicalsissn = pd.read_excel (r"F:\Nowa_praca\libri\Iteracja 2021-07\PBLsciagnieteISSNYladne2.xlsx")
glowny=periodicalsissn['tytul_szukany'].tolist()
#mask = periodicalsissn['tytul_wyciagniety'].apply(lambda x: any(item.rstrip(' :') for item in glowny if item.rstrip(' :') == str(x)))

#df3 = df[mask]
#print(df3)
similarity1=periodicalsissn[periodicalsissn['tytul_szukany'].apply(lambda x: x.rstrip(' :'))==periodicalsissn['tytul_wyciagniety'].apply(lambda x: x.rstrip(' :'))]
diff=periodicalsissn[periodicalsissn['tytul_szukany'].apply(lambda x: x.rstrip(' :'))!=periodicalsissn['tytul_wyciagniety'].apply(lambda x: x.rstrip(' :'))]
diff1=periodicalsissn[~periodicalsissn.isin(similarity1)].dropna()
#df3['tytul_szukany'] = df3['tytul_szukany'].str.rstrip(' :')
#df4=df[~mask]
similarity1.to_excel("PBLsciagnieteISSNyPewnelibri.xlsx", sheet_name='Sheet_name_1')
diff.to_excel("PBLsciagnieteISSNyNiePewnelibri.xlsx", sheet_name='Sheet_name_1')
#dictionary=pd.DataFrame.from_dict(dictionary, orient='index')  
#dictionary.to_excel("issnyskrobane.xlsx", sheet_name='Sheet_name_1')   
    #append dict
#%% issnowanie w excelu
bezissn = pd.read_excel (r"F:\Nowa_praca\libri\Iteracja 2021-07\PBL_773_ISSNrozbiteLibri_tylko_brak.xlsx")
tablewith_ISSN=bezissn.merge(similarity1, left_on='tytul', right_on='tytul_szukany', how='inner')
tablewith_ISSN.to_excel("PBLISSNzpolem773Libri_wszystko.xlsx", sheet_name='Sheet_name_1')
#%%Issnowanie w excelu z dwóch osobnych plików
bezissn = pd.read_excel (r"F:\Nowa_praca\libri\Iteracja 2021-07\PBL_773_ISSNrozbiteLibri_tylko_brak@@@.xlsx")
similarity1=pd.read_excel(r"F:\Nowa_praca\libri\Iteracja 2021-07\GOTOWY_PEWNY_PBL_ISSNzpolem773Libri_bezduplikatów.xlsx")
tablewith_ISSN=bezissn.merge(similarity1, left_on='tytul', right_on='tytul_szukany', how='inner')
tablewith_ISSN.to_excel("PEWNE_ISSNzpolem773Libri_wszystko_PBL.xlsx", sheet_name='Sheet_name_1')
#mask = bezissn['tytul'].apply(lambda x: any(item for item in tytulszukany if item == str(x)))
#df3 = bezissn[mask]
#df4=bezissn[~mask]
#df3['issny']=[dic2[name] for name in df3['tytul']]

#df4.to_excel("bezISSNzpolem773.xlsx", sheet_name='Sheet_name_1')

    




    

