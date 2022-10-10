import requests
import json
import pandas as pd
adres='89.99'
URL= 'https://finto.fi/rest/v1/ykl/data?uri=http%3A%2F%2Furn.fi%2FURN%3ANBN%3Afi%3Aau%3Aykl%3A'+adres+'&format=application/ld%2Bjson'
response = requests.get(url = URL).json()
graph=response['graph']
counter=0
englishlist=[]
swedishlist=[]
finnishlist=[]
recs2table = []
for content in graph:
    counter=counter+1
    print(counter)
    #print(content)

    if 'prefLabel' in content:



        prefLabel=content['prefLabel']
        rec_dict={}
        
        for lang in prefLabel:
            print(lang)

            try:
                rec_dict[lang['lang']]=lang['value']
            except TypeError:
                pass
                

    if 'skos:notation' in content:
        #print (content['skos:notation'])
        rec_dict['YKL Class']=content['skos:notation']
        

        recs2table.append(rec_dict)
#print(recs2table)
df = pd.DataFrame.from_dict(recs2table)


df.to_excel(adres+'.xlsx', index=False)

#%%
import pandas as pd
import re
import os
files = [file for file in os.listdir('./YKL')]
allYKL=pd.DataFrame()
for file in files:
    df = pd.read_excel('./YKL/'+file)
    
    allYKL=pd.concat([allYKL, df])
allYKL.head
allYKL.to_excel('allYKL.xlsx', index=False)
dfall = pd.read_excel('./YKL/allYKL.xlsx')
dfall=dfall.sort_values(by='YKL Class', ascending=False)
dfall['YKL Class']
dfallonly8 = dfall[(dfall['YKL Class'] >= 80) & (dfall['YKL Class'] < 90) | (dfall['YKL Class'] == 8) ]
dfallonly8.to_excel('allonly8.xlsx', index=False)
