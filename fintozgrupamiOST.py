import requests
import json
import pandas as pd
from tqdm import tqdm
URL= 'https://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2Fp7289&format=application/ld%2Bjson'
response = requests.get(url = URL).json()
#response = requests.get(url).json()
#print(json.dumps(response, indent=4))
#print(json.dumps(response['graph'], indent=4))
#print(response['results'][0]['bibjson']['license'][0]['type'])
graph=response['graph']
counter=0
englishlist=[]
swedishlist=[]
finnishlist=[]
recs2table = []
for content in graph:
    counter=counter+1
    url=content['uri'].split('/')
    #print(url)
    urlpart=url[-1]
    if urlpart.startswith('p'):
        print(urlpart)
    # wyciągam glowne grupy dla hasel po uri z api
        URL1= 'https://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2F'+urlpart+'&format=application/ld%2Bjson'
        response1 = requests.get(url = URL1).json()
        r=response1['graph']
        print(r)
        slowoeng=[]
        slowofin=[]
        slowosv=[]
        for e in r:
            
            dictslowo={}
            #print(e['type'])
            if 'prefLabel' in e:
                if 'skos:Collection' in e['type']:
                    
                    pref=e['prefLabel']
                    for lang in pref:
                        #print(lang)
                    
                        for k,v in lang.items():
                            if v=='en':
                                
                                slowo=lang['value']
                                
                                slowoeng.append(slowo)
                            if v=='fi':
                                slowof=lang['value']
                                
                                slowofin.append(slowof)
                            if v=='sv':
                                slowos=lang['value']
                                
                                slowosv.append(slowos)
        #print(slowoeng)
    #tworze slownik dla danych hasel i grup w jakich sie znajduja    
        if 'prefLabel' in content:
    
    
    
            prefLabel=content['prefLabel']
            rec_dict={}
            for lang in tqdm(prefLabel):
    
    
                rec_dict[lang['lang']]=lang['value']
                
            #print(slowoeng)
            rec_dict['Eng_Group']= '❦'.join([str(elem) for elem in slowoeng])
            rec_dict['Fi_Group']= '❦'.join([str(elem) for elem in slowofin])
            rec_dict['Sv_Group']= '❦'.join([str(elem) for elem in slowosv])
    
            recs2table.append(rec_dict)
print(recs2table)
df = pd.DataFrame.from_dict(recs2table)


df.to_excel('religious literature.xlsx', index=False)












#%%



URL= 'https://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2Fp10167&format=application/ld%2Bjson'
response = requests.get(url = URL).json()
#print((json.dumps(response, indent=4)))
r=response['graph']
#print((json.dumps(r, indent=4)))

slowoeng=[]
for e in r:
    
    dictslowo={}
    #print(e['type'])
    if 'prefLabel' in e:
        if 'skos:Collection' in e['type']:
            
            pref=e['prefLabel']
            for lang in pref:
                #print(lang)
            
                for k,v in lang.items():
                    if v=='en':
                        
                        slowo=lang['value']
                        print(slowo)
                        slowoeng.append(slowo)
print(slowoeng)
                
               
                
