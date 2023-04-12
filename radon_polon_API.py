# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:42:31 2023

@author: dariu
"""

import requests
import json
import time
import pandas as pd
from definicje import *
import requests
#DANE ZINTEGROWANE TUTAJ Mogę wydobyć orcid

    
    
    
#PRACOWNICY POL-ON
url='https://radon.nauka.gov.pl/opendata/polon/employees?resultNumbers=10&disciplineName=literaturoznawstwo&penaltyMarker=false'
  
response=requests.get(url)
response2= response.json()['results']
token= response.json()['pagination']['token']



def get_data(url):
    
    response=requests.get(url)
    
    token= response.json()['pagination'].get('token')
    
        
    response2= response.json()['results']
    personal_data=[]
    for resp in response2:
        personal_data.append(resp['personalData'])
    return personal_data, token
start_time=time.time()
all_data,token=get_data('https://radon.nauka.gov.pl/opendata/polon/employees?resultNumbers=100&disciplineName=literaturoznawstwo&penaltyMarker=false')


while token:
    
    
    
    data,token=get_data(f'https://radon.nauka.gov.pl/opendata/polon/employees?resultNumbers=100&disciplineName=literaturoznawstwo&penaltyMarker=false&token={token}')
    all_data.extend(data)
    
end_time=time.time()
print(end_time-start_time)

    
    
    
with open(r"nauki o sztuce_polon.json", 'w', encoding='utf-8') as jfile:
    json.dump(list_data, jfile, ensure_ascii=False, indent=4)
list_data = {}    
def get_names(url):
    global list_data
    
    response = requests.get(url)
    response2= response.json()['results']
    
    
    for resp in response2:
        usrname = resp['personalData']
        ident=resp['id']
        list_data[ident]=usrname
        #list_data.append(ident)
        
    try:
        token = response.json()['pagination']['token']
        
        get_names(f'https://radon.nauka.gov.pl/opendata/polon/employees?resultNumbers=100&disciplineName=literaturoznawstwo&penaltyMarker=false&token={token}')
    except KeyError:
        pass
    return

start_time=time.time()
get_names('https://radon.nauka.gov.pl/opendata/polon/employees?resultNumbers=100&disciplineName=literaturoznawstwo&penaltyMarker=false')
end_time=time.time()
print(end_time-start_time)

names=pd.DataFrame.from_dict(list_data,orient='index')
names.to_excel("literaturoznawstwo_id_names.xlsx", sheet_name='Sheet_name_1')  


#Radon Wyciągnięcie ORCIDOW
orcid_name=[]
for i, n in list_data.items():
    print(i)




    url = 'https://radon.nauka.gov.pl/opendata/scientist/search'

    

    body={
      "resultNumbers": 1,
      "token": None,
      "body": {
        "uid": i
,
        "firstName": None,
        "lastName": None,
        "employmentMarker": None,
        "employmentStatusMarker": None,
        "activePenaltyMarker": "No",
        "calculatedEduLevel": None,
        "academicDegreeMarker": None,
        "academicTitleMarker": None,
        "dataSources": None,
        "lastRefresh": None
      }}
    response=requests.post(url,  json=body)
    print(response.url)
    response2= response.json()['results']
    for result in response2:
        orcid_name.append(result['personalData'])
names=pd.DataFrame(orcid_name)
names.to_excel("literaturoznawstwo_id_names_orcid.xlsx", sheet_name='Sheet_name_1')  






###GOOD ORCID PART Zaciaganie prac po ORCID

to_compare = pd.read_excel ("C:/Users/dariu/literaturoznawstwo_id_names_orcid.xlsx", sheet_name='Sheet_name_1')
dicto = to_compare.to_dict('records')#('dict')['orcid']

alldata={}
counter=0
for values in dicto:
    counter+=1
    print(counter)
    orcid=values['orcid']
    if type(orcid)!=float:
        #print(orcid)
       
        alldata[orcid]={}
        try:
            url = f'https://pub.orcid.org/v3.0_rc2/{orcid}'
            headers = {'Accept': 'application/json'}
            
            r=requests.get(url, headers=headers)
            works=r.json()['activities-summary']['works']['group']
            #family_name=r.json()['person']['name']['family-name']['value']
            #name=r.json()['person']['name']['given-names']['value']
            papers_list=[]
            for work in works:
                summaries=work["work-summary"]
                for summary in summaries:
                    papers=summary['title']['title']['value']
                    papers_list.append(papers)
                    
            if type(values['middleName'])!=float:
                
                
                alldata[orcid][values['firstName']+' '+values['middleName']+' '+values['lastName']]=papers_list
                #alldata[orcid][values['firstName']+' '+values['middleName']+' '+values['lastName']].append(papers)
            else:
                #alldata[orcid]={}
                alldata[orcid][values['firstName']+' '+values['lastName']]=papers_list
                #alldata[orcid][values['firstName']+' '+values['lastName']].append(papers)
        except:
            continue
        

with open('orcid_works_literaturoznawstwo.json', 'w', encoding='utf8') as json_file:
    json.dump(alldata, json_file, ensure_ascii=False)                    
with open('orcid_works_literaturoznawstwo.json', encoding='utf-8') as fh:
    dataname = json.load(fh)
newdata={}
for data,value in dataname.items():
    newdata[data]=[]
    for k,v in value.items():
        newdata[data].append(k)
        for titles in v:
            newdata[data].append(titles)
            
            print(k, titles)         


names=pd.DataFrame.from_dict(newdata, orient='index')
names.to_excel('orcid_prace_literaturoznawstwo.xlsx')

