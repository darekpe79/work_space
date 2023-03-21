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
orcid_name=[]
for i, n in list_data.items():
    print(i)




    url = 'https://radon.nauka.gov.pl/opendata/scientist/search'

    

    body={
      "resultNumbers": 1,
      "token": None,
      "body": {
        "uid": i,
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

URL=' https://orcid.org/oauth/token'


HEADER= {'Accept': 'application/json'}
#VtvRRx
DATA={'client_id':'APP-VJ90JLOO02X43UOZ',
  'client_secret':'e9390289-32cb-49d1-91d6-b3fe54ca29f4',
  'grant_type':'authorization_code',
  'redirect_uri':'https://orcid.org',  
  'code':'VtvRRx',
  'scope':'/authenticate'}
response=requests.post(URL,headers=HEADER, json=DATA )
https://orcid.org/oauth/authorize?client_id=APP-VJ90JLOO02X43UOZ&response_type=code&scope=/authenticate&redirect_uri=https://orcid.org
response.json()
https://orcid.org/oauth/authorize?client_id=APP-VJ90JLOO02X43UOZ&response_type=code&scope=/authenticate&redirect_uri=https://orcid.org
access_token=e50c4b8a-0eec-4f3c-9868-886cdba3bd9c
import orcid
api = orcid.PublicAPI(institution_key, institution_secret, sandbox=True)
token = api.get_token_from_authorization_code(authorization_code,
                                              redirect_uri)
Client ID APP-VJ90JLOO02X43UOZ
Client secret e9390289-32cb-49d1-91d6-b3fe54ca29f4
import orcid
api = orcid.PublicAPI(institution_key, institution_secret, sandbox=True)
token = api.get_token_from_authorization_code('VtvRRx',
                                              'https://orcid.org')