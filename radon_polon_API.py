# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:42:31 2023

@author: dariu
"""

import requests
import json
import time
#DANE ZINTEGROWANE
url = 'https://radon.nauka.gov.pl/opendata/scientist/search'


body={
  "resultNumbers": 1,
  "token": None,
  "body": {
    "uid": None,
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
response2= response.json()['results']
for result in response2:
    print(result['personalData'])
    
    
    
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

    
    
    
with open(r"literaturoznawstwo_polon.json", 'w', encoding='utf-8') as jfile:
    json.dump(all_data, jfile, ensure_ascii=False, indent=4)
list_data = []    
def get_names(url):
    global list_data
    
    response = requests.get(url)
    response2= response.json()['results']
    
    
    for resp in response2:
        usrname = resp['personalData']
        list_data.append(usrname)
        
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