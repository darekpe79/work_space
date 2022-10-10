import requests
import json
from pymarc import MARCReader
from pymarc import parse_json_to_array
url = "https://data.bn.org.pl/api/institutions/bibs.json?author=Rokita%2C+Zbigniew+%281989-+%29&amp;limit=100"
data = requests.get(url).json()
nexturl=data['nextPage']

list=data['bibs']
while len(nexturl)>0:
    data = requests.get(data['nextPage']).json()
    nexturl=data['nextPage']
    
    
    list.extend(data['bibs'])
from pymarc import parse_json_to_array
lista=[]
counter=0
for l in list:
#    counter+=1
#    print(counter)


    marc=l['marc']
    lista.append(marc)
x=json.dumps(lista)

records = parse_json_to_array(x)
for record in records:
    print (record)
with open('my_new_marcArto_ISSN_ALL11.mrc' , 'wb') as data:
    for my_record in records:
### and write each record to it
        data.write(my_record.as_marc())

#%%
url = "https://data.bn.org.pl/api/institutions/bibs.json?author=Rokita%2C+Zbigniew+%281989-+%29&amp;limit=100"
data = requests.get(url).json()
nexturl=data['nextPage']

list=data['bibs']
while len(nexturl)>0:
    data = requests.get(data['nextPage']).json()
    nexturl=data['nextPage']
    
    
    list.extend(data['bibs'])
from pymarc import parse_json_to_array
lista=[]
counter=0
for l in list:
#    counter+=1
#    print(counter)


    marc=l['marc']
    lista.append(marc)
x=json.dumps(lista)

records = parse_json_to_array(x)
for record in records:
    print (record)
with open('my_new_marcArto_ISSN_ALL11.mrc' , 'wb') as data:
    for my_record in records:
### and write each record to it
        data.write(my_record.as_marc())

    