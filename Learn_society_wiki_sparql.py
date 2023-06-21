import requests

import pandas as pd

def get_scientific_societies():
    query = '''
    SELECT ?item ?itemLabel ?website
    WHERE {
        ?item wdt:P31/wdt:P279* wd:Q955824.
        ?item wdt:P856 ?website.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    '''
    return query


def execute_sparql_query(query):
    url = 'https://query.wikidata.org/sparql'
    headers = {'Accept': 'application/sparql-results+json'}
    data = {'query': query}
    response = requests.get(url, headers=headers, params=data)
    return response.json()

def find_scientific_societies():
    query = get_scientific_societies()
    result = execute_sparql_query(query)
    datas={}
    for item in result['results']['bindings']:
        label = item['itemLabel']['value']
        website = item['website']['value']
        item_uri = item["item"]["value"]
        print(f"learned Society: {label}")
        print(f"Website: {website}")
        print(item_uri)
        datas[item_uri]=[]
        datas[item_uri].append(label)
        datas[item_uri].append(website)
    return datas
datas=find_scientific_societies()

fin_df=pd.DataFrame.from_dict(datas, orient='index')
fin_df.to_excel("wikidata_all_instance_of_learn_societyQ955824.xlsx") 

