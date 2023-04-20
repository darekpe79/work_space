# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:08:03 2023

@author: dariu
"""
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
import regex as re
from datetime import date   
from concurrent.futures import ThreadPoolExecutor  
from geopy.geocoders import Nominatim
import csv
geolocator = Nominatim(user_agent="geoapiExercises")

Latitude = "52.23"
Longitude = "21.0111"
 
location = geolocator.reverse(Latitude+","+Longitude)
field650=pd.read_excel('C:/Users/dariu/miasto-dane.xlsx', sheet_name='Sheet_name_1',dtype=str) 
dictionary650=field650.to_dict('records')
# Display
print(location)


geodata={} 
counter=0
with open('D:/Nowa_praca/dump(1).csv', newline='', encoding='utf-8', errors='ignore' ) as f:
    reader = csv.reader(f)
    
    for row in reader:
        print(row)
        break
        
        counter+=1
        print(counter)
        
        if counter==20:
            break
        
        try:    
    
            latitude=row[2]

            longitude=row[1]
            name=row[6]
            
            if latitude and longitude:
                location = geolocator.reverse(latitude+","+longitude,language='en')
                address = location.raw['address']
                
                country = address.get('country', '')
                geodata[name]=address
        except:
            continue
        
           
with open ('geodata_all.json', 'w', encoding='utf-8') as file:
    json.dump(dictionary650,file,ensure_ascii=False) 

geodata=  list(geodata)          
        
with open('D:/pub_places.json', encoding='utf-8') as fh:
    dataname = json.load(fh)        
geodata={}   
for data in tqdm(dataname):
    try:
        latitude=data['coordinates'].split(',')[0]
        longitude =data['coordinates'].split(',')[1]
        if latitude and longitude:
            location = geolocator.reverse(latitude+","+longitude,language='en')
            address = location.raw['address']
            country = address.get('country', '')
            geodata[data['name']]=address
    except:
        continue
geodata_pd=pd.DataFrame.from_dict(geodata,orient='index') 
geodata_pd.to_excel("miasto-dane.xlsx", sheet_name='Sheet_name_1')    

test_list = [4, 5, 6, 3, 9]
insert_list = [2, 3]
# initializing position
pos = 3
test_list[3:3] = insert_list    