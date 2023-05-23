# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 18:09:31 2023

@author: dariu
"""

import csv
from tqdm import tqdm

import pandas as pd



csv_filename = 'C:/Users/dariu/Downloads/zadanie/delivery_zip.csv'
orders=[]
with open(csv_filename) as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader):
        orders.append(row)
        
csv_filename = 'C:/Users/dariu/Downloads/zadanie/customers_zip.csv'
customers_zip=[]
with open(csv_filename) as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader):
        customers_zip.append(row)
customers=set()        
for order in tqdm(orders):
    client_order_id=(order['customer_id'])
    for customer in customers_zip:
        if customer['customer_id']==client_order_id:
            
            if order['zip_del']!=customer['zip_cust']:
                customers.add(client_order_id)



#Second
                
customers_tuple=[]

for customer in tqdm(customers_zip):
    customers_tuple.append((customer['zip_cust'],customer['customer_id']))
 
    
orders_tuple=[]
for order in tqdm(orders):
    orders_tuple.append((order['zip_del'],order['customer_id']))

counter=0    
for order in tqdm(orders_tuple):
    for customer in customers_tuple:
        if order[1]==customer[1]:
            if order[0]!=customer[0]:
                counter+=1
                
            
#third
customers_dict={}

for customer in tqdm(customers_zip):
    
    customers_dict[customer['customer_id']]=customer['zip_cust'] 
customer_id   
zip_cust

sett=set()
for order in tqdm(orders):
    if order['customer_id'] in customers_dict:
        if customers_dict[order['customer_id'] ]!=order['zip_del']:
            sett.add(order['customer_id'])
    
        
    
    