# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:51:45 2022

@author: darek
"""

from pymongo import MongoClient

client = MongoClient()


client.list_database_names()
database = client['local']
collection=database['fennica']
collection_names = database.list_collection_names()
for collect in collection_names:
    print (collect)
    
x=database.fennica.find({'leader':'01178cam a2200313zi 4500'})


for i,y in enumerate(x):
    print(y)
    fields=y['fields']
    for field in fields:
        if '001' in field:
            field['001']=i
    
    
    
    
    database.fennica.update_one({'_id':y['_id']},{'$set':{'fields':fields}})
    
for y in x:
    print(y)





database.fennica.updateOne(
   { '_id': "625041854df70252b5ee5bc3" },
   {
     $set: { "size.uom": "cm", status: "P" },
     $currentDate: { lastModified: true }
   }
)

coll.update_one({'_id': 100}, {'$set': {'fields.11.650.subfields.0.a': 'dupa'}})