# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:17:03 2021

@author: darek
"""

dct={}
dct['lala']={}
print(dct)
dct['lala']['dar']={'lal':'pral'}
print(dct)
dct['lala']['dar']['for']='r'
print(dct)


lista=[
        "Sievers, Kai",
        "Seger, Tapio",
        "Sievers, Kai 1930-1994",
        "Skiftesvik, Kai",
        "Siegers, Katharina",
        "Sivenius, Kaisa",
        "Stiefvater, Maggie",
        "Sievers, Kristina",
        "Sievers, Sami",
        "Meyer, Kai",
        "Sahamies, Kaisu",
        "Siv, Kim",
        "Säteri, Kai",
        "Sievers, Mikael",
        "Severn, David",
        "Siimes, Kari",
        "Sievers, Peppi"
    ]


#print(lista)
listanazw={}

    
for name in lista:
    listanazw[name]=[]
    for e in lista:
        
        
        if name in e:
            listanazw[name].append(e)
    

print(listanazw)
dictionary={}

dictionary['nazwa']={}
dictionary['nazwa']['imię i nazwisko']={}
dictionary['nazwa']['imię i nazwisko']=[{'imię':'darek','nazwisko':'perla'}]    