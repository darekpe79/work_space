# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:00:07 2022

@author: dariu
"""

import requests
import urllib.parse
adres=urllib.parse.unquote('http://finto.fi/rest/v1/yso/data?uri=http%3A%2F%2Fwww.yso.fi%2Fonto%2Fyso%2Fp3537&format=&format=application/json')
print(adres)
number='p3537'
patternYSO=r'(?<=\/yso\/).*?(?=\$|$)'
response = requests.get(url=f'http://finto.fi/rest/v1/yso/data?uri=http://www.yso.fi/onto/yso/{number}&format=&format=application/json').json()
graph=response['graph']
for gr in graph:
    #print(g)
    if 'hiddenLabel' in gr:
        for g in gr['hiddenLabel']:
            if g['lang']=='en':
                print(g['value'])