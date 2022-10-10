# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:31:17 2021

@author: darek
"""
import pandas as pd
from random import randint
from time import sleep
from bs4 import BeautifulSoup
from requests import get
from tqdm import tqdm



URL=r'https://portal.issn.org/api/search?search[]=MUST=default={}'.format(title)
page=get(URL)
bs=BeautifulSoup(page.content, features="lxml")
block=bs.find_all('div', {'class':'item-result-block'})
#print(block)


lista=[]

for element in block:
    #print(element)
    
    
    
    tytul=element.find('h5', {'class':'item-result-title'})
    issns=element.find('div', {'class':'item-result-content-text flex-zero'})
    if tytul:
        wlasciwy=tytul.text
        
        #wlasciwy=wlasciwy.split('    ')
        #print(wlasciwy)
        wlasciwy=wlasciwy.replace('\nKey-title \xa0', '')#strip('\nKey-title \xa0')
        #print(wlasciwy.strip())
        #break

    tytul2=bs.find_all('div', {'class':'item-result-content-text'})
    for tyt in tytul2:
        if 'Title proper:' in tyt.text:
            print(tyt.text)
    
    
    
    
    
    #issns2=issns.find('p')
    #issns=issns2.text
    
            
        
    
    
        
    if issns==None:
        issns=bs.find('div', {'sidebar-accordion-list-selected-item'}).text
        issns=issns.strip('ISN :')
        #print(issns)
        
        
    elif issns is not None:
        issns2=issns.find('p')
        issns=issns2.text
        issns=issns.strip('ISN :')
    else:
        issns='brak'
    
    
    


    




    

