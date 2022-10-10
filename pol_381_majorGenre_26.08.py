# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 09:22:35 2022

@author: dariu
"""
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
#from googletrans import Translator
from itertools import zip_longest
import regex as re
import requests
import xlsxwriter
import datetime

delete=[r'\\$iMajor genre$aLyrical poetry$leng',
r'\\$iMajor genre$aFiction$leng',r'\\$iMajor genre$aOther$leng',r'\\$iMajor genre$aDrama$leng',
r'\\$iMajor genre$aSecondary literature$leng',r'\\$iMajor genre$aLiterature$leng']


genre=pd.read_excel('D:/Nowa_praca/380BN_doMrk.xlsx', sheet_name=1) 

genre_dict = dict(zip(genre['desk_380'].tolist(),genre['rodzaj'].tolist()))
pattern3=r'(?<=\$a).*?(?=\$|$)' 



for plik in ["C:/Users/dariu/bn_chapters_2022-08-26.mrk",
"C:/Users/dariu/bn_articles_2022-08-26.mrk",
"C:/Users/dariu/bn_books_2022-08-26.mrk"]:
    path=plik.split('/')[-1].split('.')[-2]
    
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    
    rekordy=[]
    for rekord in tqdm(dictrec):
        
        

        
        literature_genre=[]
        secondary=[]
        nowe={}
        # dobre381=[]
        for key, val in rekord.items():
             nowe[key]=val
            
        #     if key=='381':
                
        #         for val_381 in val.split('❦'):
                    
                    
        #             #print(val_381)
        #             if val_381 in delete:
        #                 pass
        #             else:
        #                 dobre381.append(val_381)
        #         if dobre381:
        #             nowe['381']='❦'.join(dobre381) 
        #         else:
        #             del(nowe['381'])

            
             if key=='380':

                for val_655 in val.split('❦'):
                    #print(val_655)
                    #val_655=re.findall(pattern3, val_655)
                    
                    #if val_655:
                    #    val_655=val_655[0]
                        #print(val_655)
                
                
                

                    if val_655 in genre_dict:
                        print(val_655)
                        
                        if genre_dict[val_655]==r'\\$iMajor genre$aSecondary literature$leng':
                           secondary.append(genre_dict[val_655])
                
                        
                        else:
                           literature_genre.append(genre_dict[val_655])
         

                if literature_genre and secondary:
                    if '381' in nowe:
                        list_set=unique(literature_genre+nowe['381'].split('❦'))
         
                        nowe['381']='❦'.join(list_set)
        
                    else:
                        list_set = unique(literature_genre)
                                                          
                        nowe['381']='❦'.join(list_set)
        
                        
                    if '380' in nowe:
                        
                        list_set=unique(secondary+[r'\\$iMajor genre$aLiterature$leng']+nowe['380'].split('❦'))
        
                        nowe['380']='❦'.join(list_set)
        
                    else:
        
                                                          
                        nowe['380']='❦'.join((secondary+[r'\\$iMajor genre$aLiterature$leng']))
                        
                elif literature_genre:
                    if '381' in nowe:
                        list_set=unique(literature_genre+nowe['381'].split('❦'))
          
                        nowe['381']='❦'.join(list_set)
        
                    else:
                        list_set = unique(literature_genre)
                                                          
                        nowe['381']='❦'.join(list_set)
               
                        
                    if '380' in nowe:
                        
                        list_set=([r'\\$iMajor genre$aLiterature$leng']+nowe['380'].split('❦'))
                        list_set=unique(list_set)
                     
                        nowe['380']='❦'.join(list_set)
                        
                    else:
                        
                                                          
                        nowe['380']=r'\\$iMajor genre$aLiterature$leng'
                        
                    
                elif secondary:
                    if '380' in nowe:
                        
                        list_set=unique(secondary+nowe['380'].split('❦'))
                      
                        nowe['380']='❦'.join(list_set)
                        
                    else:
                        
                                                          
                        nowe['380']=secondary[0]
        
        #output_dict = {'LDR': nowe.get('LDR')}
        #del nowe['LDR']
        #output_dict.update(dict(sorted(nowe.items())))
                
        
        
        #dictio=sortdict(nowe)
        
        
        
            
        rekordy.append(nowe)   
    date_object = datetime.date.today()        
    to_file (path+'2_'+str(date_object)+'.mrk', rekordy)  

lista=(['lala']+['kupa'])

# =381  \\$iMajor genre$aLyrical poetry$leng
# =381  \\$iMajor genre$aFiction$leng
# =381  \\$iMajor genre$aOther$leng
# =381  \\$iMajor genre$aDrama$leng
# =381  \\$iMajor genre$aSecondary literature$leng

#yso_dict={'82':['\\\\$aLyrical poetry$leng','\\\\$aLiryka$lpol','\\\\$aLyrická poezie$lcze','\\\\$aRunot$lfin'], 
#          '84':['\\\\$aFiction$leng','\\\\$aEpika$lpol', '\\\\$aProza$lcze','\\\\$aKertomakirjallisuus$lfin'],
#          '83':['\\\\$aDrama$leng','\\\\$aDramat$lpol', '\\\\$aDrama$lcze','\\\\$aNäytelmät$lfin'],
#          '80':['\\\\$aOther$leng','\\\\$aInne$lpol','\\\\$aJiný$lcze','\\\\$aMuu$lfin'],
#          '81':['\\\\$aOther$leng','\\\\$aInne$lpol','\\\\$aJiný$lcze','\\\\$aMuu$lfin'],
#          '85':['\\\\$aOther$leng','\\\\$aInne$lpol','\\\\$aJiný$lcze','\\\\$aMuu$lfin'],
#          '86':['\\\\$aSecondary literature$leng','\\\\$aLiteratura przedmiotu$lpol','\\\\$aSekundární literatura$lcze','\\\\$aToissijainen kirjallisuus$lfin']
#          }

#yso_dict={'82':['\\\\$aLyrical poetry$leng','\\\\$aLiryka$lpol','\\\\$aLyrická poezie$lcze','\\\\$aRunot$lfin'], 
#          '84':['\\\\$aFiction$leng','\\\\$aEpika$lpol', '\\\\$aProza$lcze','\\\\$aKertomakirjallisuus$lfin'],
#          '83':['\\\\$aDrama$leng','\\\\$aDramat$lpol', '\\\\$aDrama$lcze','\\\\$aNäytelmät$lfin']}