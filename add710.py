# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:07:56 2022

@author: darek
"""
from definicje import *
import json
import pandas as pd
import os
from tqdm import tqdm
pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$b).*?(?=\$|$)'
pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'


paths=["E:/Python/do_prob.mrk"]

pattern3=r'(?<=\$a).*?(?=\$|$)' 
#daty
pattern4='(?<=\$v).*?(?=\$|$)'
#pattern5='(?<=\$1http:\/\/viaf\.org\/viaf\/).*?(?=\$|$| )'
slownik={'Wojciech Sumliński Reporter':"12345"}



#val100=[]
for plik in paths:
    record=list_of_dict_from_file(plik)
    for rec in record:
        for key, val in rec.items():
            if key=='710':
                new_val=[]
                for v in val: 
                
                    #print(v)
                    name = re.findall(pattern3, v)
                    if name:
                        if name[0] in slownik:
                            
                            #print(v)
                            #index=val.index(v)
                            #print(index)
                            index=[i for i, e in enumerate(val) if e == v]
                            print(index)
                            for i in index:
                                print(val[i])
                                print(slownik[name[0]])
                                new_val=val[i]+'  '+slownik[name[0]]
                                val[i]=new_val
                
                            
                        
                
                                
                        
                    
                            
                            
                            
                
                
                
                
                
                
                
                
                
                
                
                
                

                

                
                

        



