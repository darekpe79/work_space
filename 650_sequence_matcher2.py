# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:12:33 2022

@author: dariu
"""

import pandas as pd
from difflib import SequenceMatcher    
from tqdm import tqdm
from re import search

to_compare = pd.read_excel ("C:/Users/dariu/650_to_compare.xlsx", sheet_name='to_compare')
to_compare2 = pd.read_excel ("C:/Users/dariu/FI-CZ-google_dict_same650.xlsx", sheet_name='Sheet_name_1')
to_compare['VAR1'] = to_compare.apply(lambda x : SequenceMatcher(None, x['dictionary_cz'], x['dictionaryfin']).ratio() if not any ([isinstance(x['dictionary_cz'], float),isinstance(x['dictionaryfin'], float)]) else 'brak',axis=1)
wynik_l=to_compare2[0]
pl_l=to_compare['LCSH_BN']
list_without_nan_cz = [x for x in wynik_l if not isinstance(x, float)] 
list_without_nan_fi = [x for x in pl_l if not isinstance(x, float)] 
output={}
for word in tqdm(list_without_nan_cz):
    for word2 in list_without_nan_fi:
        if word.casefold().strip()==word2.casefold().strip():
            output[word]= word2
        else:
            sorted_string = sorted(word.casefold().strip().split())
            joined1=' '.join(sorted_string)
            sorted_string2 = sorted(word2.casefold().strip().split())
            joined2=' '.join(sorted_string2)
            s = SequenceMatcher(None, joined1, joined2).ratio()
            if s=='1.0':
                output[word]= word2
excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("Bn-CZ-Fi-google_dict_same650.xlsx", sheet_name='Sheet_name_1')                 
            
from difflib import SequenceMatcher            
s = SequenceMatcher(None, "polska poezja", "poezja polska").ratio()

string_value = 'I like to sort'
sorted_string = sorted(string_value.split())
joined1=' '.join(sorted_string)
string_value1 = 'sort I like to'
sorted_string1 = sorted(string_value1.split())
joined2=' '.join(sorted_string)
s = SequenceMatcher(None, joined1, joined2).ratio()
#%% natioanlity, genre in 650
import re
to_compare = pd.read_excel ("C:/Users/dariu/650_to_compare.xlsx", sheet_name='to_compare')
genre_nationality=pd.read_excel('C:/Users/dariu/genre,nationality.xlsx', sheet_name='nationality')
nationality=genre_nationality['adjective']
pl_l=to_compare['LCSH_PBL']
fin_l=to_compare['dictionary_cz']
list_without_nan_cz = [x for x in fin_l if not isinstance(x, float)] 
list_without_nan_pl = [x for x in pl_l if not isinstance(x, float)] 
nationality= [x for x in nationality if not isinstance(x, float)] 
output={}
for field in list_without_nan_cz:
    for tekst in nationality:
        tekst=tekst.strip()
    
        if re.search(rf"{tekst}(?= |$)", field, re.IGNORECASE):
            output[field]=tekst
excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("Nationality_in_650.xlsx", sheet_name='fin') 
with pd.ExcelWriter("Nationality_in_650.xlsx", engine='openpyxl', mode='a') as writer:  
    excel.to_excel(writer, sheet_name='Cz')
#%% lemmitze    
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
to_compare = pd.read_excel ("C:/Users/dariu/650_to_compare.xlsx", sheet_name='to_compare')
pl_l=to_compare['LCSH_PBL']
fin_l=to_compare['dictionary_cz']
genre_nationality=pd.read_excel('C:/Users/dariu/genre,nationality.xlsx', sheet_name='genre')
genre=genre_nationality['Genre']
lemmatizer = WordNetLemmatizer()
zlematyzowane=[]
for g in genre:
    words = word_tokenize(g)
    lemmat=[]
    for w in words:
        w=w.casefold().strip()
        lemmatizer = WordNetLemmatizer()
        
        lemma1=lemmatizer.lemmatize(w)
        print(lemma1)
        lemmat.append(lemma1)
    zlematyzowane.append(' '.join(lemmat))    
       

    pass

    
sentence="rocks"
words = word_tokenize(sentence)
ps = PorterStemmer()
for w in words:
	rootWord=ps.stem(w)
	print(rootWord)
from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()
print("rocks :", lemmatizer.lemmatize("Songs"))
print("corpora :", lemmatizer.lemmatize("corpora"))
            