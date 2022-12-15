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
to_compare = pd.read_excel ("C:/Users/dariu/genre_in_650_lemmatized_all.xlsx", sheet_name='czech')
genre_nationality=pd.read_excel('C:/Users/dariu/genre,nationality.xlsx', sheet_name='nationality')
nationality=genre_nationality['adjective']
pl_l=to_compare[650]
fin_l=to_compare['dictionary_cz']
list_without_nan_cz = [x for x in fin_l if not isinstance(x, float)] 
list_without_nan_pl = [x for x in pl_l if not isinstance(x, float)] 
nationality= [x for x in nationality if not isinstance(x, float)] 
output={}
for field in list_without_nan_pl:
    for tekst in nationality:
        tekst=tekst.strip()
    
        if re.search(rf"{tekst}(?= |$)", field, re.IGNORECASE):
            output[field]=tekst
excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("Nationality_in_650_only_genres_Cze.xlsx", sheet_name='Nation') 
with pd.ExcelWriter("Nationality_in_650.xlsx", engine='openpyxl', mode='a') as writer:  
    excel.to_excel(writer, sheet_name='Cz')
#%% lemmatize    
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import  regex as re
to_compare = pd.read_excel ("C:/Users/dariu/genre_in_650_lemmatized_all.xlsx", sheet_name='BN')
pl_l=to_compare['LCSH_PBL']
fin_l=to_compare['LCSH_BN']
list_without_nan_fi = [x for x in fin_l if not isinstance(x, float)] 
genre_nationality=pd.read_excel('C:/Users/dariu/genre,nationality.xlsx', sheet_name='nationality')
genre=genre_nationality['adjective']
lemmatizer = WordNetLemmatizer()
zlematyzowane={}

output={}
for g in tqdm(genre):
    words = word_tokenize(g)
    lemmat=[]
    for w in words:
        w=w.casefold().strip()
        
        
        lemma1=lemmatizer.lemmatize(w)
        #print(lemma1)
        lemmat.append(lemma1)
    
    lemmatized=' '.join(lemmat)
    
    
    
    for word in list_without_nan_fi:
        words2 = word_tokenize(word)
        lemmat2=[]
        for w2 in words2:
            
        
            word2=w2.casefold().strip()
            lemma2=lemmatizer.lemmatize(word2)
            #print(lemma2)
            lemmat2.append(lemma2)
        lemmatized2=' '.join(lemmat2)
       
        if re.search(rf"(?<= |^|-){lemmatized}(?= |$)", lemmatized2, re.IGNORECASE):
     
            output[word]=[lemmatized2,lemmatized]
            zlematyzowane[lemmatized]=lemmatized2
            

excel=pd.DataFrame.from_dict(output, orient='index') 
excel.to_excel("genre_in_650_lemmatized_BN.xlsx", sheet_name='fin')     

  
    
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

#%% NLP

from sklearn.feature_extraction.text import CountVectorizer
class Category:
  BOOKS = "BOOKS"
  CLOTHING = "CLOTHING"

train_x = ["i love the book", "this is a great book", "the fit is great", "i love the shoes"]
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]
vectorizer = CountVectorizer(binary=True)
train_x_vectors = vectorizer.fit_transform(train_x)
print(vectorizer.get_feature_names_out())
print(train_x_vectors.toarray())
from sklearn import svm

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
test_x = vectorizer.transform(['i love the books'])

clf_svm.predict(test_x)

#ngram_range

from sklearn import svm

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
test_x = vectorizer.transform(['i love the books'])

clf_svm.predict(test_x)
#train with spacy vectors
import spacy

nlp = spacy.load("en_core_web_lg")
docs = [nlp(text) for text in train_x]
print(docs[0].vector)
train_x_word_vectors = [x.vector for x in docs]
clf_svm_wv = svm.SVC(kernel='linear')
clf_svm_wv.fit(train_x_word_vectors, train_y)
test_x =["I love my earings"]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors =  [x.vector for x in test_docs]

clf_svm_wv.predict(test_x_word_vectors)

     



















