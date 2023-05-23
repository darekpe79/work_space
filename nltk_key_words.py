# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:29:19 2023

@author: dariu
"""

from nltk import tokenize
from operator import itemgetter
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
stopword=set(stopwords.words('english'))
wn = nltk.WordNetLemmatizer() #Lemmatizer

def clean_the_text(text):
        
        #Replace non-word characters with empty space
        text = re.sub('[^A-Za-z0-9\s]', ' ', text)
        
        #Remove punctuation
        text = ''.join([word for word in text if word not in
               string.punctuation])
        
        #Bring text to lower case
        text = text.lower()
        
        #Tokenize the text
        tokens = re.split('\W+', text)
        
        #Remove stopwords
        text = [word for word in tokens if word not in stopword]
        
        #Lemmatize the words
        text = [wn.lemmatize(word) for word in text]
        
        #Return text
        return text
def common_words(l_1, l_2):
        matching_words = set.intersection(set(l_1), set(l_2))
        return matching_words
