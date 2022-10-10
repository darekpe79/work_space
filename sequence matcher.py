# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:00:30 2021

@author: darek
"""

from difflib import *
str='cba'
str1='abc'
match=SequenceMatcher(a=str, b=str1)

print(match.ratio())
def matcher(str1, str2):
    match=SequenceMatcher(a=str1, b=str2)
    return match.ratio()
print (matcher ('str', 'slta'))
    
print(matcher(str, str1))

word_list = ['acdefgh', 'abcd','adef','cdea']
str1 = 'abcd'
matches = get_close_matches(str1, word_list, n=2, cutoff=0.3)
print(matches)  
  