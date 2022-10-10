# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:19:15 2021

@author: darek
"""

def initials(phrase):
    words = phrase.split()
    result = ""
    for word in words:
        result += word[0].upper()
    return result

print(initials("Universal Serial Bus")) # Should be: USB
print(initials("local area network")) # Should be: LAN
print(initials("Operating system")) # Should be: OS

string='abcd'
new_str=''
reverse=''
for letter in string.strip():
    
    new_str=new_str+letter.replace(' ', '')
    reverse=letter.replace(' ', '')+reverse

def get_word(sentence, n):
	# Only proceed if n is positive 
	if n > 0:
		words = sentence.split()
		# Only proceed if n is not more than the number of words 
		if n <= len(words):
			return(words[n-1])
	return('')
def convert_seconds(seconds):
    hours = seconds//3600
    minutes= (seconds-hours*3600)//60
    remaining_seconds=seconds-hours*3600-minutes*60
    return hours, minutes, remaining_seconds
x=convert_seconds(3602)



