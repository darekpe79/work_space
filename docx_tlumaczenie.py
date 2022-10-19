# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:58:43 2022

@author: dariu
"""

from docx import Document
from io import StringIO
import docx
import translators as ts
from tqdm import tqdm
import regex as re

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in tqdm(doc.paragraphs):
        fullText.append(ts.google(para.text, from_language='pl', to_language='en'))
    return '\n'.join(fullText)
tekst=getText('C:/Users/dariu/Downloads/Globalizacja_a_wartościowanie_na_pr (1).docx')


from unidecode import unidecode
def prepare_title(title):
  
  title = "".join(e for e in title if e.isalnum()).strip().lower()
  hashed_title = unidecode(title)
  return hashed_title
prepare_title('Bielsko-Biała')

from cleantext import clean
import cleantext
clean("[,.Bielsko-Biała]")
cleantext.clean_words('Your s$ample !!!! tExt3% to   cleaN566556+2+59*/133 wiLL GO he123re', all=True)
import string
text = "[,.Bielsko-Biała]"
text_clean = "".join([i for i in text if i not in string.punctuation])
text_clean


