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

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in tqdm(doc.paragraphs):
        fullText.append(ts.google(para.text, from_language='pl', to_language='en'))
    return '\n'.join(fullText)
tekst=getText('C:/Users/dariu/Downloads/Globalizacja_a_wartościowanie_na_pr (1).docx')

