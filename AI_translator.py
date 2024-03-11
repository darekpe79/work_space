# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:47:27 2024

@author: dariu
"""

from transformers import pipeline
generator = pipeline("translation", model="sdadas/flan-t5-base-translator-en-pl")
sentence = "A team of astronomers discovered an extraordinary planet in the constellation of Virgo."
print(generator(sentence, max_length=512))
# [{'translation_text': 'Zespół astronomów odkrył niezwykłą planetę w gwiazdozbiorze Panny.'}]
