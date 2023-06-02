# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:40:50 2023

@author: dariu
"""

import pdfreader
from definicje import *
from pdfreader import PDFDocument, SimplePDFViewer
from io import BytesIO
fd = open("C:/Users/dariu/Downloads/Dialnet-LaConstitucionDelLexicoExtremenoSegunSeMuestraEnEl-798510 (1).pdf", "rb")
doc = pdfreader.PDFDocument(fd)
all_pages = [p for p in doc.pages()]
all_pages[1].strings
with open("C:/Users/dariu/Downloads/Dialnet-LaGuerraEnUnNovelistaDeshumanizado-69059.pdf", "rb") as f:
    stream = BytesIO(f.read())
    
doc2 = PDFDocument(stream)
l=doc.metadata
doc.header.version
viewer = SimplePDFViewer(fd)
viewer.metadata

for canvas in viewer:
    page_strings = canvas.strings
    page_text = canvas.text_content

viewer.navigate(1)
viewer.render()  
plain_text = "".join(viewer.canvas.strings)
lista=viewer.canvas.strings
compose_data('Dialectolog’a, geograf’a lingŸ’stica, extreme–o, lŽxico.')

from PyPDF2 import PdfReader
reader = PdfReader("C:/Users/dariu/Downloads/Dialnet-PoeticaEInnovacionEnLaAlejandraDeLicofron-798513.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()


match = re.search(r"Keywords: ([\w\s,]+)", text)
if match:
    keywords = match.group(1)
    keyword_list = [keyword.strip() for keyword in keywords.split(',')]
    print(keyword_list)

match = re.search(r"Palabras clave: ([\w\s,]+)", text)
if match:
    keywords = match.group(1)
    keyword_list = [keyword.strip() for keyword in keywords.split(',')]
    print(keyword_list)