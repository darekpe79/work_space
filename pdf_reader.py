# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:40:50 2023

@author: dariu
"""

import pdfreader
from pdfreader import PDFDocument, SimplePDFViewer
from io import BytesIO
fd = open("C:/Users/dariu/Downloads/Dialnet-LaGuerraEnUnNovelistaDeshumanizado-69059.pdf", "rb")
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
text='\n'.join(viewer.canvas.strings)  
