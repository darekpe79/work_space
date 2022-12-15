# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:30:40 2022

@author: dariu
"""
import os
from pathlib import Path
from filecmp import cmp
from tqdm import tqdm
  
  
# list of all documents
DATA_DIR = Path('C:/Users/dariu/compare')
files = sorted(os.listdir(DATA_DIR))
  
# List having the classes of documents
# with the same content
duplicateFiles = []
  
# comparison of the documents
for file_x in tqdm(files):
  
    if_dupl = False
  
    for class_ in duplicateFiles:
        # Comparing files having same content using cmp()
        # class_[0] represents a class having same content
        if_dupl = cmp(
            DATA_DIR / file_x,
            DATA_DIR / class_[0],
            shallow=False
        )
        if if_dupl:
            class_.append(file_x)
            break
  
    if not if_dupl:
        duplicateFiles.append([file_x])
  
# Print results
print(duplicateFiles)
