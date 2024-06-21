# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:28:50 2024

@author: dariu
"""

import pandas as pd

# Przykładowy DataFrame
data = {'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]}
df = pd.DataFrame(data)

# Iteracja po kolumnach
for column_name, series in df.iteritems():
    print(f"Column: {column_name}")
    print(series)
    
    
for column_name, series in df.iteritems():
    print(f"Column: {column_name}")
    for index, value in series.items():
        print(f"Row {index}: {value}")
    
import pandas as pd

# Przykładowy DataFrame
data = {'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]}
df = pd.DataFrame(data)

# Iteracja po wierszach
for index, row in df.iterrows():
    print(f"Index: {index}")
    print(row['name'])