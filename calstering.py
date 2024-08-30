# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:02:17 2024

@author: dariu
"""

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Dane klientów
data = np.array([
    [25, 4, 2, 10],  # Klient A
    [45, 8, 5, 50],  # Klient B
    [30, 5, 3, 20],  # Klient C
    [50, 10, 6, 60], # Klient D
    [35, 6, 4, 30],  # Klient E
    [28, 7, 4, 25]   # Klient F
])

# Klasteryzacja hierarchiczna aglomeracyjna
linked = linkage(data, 'ward')

# Wizualizacja dendrogramu
plt.figure(figsize=(10, 7))
dendrogram(linked, 
           orientation='top',
           labels=['A', 'B', 'C', 'D', 'E', 'F'],
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Dane klientów
data = np.array([
    [25, 4, 2, 10],  # Klient A
    [45, 8, 5, 50],  # Klient B
    [30, 5, 3, 20],  # Klient C
    [50, 10, 6, 60], # Klient D
    [35, 6, 4, 30],  # Klient E
    [28, 7, 4, 25]   # Klient F
])

# Klasteryzacja hierarchiczna aglomeracyjna - Complete Linkage
linked_complete = linkage(data, 'complete')

# Wizualizacja dendrogramu
plt.figure(figsize=(10, 7))
dendrogram(linked_complete, 
           orientation='top',
           labels=['A', 'B', 'C', 'D', 'E', 'F'],
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Complete Linkage Dendrogram")
plt.show()

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Dane klientów
data = np.array([
    [25, 4, 2, 10],  # Klient A
    [45, 8, 5, 50],  # Klient B
    [30, 5, 3, 20],  # Klient C
    [50, 10, 6, 60], # Klient D
    [35, 6, 4, 30],  # Klient E
    [28, 7, 4, 25]   # Klient F
])

# Klasteryzacja hierarchiczna aglomeracyjna - Average Linkage
linked_average = linkage(data, 'average')

# Wizualizacja dendrogramu
plt.figure(figsize=(10, 7))
dendrogram(linked_average, 
           orientation='top',
           labels=['A', 'B', 'C', 'D', 'E', 'F'],
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Average Linkage Dendrogram")
plt.show()

