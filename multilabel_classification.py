# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:31:56 2024

@author: dariu
"""

genres = [["poezja"], ["proza", "poezja"], ["proza"], ["dramat", "artykuł"]]

# Tworzenie unikalnego zestawu etykiet
unique_labels = set(label for sublist in genres for label in sublist)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

# Funkcja przekształcająca listę etykiet w binarną reprezentację
def labels_to_binary(label_list, label_to_index):
    binary_vector = [0] * len(label_to_index)
    for label in label_list:
        index = label_to_index[label]
        binary_vector[index] = 1
    return binary_vector

# Przekształcanie wszystkich etykiet
binary_labels = [labels_to_binary(sublist, label_to_index) for sublist in genres]

print(binary_labels)
