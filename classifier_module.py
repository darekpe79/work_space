# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:23:58 2024

@author: dariu
"""

from transformers import pipeline

def classify_text(label, text="Przykładowy tekst."):
    classifier = pipeline("zero-shot-classification", model="xlm-roberta-large", framework="pt")
    result = classifier(text, candidate_labels=[label], multi_label=True)
    return label, result