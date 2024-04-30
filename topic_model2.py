# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:26:03 2024

@author: dariu
"""

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Przykładowe dokumenty
docs = [
    "Apple is looking into buying U.K. startup for $1 billion",
    "Autonomous cars shift insurance liability toward manufacturers",
    "San Francisco considers banning sidewalk delivery robots",
    "Climate change is threatening the arctic wildlife",
    "Investments in renewable energy have been increasing steadily",
    "The stock market faced a significant downturn yesterday",
    "Advancements in artificial intelligence are revolutionizing healthcare",
    "Electric vehicles are gaining popularity globally"
]


# Inicjalizacja BERTopic
topic_model = BERTopic(low_memory=True, umap_model=None)

# Trenowanie modelu na dokumentach
topics, probabilities = topic_model.fit_transform(docs)

# Wypisanie wyników
print("Znalezione tematy i ich reprezentatywne słowa:")
for topic, words in topic_model.get_topic_info().iterrows():
    if topic == -1:
        print(f"Noise cluster (Topic {topic}): {words['Name']}")
    else:
        print(f"Topic {topic}: {words['Name']}")

print("\nLista słów dla każdego tematu:")
for topic_number in sorted(topic_model.get_topics()):
    print(f"Topic {topic_number}: {topic_model.get_topic(topic_number)}")

# Można również wyświetlić dokumenty przypisane do każdego tematu
print("\nDokumenty przypisane do tematów:")
for doc, topic_number in zip(docs, topics):
    print(f"Dokument: \"{doc}\" -> Temat: {topic_number}")
