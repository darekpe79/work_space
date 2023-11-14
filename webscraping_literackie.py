# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:19:53 2023

@author: dariu
"""

import requests
from bs4 import BeautifulSoup

url = "https://dekadaliteracka.pl/2023/09/grafonotki-czym-sa-i-kiedy-sie-przydaja/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
title = soup.title.text
date = soup.find('time').text
print(f"Date of Publication: {date}")
author_tag = soup.find('a', href=True, attrs={"href": lambda x: x and "author" in x})
if author_tag:
    author = author_tag.text
    print(f"Author: {author}")
else:
    print("Author not found.")
print(title)


import requests
from bs4 import BeautifulSoup

url = 'https://dekadaliteracka.pl/'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the titles using the provided tag and class
titles = soup.find_all('h1', class_='post-title')

# Print the titles
for title in titles:
    print(title.a.text.strip())
    
    
from bs4 import BeautifulSoup
import requests

url = "https://film.dziennik.pl/recenzje/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extracting links based on the specific structure of the articles
article_links = soup.select('div.listItem.listItemSolr.itarticle a')

extracted_links = [link['href'] for link in article_links if 'artykuly' in link['href']]

for link in extracted_links:
    print(link)
    
from bs4 import BeautifulSoup
import requests
import time
import json
base_url = "https://film.dziennik.pl/recenzje"
current_page = ""
all_links = []

while True:
    url = base_url + current_page
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extracting links based on the specific structure of the articles
    article_links = soup.select('div.listItem.listItemSolr.itarticle a')
    extracted_links = [link['href'] for link in article_links if 'artykuly' in link['href']]
    all_links.extend(extracted_links)

    # Check for the presence of the 'next' link
    next_link = soup.select_one('a.next')
    if next_link:
        # Extract the page number from the 'next' link's href and append it to the base URL
        current_page = "," + next_link['href'].split(',')[-1]
        time.sleep(1)
    else:
        break

# for link in all_links:
#     print(link)



def extract_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
        
    related_topics_div = soup.find('div', id="relatedTopics")
    
    topics = []
    if related_topics_div:
        for span in related_topics_div.find_all('span', class_="relatedTopic"):
            a_tag = span.find('a')
            if a_tag:
                topic_info = {
                    "title": a_tag.get('title'),
                    "href": a_tag.get('href')
                }
                topics.append(topic_info)

    # Extracting title
    title = soup.title.text.split(" - ")[0]

    # Extracting author
    author_div = soup.select_one('.name.nameOfAuthor')
    author = author_div.text.strip() if author_div else "Unknown"

    # Extracting date
    date_div = soup.select_one('.datePublished')
    date = date_div.text.strip() if date_div else "Unknown"
    
    script_tag = soup.find('script', type="application/ld+json")
    data = json.loads(script_tag.string)
    article_body = data.get("articleBody", "Article body not found.")
    article_soup = BeautifulSoup(article_body, 'html.parser')

    list_zobacz_rowniez = []
    for a_tag in article_soup.find_all('a'):
        preceding_text = a_tag.previous_sibling
        if preceding_text and "Zobacz również" in preceding_text:
            link_url = a_tag['href']
            list_zobacz_rowniez.append(link_url)
            a_tag.extract()

    links = []
    for a_tag in article_soup.find_all('a'):
        link_info = {
            "href": a_tag.get('href'),
            "text": a_tag.text
        }
        links.append(link_info)
        a_tag.extract()

    updated_article_body = article_soup.get_text()

    return {
        'url': url,
        "title": title,
        "author": author,
        "date": date,
        "text": updated_article_body,
        'links': links,
        "zobacz_rowniez": list_zobacz_rowniez,
        "topics":topics
    }


details_list=[]
for link in all_links:
    time.sleep(1)
    print(link)
    
    details = extract_details(link)
    details_list.append(details)
details=extract_details("https://film.dziennik.pl/recenzje/artykuly/8485383,ennio-morricone-film-recenzja-dobrycynk.html")

with open('film_dziennik_links_splitted.json', 'w', encoding='utf-8') as out:
    json.dump(details_list, out, ensure_ascii=False, indent=4)
    
from bs4 import BeautifulSoup

html = '''Zobacz również<a href='https://film.dziennik.pl/nowosci-vod/artykuly/9297931,infamia-nie-przynosi-wstydu-ani-romom-ani-hip-hopowi-dobrycynk.html' id='df69ed17-37b9-4cea-981d-8adf3520bd53'>'Infamia' nie przynosi wstydu. Ani Romom, ani hip-hopowi [#DobryCynk]</a>'''

soup = BeautifulSoup(html, 'html.parser')

# Wyszukaj wszystkie linki, które zaczynają się od "Zobacz również"
for link in soup.find_all('a', string=lambda text: text and text.startswith('Zobacz również')):
    link.extract()

print(soup.get_text())


import requests
from bs4 import BeautifulSoup

def extract_related_topics(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    related_topics_div = soup.find('div', id="relatedTopics")
    
    topics = []
    if related_topics_div:
        for span in related_topics_div.find_all('span', class_="relatedTopic"):
            a_tag = span.find('a')
            if a_tag:
                topic_info = {
                    "title": a_tag.get('title'),
                    "href": a_tag.get('href')
                }
                topics.append(topic_info)
    
    return topics

# Test
topics = extract_related_topics("https://film.dziennik.pl/news/artykuly/9318123,trzy-lata-spokoju-scenarzysci-z-hollywood-zawarli-nowy-kontrakt.html")
print(topics)
