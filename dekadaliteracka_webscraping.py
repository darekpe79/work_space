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
    else:
        break

for link in all_links:
    print(link)


from bs4 import BeautifulSoup
import requests

def extract_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extracting title
    title = soup.title.text.split(" - ")[0]

    # Extracting author
    author_div = soup.select_one('.name.nameOfAuthor')
    author = author_div.text.strip() if author_div else "Unknown"

    # Extracting date
    date_div = soup.select_one('.datePublished')
    date = date_div.text.strip() if date_div else "Unknown"

    return {
        "title": title,
        "author": author,
        "date": date
    }

url = "https://film.dziennik.pl/recenzje/artykuly/9304085,turkusowa-suknia-to-rekodzielo-najwyzszej-proby-dobrycynk.html"
details = extract_details(url)
print(details)
