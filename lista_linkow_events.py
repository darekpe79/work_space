# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:51:02 2024

@author: dariu
"""

import requests
from bs4 import BeautifulSoup
import time

def google_search(query):
    # Przygotuj zapytanie do wyszukiwarki Google
    search_url = f"https://www.google.com/search?q={query}&num=100"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    
    # Debug: Sprawdź status odpowiedzi
    if response.status_code != 200:
        print(f"Request failed with status code: {response.status_code}")
        return None
    
    return response.text

def extract_links(html):
    # Sprawdź, czy HTML nie jest pusty
    if html is None:
        return []
    
    # Przetwarzanie HTML i wyciąganie linków
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    keywords = ["program", "agenda", "schedule", "timetable", "harmonogram", "rozkład", "grafik"]
    for item in soup.find_all('a'):
        href = item.get('href')
        if href and 'http' in href:
            # Zapisz linki zawierające słowa kluczowe
            if any(keyword in href.lower() for keyword in keywords):
                links.append(href)
    
    # Debug: Wydrukuj znalezione linki dla danego HTML
    if not links:
        print("No links found in the current HTML.")
    else:
        print(f"Found {len(links)} links.")
    
    return links

def main():
    search_queries = [  # Angielskie wersje
        "conference 2024 program",
        "conference 2024 agenda",
        "conference 2024 schedule",
        "conference 2024 timetable",
        "festival 2024 program",
        "festival 2024 agenda",
        "festival 2024 schedule",
        "festival 2024 timetable",
        "webinar 2024 program",
        "webinar 2024 agenda",
        "webinar 2024 schedule",
        "webinar 2024 timetable",
        "concert 2024 program",
        "concert 2024 agenda",
        "concert 2024 schedule",
        "concert 2024 timetable",
        "summit 2024 program",
        "summit 2024 agenda",
        "summit 2024 schedule",
        "summit 2024 timetable",
        "workshop 2024 program",
        "workshop 2024 agenda",
        "workshop 2024 schedule",
        "workshop 2024 timetable",
        "symposium 2024 program",
        "symposium 2024 agenda",
        "symposium 2024 schedule",
        "symposium 2024 timetable",
        "expo 2024 program",
        "expo 2024 agenda",
        "expo 2024 schedule",
        "expo 2024 timetable",
        "forum 2024 program",
        "forum 2024 agenda",
        "forum 2024 schedule",
        "forum 2024 timetable",

        # Polskie wersje
        "konferencja 2024 program",
        "konferencja 2024 agenda",
        "konferencja 2024 harmonogram",
        "konferencja 2024 rozkład",
        "konferencja 2024 grafik",
        "festiwal 2024 program",
        "festiwal 2024 agenda",
        "festiwal 2024 harmonogram",
        "festiwal 2024 rozkład",
        "festiwal 2024 grafik",
        "webinar 2024 program",
        "webinar 2024 agenda",
        "webinar 2024 harmonogram",
        "webinar 2024 rozkład",
        "webinar 2024 grafik",
        "koncert 2024 program",
        "koncert 2024 agenda",
        "koncert 2024 harmonogram",
        "koncert 2024 rozkład",
        "koncert 2024 grafik",
        "szczyt 2024 program",
        "szczyt 2024 agenda",
        "szczyt 2024 harmonogram",
        "szczyt 2024 rozkład",
        "szczyt 2024 grafik",
        "warsztat 2024 program",
        "warsztat 2024 agenda",
        "warsztat 2024 harmonogram",
        "warsztat 2024 rozkład",
        "warsztat 2024 grafik",
        "sympozjum 2024 program",
        "sympozjum 2024 agenda",
        "sympozjum 2024 harmonogram",
        "sympozjum 2024 rozkład",
        "sympozjum 2024 grafik",
        "expo 2024 program",
        "expo 2024 agenda",
        "expo 2024 harmonogram",
        "expo 2024 rozkład",
        "expo 2024 grafik",
        "forum 2024 program",
        "forum 2024 agenda",
        "forum 2024 harmonogram",
        "forum 2024 rozkład",
        "forum 2024 grafik"
    ]

    all_links = []
    for query in search_queries:
        print(f"Searching for: {query}")
        html = google_search(query)
        
        # Debug: Wydrukuj część HTML, jeśli jest pusty
        if html:
            print("HTML content retrieved, length:", len(html))
        else:
            print("Failed to retrieve HTML content.")
        
        links = extract_links(html)
        all_links.extend(links)
        time.sleep(60)  # Aby uniknąć blokady IP

    # Usuń duplikaty
    unique_links = list(set(all_links))

    # Zapisz linki do pliku
    with open("event_program_links.txt", "w") as file:
        for link in unique_links:
            file.write(f"{link}\n")

    print(f"Found {len(unique_links)} unique links.")

    return unique_links

if __name__ == "__main__":
    links = main()
    print("Found links:", links)
    
    
import requests
from bs4 import BeautifulSoup
import time
import os

def google_search(query, start=0):
    # Przygotuj zapytanie do wyszukiwarki Google z parametrem start
    search_url = f"https://www.google.com/search?q={query}&num=100&start={start}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, headers=headers)
    
    # Sprawdź status odpowiedzi
    if response.status_code != 200:
        print(f"Request failed with status code: {response.status_code}")
        return None
    
    return response.text

    


def extract_links(html):
    # Sprawdź, czy HTML nie jest pusty
    if html is None:
        print("HTML is None, returning empty list.")
        return []
    
    # Przetwarzanie HTML i wyciąganie linków
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    keywords = ["program", "agenda", "schedule", "timetable", "harmonogram", "rozkład", "grafik"]
    
    print("Starting to extract links...")
    for item in soup.find_all('a'):
        href = item.get('href')
        if href and 'http' in href:
            # Zapisz linki zawierające słowa kluczowe
            if any(keyword in href.lower() for keyword in keywords):
                links.append(href)
                print(f"Link added: {href}")
    
    # Debug: Wydrukuj liczbę znalezionych linków
    print(f"Found {len(links)} links after processing.")
    
    return links

def download_pdf(link, save_directory):
    # Sprawdź, czy link prowadzi do pliku PDF
    if link.endswith('.pdf'):
        try:
            response = requests.get(link)
            response.raise_for_status()
            filename = os.path.join(save_directory, os.path.basename(link))
            
            # Zapisz plik PDF
            with open(filename, 'wb') as pdf_file:
                pdf_file.write(response.content)
            
            print(f"Downloaded PDF: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {link}: {e}")
    else:
        print(f"Link is not a PDF: {link}")

def main():
    years = range(2020, 2025)  # Lata od 2020 do 2024
    base_queries = [  # Podstawowe zapytania bez roku
        "conference program",
        "conference agenda",
        "conference schedule",
        "conference timetable",
        "festival program",
        "festival agenda",
        "festival schedule",
        "festival timetable",
        "webinar program",
        "webinar agenda",
        "webinar schedule",
        "webinar timetable",
        "concert program",
        "concert agenda",
        "concert schedule",
        "concert timetable",
        "summit program",
        "summit agenda",
        "summit schedule",
        "summit timetable",
        "workshop program",
        "workshop agenda",
        "workshop schedule",
        "workshop timetable",
        "symposium program",
        "symposium agenda",
        "symposium schedule",
        "symposium timetable",
        "expo program",
        "expo agenda",
        "expo schedule",
        "expo timetable",
        "forum program",
        "forum agenda",
        "forum schedule",
        "forum timetable",

        # Polskie wersje
        "konferencja program",
        "konferencja agenda",
        "konferencja harmonogram",
        "konferencja rozkład",
        "konferencja grafik",
        "festiwal program",
        "festiwal agenda",
        "festiwal harmonogram",
        "festiwal rozkład",
        "festiwal grafik",
        "webinar program",
        "webinar agenda",
        "webinar harmonogram",
        "webinar rozkład",
        "webinar grafik",
        "koncert program",
        "koncert agenda",
        "koncert harmonogram",
        "koncert rozkład",
        "koncert grafik",
        "szczyt program",
        "szczyt agenda",
        "szczyt harmonogram",
        "szczyt rozkład",
        "szczyt grafik",
        "warsztat program",
        "warsztat agenda",
        "warsztat harmonogram",
        "warsztat rozkład",
        "warsztat grafik",
        "sympozjum program",
        "sympozjum agenda",
        "sympozjum harmonogram",
        "sympozjum rozkład",
        "sympozjum grafik",
        "expo program",
        "expo agenda",
        "expo harmonogram",
        "expo rozkład",
        "expo grafik",
        "forum program",
        "forum agenda",
        "forum harmonogram",
        "forum rozkład",
        "forum grafik"
    ]

    save_directory = "downloaded_pdfs"
    os.makedirs(save_directory, exist_ok=True)

    all_links = []
    
    # Tworzenie zapytań dla różnych lat
    for year in years:
        search_queries = [f"{query} {year}" for query in base_queries]
        for query in search_queries:
            for start in range(0, 200, 100):  # Stronicowanie, aby uzyskać do 300 wyników (z 3 stron)
                print(f"Searching for: {query} (start={start})")
                html = google_search(query, start=start)
                
                if html:
                    print("HTML content retrieved, length:", len(html))
                else:
                    print("Failed to retrieve HTML content.")
                
                links = extract_links(html)
                all_links.extend(links)
                
                time.sleep(55)  # Aby uniknąć blokady IP

    # Usuń duplikaty
    unique_links = list(set(all_links))

    # Zapisz linki do pliku
    with open("event_program_links.txt", "w") as file:
        for link in unique_links:
            file.write(f"{link}\n")

    print(f"Found {len(unique_links)} unique links.")
    
    # Pobierz wszystkie pliki PDF
    for link in unique_links:
        download_pdf(link, save_directory)

    return unique_links

if __name__ == "__main__":
    links = main()
    print("Found links:", links)




