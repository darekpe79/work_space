
    
    
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
    


    return unique_links

if __name__ == "__main__":
    links = main()
    print("Found links:", links)



import os
import requests
import re

def sanitize_filename(filename):
    # Usunięcie niepoprawnych znaków z nazwy pliku
    return re.sub(r'[<>:"/\\|?*\%]', '_', filename)

def download_pdf(link, save_directory, retries=3):
    # Sprawdź, czy link prowadzi do pliku PDF
    if link.endswith('.pdf'):
        for attempt in range(retries):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(link, headers=headers, timeout=10)
                response.raise_for_status()

                # Usuń niepoprawne znaki z nazwy pliku
                filename = os.path.basename(link)
                filename = sanitize_filename(filename)
                filepath = os.path.join(save_directory, filename)

                # Próbuj zapisać plik PDF
                try:
                    with open(filepath, 'wb') as pdf_file:
                        pdf_file.write(response.content)
                    print(f"Downloaded PDF: {filepath}")
                    break  # Jeśli sukces, przerwij pętlę prób
                except OSError as e:
                    print(f"Failed to save the PDF {filepath}: {e}")
                    break  # Jeśli wystąpił błąd zapisu, nie próbuj ponownie

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed for {link}: {e}")
                if attempt + 1 == retries:
                    print(f"All {retries} attempts failed for {link}. Skipping...")
    else:
        print(f"Link is not a PDF: {link}")

def process_links_from_file(file_path, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    with open(file_path, 'r') as file:
        links = file.readlines()

    for link in links:
        link = link.strip()  # Usuń białe znaki na początku i końcu linku
        download_pdf(link, save_directory)

# Ścieżka do pliku z linkami
file_path = "event_program_links.txt"
# Ścieżka do katalogu, w którym mają być zapisane pliki PDF
save_directory = "downloaded_pdfs"

# Przetwarzanie linków z pliku
process_links_from_file(file_path, save_directory)


