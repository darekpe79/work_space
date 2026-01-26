from playwright.sync_api import sync_playwright
import time
import random
import os
from datetime import datetime

def random_delay(min_sec=1, max_sec=3):
    """Dodaje losowe opóźnienie"""
    time.sleep(random.uniform(min_sec, max_sec))

def human_like_typing(page, selector, text):
    """Symuluje ludzkie pisanie bez zbędnych ruchów myszą"""
    # Znajdź element
    element = page.locator(selector).first
    
    # Kliknij
    element.click()
    random_delay(0.5, 1)
    
    # Wpisz tekst z naturalnymi przerwami
    for char in text:
        delay = random.uniform(50, 150)  # ms
        # Czasami dłuższa przerwa
        if random.random() < 0.03:  # 3% szans
            delay = random.uniform(300, 800)
        page.keyboard.type(char, delay=delay)
    
    random_delay(0.5, 1)

def search_google_simple(page, query):
    """Proste wyszukiwanie bez zbędnych akcji"""
    print(f"🔍 Szukam: '{query}'")
    
    # Opcja 1: Bezpośredni URL (najmniej wykrywalny!)
    search_url = "https://www.google.com/search?"
    params = {
        'q': query,
        'hl': 'pl',
        'gl': 'PL',
        'lr': 'lang_pl',
        'num': '10',
        'safe': 'active',
        'pws': '0',
        'nfpr': '1',
        'complete': '0',
    }
    
    from urllib.parse import urlencode
    full_url = search_url + urlencode(params)
    
    # Przejdź bezpośrednio do wyników
    print(f"🌐 Przechodzę bezpośrednio do wyników...")
    page.goto(full_url, wait_until="domcontentloaded")
    random_delay(3, 5)
    
    return page

def check_and_handle_cookies(page):
    """Sprawdza czy są ciasteczka i obsługuje je mądrze"""
    content = page.content().lower()
    
    # Sprawdź czy w ogóle jest modal z ciasteczkami
    if any(phrase in content for phrase in ['ciasteczka', 'cookies', 'zaakceptuj', 'accept']):
        print("🍪 Wykryto komunikat o ciasteczkach")
        
        # OPCJA 1: Zignoruj - przewiń w dół i kliknij gdzieś indziej
        print("   → Ignoruję i przewijam w dół...")
        page.evaluate("window.scrollTo(0, 300)")
        random_delay(1, 2)
        
        # Kliknij gdzieś poza modalem (w puste miejsce)
        page.mouse.click(50, 50)
        random_delay(1, 2)
        
        # OPCJA 2: Jeśli trzeba, użyj klawisza ESC
        page.keyboard.press('Escape')
        random_delay(1, 2)
        
        return True
    return False

# Główny kod
def main():
    # Utwórz folder na wyniki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"google_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Wyniki: {os.path.abspath(results_dir)}")
    
    with sync_playwright() as p:
        # PROSTA konfiguracja - im mniej argów, tym lepiej
        browser = p.chromium.launch(
            headless=False,  # Widoczna przeglądarka
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )
        
        # Standardowy user-agent - nie zmieniaj za często
        context = browser.new_context(
            viewport={'width': 1366, 'height': 768},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            locale='pl-PL',
            timezone_id='Europe/Warsaw',
        )
        
        page = context.new_page()
        
        print("="*60)
        print("🚀 Rozpoczynam wyszukiwanie (BEZ akceptacji ciasteczek)")
        print("="*60)
        
        # TRY #1: Bezpośrednie wyszukiwanie
        try:
            query = "python programming tutorial"
            page = search_google_simple(page, query)
            
            # Sprawdź ciasteczka (ale nie akceptuj!)
            check_and_handle_cookies(page)
            
            # Sprawdź czy nie ma captchy
            content = page.content()
            
            if any(word in content.lower() for word in ['captcha', 'żeby pokazać', 'robot']):
                print("\n❌ Wykryto captcha w pierwszej próbie")
                raise Exception("Captcha detected")
            
            # SUKCES!
            print(f"\n✅ SUKCES! Wyniki załadowane")
            print(f"📄 Tytuł: {page.title()}")
            print(f"🌐 URL: {page.url}")
            
            # Zapisz wyniki
            page.screenshot(path=f'{results_dir}/success.png', full_page=True)
            
            with open(f'{results_dir}/results.html', 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Pokaż fragment
            print(f"\n📄 Fragment HTML (500 znaków):")
            print(content[:500])
            
        except Exception as e:
            print(f"\n⚠️  Błąd: {e}")
            print("\n🔄 Próbuję metody alternatywnej...")
            
            # TRY #2: Użyj DuckDuckGo (bez ciasteczek i captchy)
            try:
                print("🌐 Przechodzę na DuckDuckGo...")
                page.goto(f"https://duckduckgo.com/?q=python+programming+tutorial&kl=pl-pl", 
                         wait_until="domcontentloaded")
                random_delay(3, 5)
                
                print(f"✅ DuckDuckGo załadowany")
                print(f"📄 Tytuł: {page.title()}")
                
                # Zapisz
                page.screenshot(path=f'{results_dir}/duckduckgo.png', full_page=True)
                content = page.content()
                with open(f'{results_dir}/duckduckgo_results.html', 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            except Exception as e2:
                print(f"❌ DuckDuckGo też nie działa: {e2}")
        
        finally:
            # Zamknij
            random_delay(2, 3)
            browser.close()
    
    print("\n" + "="*60)
    print("📋 PODSUMOWANIE:")
    print("="*60)
    print("""
KLUCZOWE ZASADY unikania captchy:
1. NIE AKCEPTUJ CIASTECZEK - to często triggeruje captcha
2. Używaj BEZPOŚREDNIEGO URL z parametrami
3. Ogranicz ruchy myszą i klikanie
4. Jedno zapytanie na sesję
5. Używaj standardowego user-agenta
6. Jeśli Google blokuje, użyj DuckDuckGo

Pliki zapisane w: {results_dir}/
    """.format(results_dir=results_dir))

if __name__ == "__main__":
    main()