# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 08:57:50 2023

@author: dariu
"""

class Animal:
    def __init__(self, name):
        self.name = name

    def sound(self):
        return "Makes a sound"

# Klasa pochodna (dziecko)
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Wywołanie konstruktora klasy nadrzędnej
        self.breed = breed

    def sound(self):
        original_sound = super().sound()  # Wywołanie metody z klasy nadrzędnej
        return f"{original_sound}, but as a dog it barks!"

# Użycie klasy pochodnej
dog = Dog("Buddy", "Golden Retriever")
print(dog.sound())  

class Rodzic:
    def __init__(self,name):
        self.name=name
    def przodek(self):
        return "jest ojcem"

class Syn(Rodzic):
    def __init__(self, imie_syna):
        self.imie_syna=imie_syna
        
    def ident(self):
        rodzic=super().sound()
        return f"imię {imie_syna}, rodzic {name}"
    
class Rodzic:
    def __init__(self, name):
        self.name = name

    def przodek(self):
        return "jest ojcem"

class Syn(Rodzic):
    def __init__(self, imie_syna, name):
        super().__init__(name)  # Wywołanie konstruktora klasy bazowej
        self.imie_syna = imie_syna
        
    def ident(self):
        rodzic_info = super().przodek()
        return f"imię {self.imie_syna}, rodzic {self.name}, który {rodzic_info}"

# Przykładowe użycie
syn = Syn("Jan", "Adam")
print(syn.ident())
class Rodzic:
    def __init__(self, nazwisko):
        self.nazwisko = nazwisko

class Dziecko(Rodzic):
    def __init__(self, imie, nazwisko):
        Rodzic.__init__(self, nazwisko) #super().__init__(nazwisko) #my korzystamy bezpośrednie wywołanie konstruktora klasy bazowej
        self.imie = imie

# Użycie
dziecko = Dziecko("Jan", "Kowalski")
print(f"Imię: {dziecko.imie}, Nazwisko: {dziecko.nazwisko}")


class Osoba:
    def __init__(self, imie, nazwisko):
        self.imie = imie
        self.nazwisko = nazwisko

    def przedstaw_sie(self):
        return f"Nazywam się {self.imie} {self.nazwisko}."

class Pracownik(Osoba):
    def __init__(self, imie, nazwisko, stanowisko):
        super().__init__(imie, nazwisko)
        self.stanowisko = stanowisko

    def przedstaw_sie(self):
        # Rozszerzanie metody z klasy bazowej
        podstawowe_przedstawienie = super().przedstaw_sie()
        return f"{podstawowe_przedstawienie} Jestem {self.stanowisko}."

# Użycie
osoba = Osoba("Jan", "Kowalski")
pracownik = Pracownik("Anna", "Nowak", "inżynier")

print(osoba.przedstaw_sie())  # Wykorzystanie metody z klasy bazowej
print(pracownik.przedstaw_sie())  # Wykorzystanie rozszerzonej metody z klasy pochodnej

class Samochod:
    def __init__(self, marka, model, rok_produkcji):
        self.marka = marka
        self.model = model
        self.rok_produkcji = rok_produkcji
        self.przebieg = 0  # wartość domyślna

    def pokaz_info(self):
        return f"Samochód: {self.marka} {self.model}, Rok produkcji: {self.rok_produkcji}, Przebieg: {self.przebieg} km"

    def aktualizuj_przebieg(self, km):
        if km >= self.przebieg:
            self.przebieg = km
        else:
            print("Nie można cofnąć licznika przebiegu!")
            
    def pokaz_przebieg(self):
        return f"Przebieg samochodu: {self.przebieg} km"
    
    
    
    


    
moj_samochod = Samochod("Toyota", "Corolla", 2018)
moj_samochod.aktualizuj_przebieg(15000)
print(moj_samochod.pokaz_przebieg())


class SamochodElektryczny(Samochod):
    def __init__(self, marka, model, rok_produkcji, pojemnosc_baterii):
        super().__init__(marka, model, rok_produkcji)
        self.pojemnosc_baterii = pojemnosc_baterii

    def pokaz_zasieg(self):
        return f"Zasięg samochodu: {self.pojemnosc_baterii * 5} km"
    
    
class A:
    def __init__(self):
        super().__init__()
        print("Konstruktor klasy A")

class B:
    def __init__(self):
        super().__init__()
        print("Konstruktor klasy B")

class C:
    def __init__(self):
        super().__init__()
        print("Konstruktor klasy C")

class D(A, B, C):
    def __init__(self):
        super().__init__()  # Automatyczne zarządzanie kolejnością wywołań konstruktorów klas bazowych
        print("Konstruktor klasy D")

# Tworzenie obiektu klasy D
obj = D()
print(D.mro())

class Samochod:
    def __init__(self, silnik, kolor):
        self.silnik = silnik
        self.kolor = kolor

    def pokaz_dane(self):
        return f"Samochód z silnikiem: {self.silnik.typ} i kolorem: {self.kolor.nazwa}"
class Silnik:
    def __init__(self, typ):
        self.typ = typ  # np. 'diesel', 'benzyna'

class Kolor:
    def __init__(self, nazwa):
        self.nazwa = nazwa  # np. 'czerwony', 'niebieski'

# Tworzymy instancje klas atrybutów
moj_silnik = Silnik('diesel')
moj_kolor = Kolor('niebieski')

# Tworzymy instancję klasy Samochod
moj_samochod = Samochod(moj_silnik, moj_kolor)

# Wyświetlamy dane samochodu
print(moj_samochod.pokaz_dane())

sam=Samochod('diesel', 'niebieski')
print(sam.pokaz_dane())




class Silnik:
    def __init__(self, typ):
        self.typ = typ  # np. 'diesel', 'benzyna'

class Kolor:
    def __init__(self, nazwa, odcien):
        self.nazwa = nazwa  # np. 'czerwony', 'niebieski'
        self.odcien = odcien  # np. 'jasny', 'ciemny'

class Samochod:
    def __init__(self, silnik, kolor):
        self.silnik = silnik
        self.kolor = kolor

    def pokaz_dane(self):
        return f"Samochód z silnikiem: {self.silnik.typ}, Kolor: {self.kolor.nazwa}, Odcień: {self.kolor.odcien}"


# Tworzymy instancje klas atrybutów
moj_silnik = Silnik('diesel')
moj_kolor = Kolor('niebieski', 'jasny')

# Tworzymy instancję klasy Samochod
moj_samochod = Samochod(moj_silnik, moj_kolor)

# Wyświetlamy dane samochodu
print(moj_samochod.pokaz_dane())
