# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:54:36 2024

@author: dariu
"""

from pymarc import MARCReader, MARCWriter, Field
from tqdm import tqdm


# Lista plików MARC do przetworzenia
pliki_marc = ["D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/sp_ksiazki_composed_unify2_do_wyslanianew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/bn_articles_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/bn_books_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/bn_chapters_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/cz_articles0_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/cz_articles1_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/cz_articles2_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/cz_articles3_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/cz_articles4_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/cz_books_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/cz_chapters_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/es_articles_sorted_31new_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/fi_arto_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/fi_fennica_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/NEW-marc_bn_articles_2023-08-07new_viafnew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/NEW-marc_bn_books_2023-08-07_processednew_viafnew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/NEW-marc_bn_chapters_2023-08-07_processednew_viafnew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/pbl_articles_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc",
"D:/Nowa_praca/18102023_nowe_marki_z_probka_szkotow/pbl_books_21-02-2023composenew_viafnew_viaf_processednew_viaf.mrc"]

# Tagi do sprawdzenia
tagi_do_sprawdzenia = ['600', '100', '700']

# Pętla przez wszystkie pliki
# Tworzenie obiektu MARCWriter do zapisu wszystkich rekordów

for plik in tqdm(pliki_marc):
    with open(plik, 'rb') as fh:
        with open(plik+'tmp', 'wb') as writer:
            reader = MARCReader(fh)
            for record in tqdm(reader):
                my = record.get_fields('100','600','700')
                for m in my:
                    if m.get_subfields('1'):
                        if m['1']=='http://viaf.org/viaf/311314437' or m['1']=='http://viaf.org/viaf/101777210':
                            if m.get_subfields('d'):
                                m.delete_subfield('d')
                                m.add_subfield('d', '(1951-2017)')
                            print(m['1'])
                            m.delete_subfield('1')
                            m.add_subfield('1', 'http://viaf.org/viaf/101777210')
                            # if m.get_subfields('d'):
                            #     m.delete_subfield('d')
                            #     m.add_subfield('d', '(1951-2017)')
                            
                writer.write(record.as_marc())

from pymarc import MARCReader, TextWriter                
pliki_marc=["D:/Nowa_praca/08.02.2024_marki/pbl_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_articles_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/bn_chapters_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles0_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles1_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles2_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles3_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_articles4_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_books__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/cz_chapters__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/es_articles__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/es_ksiazki__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/fi_arto__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/fi_fennica_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_articles_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_books_08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/NEW-marc_bn_chapters__08-02-2024.mrc",
"D:/Nowa_praca/08.02.2024_marki/pbl_articles_08-02-2024.mrc"]           
for plik in tqdm(pliki_marc):
    # Przygotowanie nazwy pliku wyjściowego, zamieniając rozszerzenie na .mrk
    nazwa_pliku_wyjsciowego = plik.replace('.mrc', '.mrk')
    
    with open(plik, 'rb') as fh, open(nazwa_pliku_wyjsciowego, 'w', encoding='utf-8') as writer_fh:
        reader = MARCReader(fh)
        writer = TextWriter(writer_fh)
        for record in reader:
            # Tutaj możesz dokonywać modyfikacji rekordu
            # Na przykład, używając wcześniej zdefiniowanej funkcji modyfikuj_podpole
            # modyfikuj_podpole(record, tagi_do_sprawdzenia, 'http://viaf.org/viaf/311314437', 'http://viaf.org/viaf/101777210/', '(1951–2017)')
            
            writer.write(record)
