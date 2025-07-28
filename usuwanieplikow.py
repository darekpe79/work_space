# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 13:18:29 2025

@author: darek
"""

import os

# Ścieżka do katalogu
folder_path = r'D:/Nowa_praca/pdfy_do zrobienia/output_dhq-20250709T065853Z-1-001/output_dhq/pdfs/'

# Licznik usuniętych plików
deleted_files = 0

# Sprawdź, czy katalog istnieje
if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                deleted_files += 1
        except Exception as e:
            print(f"Nie udało się usunąć: {file_path} — {e}")
else:
    print(f"Katalog nie istnieje: {folder_path}")

print(f"Usunięto {deleted_files} plików.")


import os
import shutil
import getpass

recycle_base = r"C:\$Recycle.Bin"
current_user = getpass.getuser()

deleted_files = 0

# Iteruj po katalogach użytkowników w $Recycle.Bin
for sid in os.listdir(recycle_base):
    sid_path = os.path.join(recycle_base, sid)
    if not os.path.isdir(sid_path):
        continue

    # Iteruj po plikach użytkownika
    for item in os.listdir(sid_path):
        item_path = os.path.join(sid_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
                deleted_files += 1
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
                deleted_files += 1
        except Exception as e:
            print(f"Nie udało się usunąć: {item_path} — {e}")

print(f"Usunięto {deleted_files} plików z kosza.")

import os
import shutil
import string

deleted_files = 0
errors = 0

for drive_letter in string.ascii_uppercase:
    recycle_path = f"{drive_letter}:\\$RECYCLE.BIN"
    if not os.path.exists(recycle_path):
        continue

    try:
        for sid in os.listdir(recycle_path):
            sid_path = os.path.join(recycle_path, sid)
            if not os.path.isdir(sid_path):
                continue

            for item in os.listdir(sid_path):
                item_path = os.path.join(sid_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                        deleted_files += 1
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        deleted_files += 1
                except Exception as e:
                    print(f"❌ Nie udało się usunąć: {item_path} — {e}")
                    errors += 1

    except PermissionError:
        print(f"🔒 Brak dostępu do: {recycle_path} – uruchom jako administrator.")
    except Exception as e:
        print(f"⚠️ Błąd przy przetwarzaniu {recycle_path}: {e}")

print(f"\n✅ Usunięto {deleted_files} plików z koszy na wszystkich dyskach.")
if errors:
    print(f"⚠️ Wystąpiło {errors} błędów (brak dostępu lub zablokowane pliki).")

