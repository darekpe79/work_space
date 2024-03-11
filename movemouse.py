# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 06:52:47 2024

@author: dariu
"""
import pyautogui
import time

try:
    while True:
        # Przesuń w górę
        pyautogui.move(0, -100)  # Przesuwa kursor o 100 pikseli w górę
        time.sleep(0.5)  # Czekaj pół sekundy
        
        # Przesuń w dół
        pyautogui.move(0, 100)  # Przesuwa kursor o 100 pikseli w dół
        time.sleep(0.5)  # Czekaj pół sekundy
except KeyboardInterrupt:
    print("Program zakończony przez użytkownika.")

