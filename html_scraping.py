# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 11:17:21 2025

@author: darek
"""



import httpx

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
}

r = httpx.get("https://quotes.toscrape.com/", headers=headers)
print(r.status_code)


print(r.text)




