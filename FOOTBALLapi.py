# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:59:20 2024

@author: dariu
"""

import requests

url = "https://api-football-v1.p.rapidapi.com/v3/teams"

querystring = {"id":"3498"}

headers = {
	"x-rapidapi-key": "3e405b6251mshd60581ae73af7bap1e26e1jsn43625ecf6667",
	"x-rapidapi-host": "api-football-v1.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())