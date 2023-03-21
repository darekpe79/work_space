# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:08:09 2023

@author: dariu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 19:06:39 2021

@author: darek
"""

import requests
url = "https://api-football-beta.p.rapidapi.com/standings" 
querystring = {"season":"2022","league":"107"}
headers = {'x-rapidapi-host': "api-football-beta.p.rapidapi.com", 'x-rapidapi-key': "3e405b6251mshd60581ae73af7bap1e26e1jsn43625ecf6667" }

response = requests.request("GET", url, headers=headers, params=querystring) 
print(response.text)
with open("/home/pi/darek/league_stand.json", "w") as outfile:
    outfile.write(response.text)
    
import requests
import json
url = "https://api-football-beta.p.rapidapi.com/teams/statistics"

querystring = {"team":"3498","season":"2022","league":"107"}

headers = {
    'x-rapidapi-host': "api-football-beta.p.rapidapi.com",
    'x-rapidapi-key': "3e405b6251mshd60581ae73af7bap1e26e1jsn43625ecf6667"
    }

response = requests.request("GET", url, headers=headers, params=querystring)
print(response.text)