# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:06:04 2025

@author: darek
"""

import re
import textwrap
import pandas as pd


# Raw text excerpt again
raw_text = textwrap.dedent("""
Acosta , Iosephus de , S. I., vita 299 ,

manifestat Borgiae suum deside-
rium Indias adeundi , 300-302 ; pro

cleri educatione , 302 ; P. Ludovi-
cum Guzmán ad missiones praesen-
tat, 303 ; revelat P. Nadal volun-
tatem suam ad missi ones indicas et

varia de se ipso , 300 , 301 10; con-
ficit litteras annuas ,

Roma vel Burgos destinatus , 322 ;
Peruae missionarius renuntiatus ,

3897, 390, 39120, 371 ; nuntiat Bor-
giae praeludia itineris , 439-442 ;

quaerit de P. Fonseca , 442 ; propo-
nit ad Sacros Ordines F. D. Mar-
tínez , 442 ; Hispali versatur et San-
lúcar , 440 ; in insula S. Ioannis

et S. Dominici , 443 ; eius iter a

Sacchini narratur , 47s .; Limam at-
tigit, 36 , 505 ; confessarius in col-
legio limensi et magister novitio-
rum , 505 ; concionator Limae , 703 ;

missionarius per regionem de La

Plata 629 3, 709 ; eius missio per Pe-
ruam , 6293,706 ; ad Potosí , 709 ; quae-
stiones morales cum eo conferendae ,

632 ; provincialis designatus Pe-
ruae, 37 ; eius actio in Congrega-
tionibus provincialibus , 38 ; scrip-
tor , 321 ; informationes de eo ,

3019 ; laudatur , 505 , 507 , 509 ,
589 .
""")

# Normalise line breaks – join hyphenated lines, remove newlines inside sentences
clean = re.sub(r'\n+', ' ', raw_text)                # flatten
clean = re.sub(r'\s{2,}', ' ', clean)                # collapse spaces
clean = re.sub(r'-\s', '', clean)                    # remove soft hyphen wraps

# Split by semicolon OR dot that ends a phrase
phrases = re.split(r'[;.]', clean)
phrases = [p.strip() for p in phrases if p.strip()]

records = []
term = "Acosta , Iosephus de"
for ph in phrases:
    # detect trailing pages (digits, ranges, separated by , or space)
    m = re.search(r'(.*?)(\d[\d\-\s,]*)$', ph)
    if not m:
        continue
    text = m.group(1).strip(' ,')
    pages = m.group(2)
    # expand pages list
    page_list = re.findall(r'\d+', pages)
    for p in page_list:
        records.append({'Term': term, 'Note': text, 'Page': int(p)})

df = pd.DataFrame(records)
tools.display_dataframe_to_user("Acosta parsed", df.head(40))
