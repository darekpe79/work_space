

import requests

r = requests.get(
    "https://api.redalyc.org/journal",
    params={"subject_area": "Social Sciences", "from_year": 2024, "per_page": 100},
    timeout=30
)
data = r.json()



from sickle import Sickle

URL = "http://148.215.1.70/redalyc/oai"   # HTTP, nie HTTPS!
UA  = {"User-Agent": "redalyc-harvester/0.1 (dariusz.perlinski@ibl.waw.pl)"}

sickle = Sickle(URL, timeout=60, headers=UA)

# ISSN czasopisma z dziedziny "Lengua y Literatura"
ISSN = "1132-3310"        # Francofonía (Hiszpania) - przykład

records = sickle.ListRecords(
    metadataPrefix="oai_dc",
    set=ISSN,             # <-- tu podajesz ISSN jako setSpec
    
)

for rec in records:
    md = rec.metadata
    print(md)
    print(md["title"][0])
    print(md["identifier"][0])
