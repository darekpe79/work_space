# # -*- coding: utf-8 -*-
# """
# Created on Tue Apr  1 12:10:26 2025

# @author: darek
# """



# import pandas as pd
# import yaml
# from collections import defaultdict, Counter

# # Wczytaj dane z Excela
# df = pd.read_excel("D:/Nowa_praca/KPO/YAML/KPO moduł 18 yamle_217cab33b2adf95c7e2afa28b6b496f7.xlsx", sheet_name="walidacja")

# # Inicjalizacja głównej struktury
# main_event = {
#     "name": None,
#     "description": None,
#     "location": None,
#     "start_date": None,
#     "end_date": None,
#     "type_of_event": None,
#     "organizers": [],
#     "list_of_event_days": [
#         {
#             "description": "",  # będzie do uzupełnienia
#             "start_date": None,
#             "end_date": None,
#             "list_of_subevents": []
#         }
#     ]
# }

# # Bufory pomocnicze
# subevents = defaultdict(lambda: {
#     "title": None,
#     "description": None,
#     "performers": [],
#     "attendees": None,
#     "start_date": None,
#     "end_date": None,
#     "list_of_creative_works": None,
#     "location": None,
#     "type_of_event": None,
#     "topic_person": [],
#     "creative_work_titles": [],
#     "creative_work_authors": []
# })
# place_counter = Counter()
# organizer_counter = Counter()

# # Parsowanie wierszy
# for _, row in df.iterrows():
#     key = str(row['klucz']).strip()
#     sub_id = int(row['numer dla subeventu']) if not pd.isna(row['numer dla subeventu']) else None
#     value = str(row['wartość']).strip() if not pd.isna(row['wartość']) else None

#     if key.startswith("wydarzenie -"):
#         attr = key.replace("wydarzenie - ", "").strip()
#         if attr == "tytuł":
#             main_event["name"] = value
#         elif attr == "opis":
#             main_event["description"] = value
#         elif attr == "miejsce":
#             if value != "[automatycznie]":
#                 main_event["location"] = value
#         elif attr == "data start":
#             main_event["start_date"] = value
#         elif attr == "data stop":
#             main_event["end_date"] = value
#         elif attr == "typ wydarzenia":
#             main_event["type_of_event"] = value
#         elif attr == "organizator":
#             if value != "[automatycznie]":
#                 main_event["organizers"].append(value)

#     elif key.startswith("subevent -"):
#         attr = key.replace("subevent - ", "").strip()

#         if attr == "tytuł":
#             subevents[sub_id]["title"] = value
#         elif attr == "opis":
#             subevents[sub_id]["description"] = value
#         elif attr == "miejsce":
#             subevents[sub_id]["location"] = value
#             place_counter[value] += 1
#         elif attr == "typ wydarzenia":
#             subevents[sub_id]["type_of_event"] = value
#         elif attr == "data":
#             date = value
#         elif attr == "godzina start":
#             subevents[sub_id]["start_date"] = f"{date}T{value}"
#         elif attr == "godzina stop":
#             subevents[sub_id]["end_date"] = f"{date}T{value}"
#         elif attr == "osoba - uczestnik":
#             if value:
#                 subevents[sub_id]["performers"].append(value)
#         elif attr == "osoba - temat":
#             if value:
#                 subevents[sub_id]["topic_person"].append(value)
#         elif attr == "dzieło - tytuł":
#             subevents[sub_id]["creative_work_titles"].append(value)
#         elif attr == "dzieło - osoba (twórca/współtwórca)":
#             subevents[sub_id]["creative_work_authors"].append(value)

# # Przypisz najczęstsze automatyczne miejsce i organizatora
# if not main_event["location"]:
#     most_common_loc = place_counter.most_common(1)
#     if most_common_loc:
#         main_event["location"] = most_common_loc[0][0]

# if not main_event["organizers"]:
#     most_common_org = organizer_counter.most_common(1)
#     if most_common_org:
#         main_event["organizers"] = [most_common_org[0][0]]

# # Dodaj subeventy do struktury
# main_event["list_of_event_days"][0]["list_of_subevents"] = []

# for i in sorted(subevents.keys()):
#     sub = subevents[i]
#     cw = None
#     if sub["creative_work_titles"] or sub["creative_work_authors"]:
#         cw = [{
#             "title": title,
#             "creators": sub["creative_work_authors"]
#         } for title in sub["creative_work_titles"]]

#     sub_yaml = {
#         "title": sub["title"],
#         "description": sub["description"],
#         "performers": sub["performers"] if sub["performers"] else None,
#         "attendees": None,
#         "start_date": sub["start_date"],
#         "end_date": sub["end_date"],
#         "list_of_creative_works": cw,
#     }

#     main_event["list_of_event_days"][0]["list_of_subevents"].append(sub_yaml)

# # Zapisz do YAML
# with open("wydarzenie.yaml", "w", encoding="utf-8") as f:
#     yaml.dump(main_event, f, allow_unicode=True, sort_keys=False)
    
    
    
# # -*- coding: utf-8 -*-
# """
# Created on Tue Apr  1 12:10:26 2025

# @author: darek
# """

# import pandas as pd
# import yaml
# from collections import defaultdict, Counter

# # Wczytaj dane z Excela
# df = pd.read_excel(
#     "D:/Nowa_praca/KPO/YAML/KPO moduł 18 yamle_217cab33b2adf95c7e2afa28b6b496f7.xlsx",
#     sheet_name="walidacja"
# )

# # Inicjalizacja głównej struktury
# main_event = {
#     "name": None,
#     "description": None,
#     "location": None,
#     "start_date": None,
#     "end_date": None,
#     "type_of_event": None,
#     "organizers": [],
#     "list_of_event_days": [
#         {
#             "description": "",  # Do uzupełnienia, jeśli potrzeba
#             "start_date": None,
#             "end_date": None,
#             "list_of_subevents": []
#         }
#     ]
# }

# # Bufory pomocnicze
# subevents = defaultdict(lambda: {
#     "title": None,
#     "description": None,
#     "performers": [],
#     "attendees": None,
#     "start_date": None,
#     "end_date": None,
#     "list_of_creative_works": [],
#     "location": None,
#     "type_of_event": None,
#     "topic_person": [],
#     "creative_work_titles": [],
#     "creative_work_authors": []
# })
# place_counter = Counter()
# organizer_counter = Counter()
# dates_by_subevent = {}

# # Parsowanie wierszy
# for _, row in df.iterrows():
#     key = str(row['klucz']).strip()
#     sub_id = int(row['numer dla subeventu']) if not pd.isna(row['numer dla subeventu']) else None
#     value = str(row['wartość']).strip() if not pd.isna(row['wartość']) else None

#     if key.startswith("wydarzenie -"):
#         attr = key.replace("wydarzenie - ", "").strip()
#         if attr == "tytuł":
#             main_event["name"] = value
#         elif attr == "opis":
#             main_event["description"] = value
#         elif attr == "miejsce":
#             if value != "[automatycznie]":
#                 main_event["location"] = value
#             else:
#                 main_event["location"] = None
#         elif attr == "data start":
#             main_event["start_date"] = value
#         elif attr == "data stop":
#             main_event["end_date"] = value
#         elif attr == "typ wydarzenia":
#             main_event["type_of_event"] = value
#         elif attr == "organizator":
#             if value != "[automatycznie]":
#                 main_event["organizers"].append(value)

#     elif key.startswith("subevent -"):
#         attr = key.replace("subevent - ", "").strip()

#         if attr == "tytuł":
#             subevents[sub_id]["title"] = value
#         elif attr == "opis":
#             subevents[sub_id]["description"] = value
#         elif attr == "miejsce":
#             subevents[sub_id]["location"] = value
#             place_counter[value] += 1
#         elif attr == "typ wydarzenia":
#             subevents[sub_id]["type_of_event"] = value
#         elif attr == "data":
#             dates_by_subevent[sub_id] = value
#         elif attr == "godzina start":
#             date = dates_by_subevent.get(sub_id)
#             if date:
#                 subevents[sub_id]["start_date"] = f"{date}T{value}"
#         elif attr == "godzina stop":
#             date = dates_by_subevent.get(sub_id)
#             if date:
#                 subevents[sub_id]["end_date"] = f"{date}T{value}"
#         elif attr == "osoba - uczestnik":
#             if value:
#                 subevents[sub_id]["performers"].append(value)
#         elif attr == "osoba - temat":
#             if value:
#                 subevents[sub_id]["topic_person"].append(value)
#         elif attr == "dzieło - tytuł":
#             if value:
#                 subevents[sub_id]["creative_work_titles"].append(value)
#         elif attr == "dzieło - osoba (twórca/współtwórca)":
#             if value:
#                 subevents[sub_id]["creative_work_authors"].append(value)

# # Uzupełnij automatyczne dane jeśli brak
# if not main_event.get("location"):
#     most_common_loc = place_counter.most_common(1)
#     if most_common_loc:
#         main_event["location"] = most_common_loc[0][0]

# if not main_event.get("organizers"):
#     most_common_org = organizer_counter.most_common(1)
#     if most_common_org:
#         main_event["organizers"] = [most_common_org[0][0]]

# # Dodaj subeventy do głównego wydarzenia
# for i in sorted(subevents.keys()):
#     sub = subevents[i]
#     sub_yaml = {}

#     if sub["title"]: sub_yaml["title"] = sub["title"]
#     if sub["description"]: sub_yaml["description"] = sub["description"]
#     if sub["performers"]: sub_yaml["performers"] = sub["performers"]
#     if sub["start_date"]: sub_yaml["start_date"] = sub["start_date"]
#     if sub["end_date"]: sub_yaml["end_date"] = sub["end_date"]

#     # Tworzenie listy dzieł
#     if sub["creative_work_titles"] or sub["creative_work_authors"]:
#         works = []
#         for title in sub["creative_work_titles"]:
#             work = {"title": title}
#             if sub["creative_work_authors"]:
#                 work["creators"] = sub["creative_work_authors"]
#             works.append(work)
#         sub_yaml["list_of_creative_works"] = works

#     main_event["list_of_event_days"][0]["list_of_subevents"].append(sub_yaml)

# # Usuń puste klucze z main_event (opcjonalne, ale ładniejsze)
# main_event = {k: v for k, v in main_event.items() if v not in [None, [], ""]}

# # Zapisz jako YAML
# with open("wydarzenie.yaml", "w", encoding="utf-8") as f:
#     yaml.dump(main_event, f, allow_unicode=True, sort_keys=False)


# -*- coding: utf-8 -*-
"""
Przekształcanie danych z Excela do YAML – uproszczona wersja (subevents jako osobne wpisy)
@author: darek
"""

import pandas as pd
import yaml
from collections import defaultdict, Counter

# Wczytanie danych
df = pd.read_excel(
    "D:/Nowa_praca/KPO/YAML/KPO moduł 18 yamle_217cab33b2adf95c7e2afa28b6b496f7.xlsx",
    sheet_name="walidacja"
)

# Struktura główna
main_event = {
    "name": None,
    "description": None,
    "location": None,
    "start_date": None,
    "end_date": None,
    "type_of_event": None,
    "organizers": [],
    "subevents": []
}

# Bufory
subevents = defaultdict(lambda: {
    "performers": [],
    "creative_work_titles": [],
    "creative_work_authors": [],
    "topic_persons": []
})
place_counter = Counter()
organizer_counter = Counter()
dates = {}

# Przetwarzanie danych
for _, row in df.iterrows():
    key = str(row['klucz']).strip()
    sub_id = int(row['numer dla subeventu']) if not pd.isna(row['numer dla subeventu']) else None
    value = str(row['wartość']).strip() if not pd.isna(row['wartość']) else None

    # Główne wydarzenie
    if key.startswith("wydarzenie -"):
        attr = key.replace("wydarzenie - ", "")
        if attr == "tytuł":
            main_event["name"] = value
        elif attr == "opis":
            main_event["description"] = value
        elif attr == "miejsce" and value != "[automatycznie]":
            main_event["location"] = value
        elif attr == "data start":
            main_event["start_date"] = value
        elif attr == "data stop":
            main_event["end_date"] = value
        elif attr == "typ wydarzenia":
            main_event["type_of_event"] = value
        elif attr == "organizator" and value != "[automatycznie]":
            organizer_counter[value] += 1
            if value not in main_event["organizers"]:
                main_event["organizers"].append(value)

    # Subeventy
    elif key.startswith("subevent -") and sub_id is not None:
        attr = key.replace("subevent - ", "").strip()
        sub = subevents[sub_id]

        if attr == "tytuł":
            sub["title"] = value
        elif attr == "opis":
            sub["description"] = value
        elif attr == "miejsce":
            sub["location"] = value
            if value:
                place_counter[value] += 1
        elif attr == "typ wydarzenia":
            sub["type_of_event"] = value
        elif attr == "data":
            dates[sub_id] = value
        elif attr == "godzina start":
            sub["start_date"] = f"{dates.get(sub_id, '')}T{value}"
        elif attr == "godzina stop":
            sub["end_date"] = f"{dates.get(sub_id, '')}T{value}"
        elif attr == "osoba - uczestnik" and value:
            sub["performers"].append(value)
        elif attr == "osoba - temat" and value:
            sub["topic_persons"].append(value)
        elif attr == "dzieło - tytuł" and value:
            sub["creative_work_titles"].append(value)
        elif attr == "dzieło - osoba (twórca/współtwórca)" and value:
            sub["creative_work_authors"].append(value)

# Domknięcie braków
if not main_event.get("location"):
    all_places = list(dict.fromkeys([place for place in place_counter if place]))
    if all_places:
        main_event["location"] = all_places if len(all_places) > 1 else all_places[0]


if not main_event.get("organizers"):
    unique_orgs = list(organizer_counter.keys())
    if unique_orgs:
        main_event["organizers"] = unique_orgs

# Budowanie subeventów
for sub in subevents.values():
    sub_yaml = {}

    for field in ["title", "description", "start_date", "end_date", "location", "type_of_event"]:
        if sub.get(field):
            sub_yaml[field] = sub[field]

    if sub.get("performers"):
        sub_yaml["performers"] = sub["performers"]

    if sub.get("topic_persons"):
        sub_yaml["topic_persons"] = sub["topic_persons"]

    if sub["creative_work_titles"]:
        works = []
        for title in sub["creative_work_titles"]:
            work = {}
            if title:
                work["title"] = title
            if sub["creative_work_authors"]:
                work["creators"] = sub["creative_work_authors"]
            if work:
                works.append(work)
        if works:
            sub_yaml["list_of_creative_works"] = works

    main_event["subevents"].append(sub_yaml)

# Zapis do YAML
with open("wydarzenie.yaml", "w", encoding="utf-8") as f:
    yaml.dump(main_event, f, allow_unicode=True, sort_keys=False)



