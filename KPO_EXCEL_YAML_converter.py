import pandas as pd
import yaml
from collections import defaultdict, Counter
from datetime import datetime

# Wczytanie danych
df = pd.read_excel(
    "D:/Nowa_praca/KPO/Excel_do_YAML/b3272a7f6c9846ee1c5f17074e533ea2.xlsx",
    sheet_name="Arkusz1"
)

# Główna struktura wydarzenia
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

subevents = defaultdict(lambda: {
    "performers": [],
    "creative_work_titles": [],
    "creative_work_authors": [],
    "topic_persons": [],
})
place_counter = Counter()
organizer_counter = Counter()
dates = {}

# Pomocnicza funkcja do parsowania daty
def try_parse_date(value):
    if not value or str(value).strip() == "":
        return None
    value = str(value).strip()

    # Jeśli Excelowa data z godziną 00:00:00, obetnij to
    if value.endswith(" 00:00:00"):
        value = value.replace(" 00:00:00", "")

    formats = ["%Y-%m-%d", "%d.%m.%Y", "%d.%m", "%Y/%m/%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%d.%m":
                return value
            return dt.strftime("%Y-%m-%d")  # bez godziny!
        except:
            continue
    return value


# Przetwarzanie wierszy
for _, row in df.iterrows():
    key = str(row['klucz']).strip()
    sub_id = int(row['numer dla subeventu']) if not pd.isna(row['numer dla subeventu']) else None
    value = str(row['wartość']).strip() if not pd.isna(row['wartość']) else None

    # Wydarzenie główne
    if key.startswith("wydarzenie -"):
        attr = key.replace("wydarzenie - ", "")
        if attr == "tytuł":
            main_event["name"] = value
        elif attr == "opis":
            main_event["description"] = value
        elif attr == "miejsce" and value != "[automatycznie]":
            main_event["location"] = value
        elif attr == "data start":
            parsed = try_parse_date(value)
            if parsed:
                main_event["start_date"] = parsed
        elif attr == "data stop":
            parsed = try_parse_date(value)
            if parsed:
                main_event["end_date"] = parsed
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
            dates[sub_id] = try_parse_date(value)
        elif attr == "godzina start":
            date = dates.get(sub_id)
            if date and value:
                sub["start_date"] = f"{date}T{value}"
            elif date:
                sub["start_date"] = date
        elif attr == "godzina stop":
            date = dates.get(sub_id)
            if date and value:
                sub["end_date"] = f"{date}T{value}"
            elif date:
                sub["end_date"] = date
        elif attr == "osoba - uczestnik" and value:
            sub["performers"].append(value)
        elif attr == "osoba - temat" and value:
            sub["topic_persons"].append(value)
        elif attr == "dzieło - tytuł" and value:
            sub["creative_work_titles"].append(value)
        elif attr == "dzieło - osoba (twórca/współtwórca)" and value:
            sub["creative_work_authors"].append(value)

# Domyślne lokalizacje i organizatorzy
if not main_event.get("location"):
    places = list(dict.fromkeys(place_counter))
    if places:
        main_event["location"] = places if len(places) > 1 else places[0]

if not main_event.get("organizers"):
    orgs = list(dict.fromkeys(organizer_counter))
    if orgs:
        main_event["organizers"] = orgs

# Budowanie YAML-a
for sub in subevents.values():
    sub_yaml = {}

    for field in ["title", "description", "start_date", "end_date", "location", "type_of_event"]:
        val = sub.get(field)
        if val:
            sub_yaml[field] = val

    if sub.get("performers"):
        sub_yaml["performers"] = sub["performers"]

    if sub.get("topic_persons"):
        sub_yaml["topic_persons"] = sub["topic_persons"]

    if sub["creative_work_titles"]:
        works = []
        for title in sub["creative_work_titles"]:
            if not title:
                continue
            work = {"title": title}
            if sub["creative_work_authors"]:
                work["creators"] = sub["creative_work_authors"]
            works.append(work)
        if works:
            sub_yaml["list_of_creative_works"] = works

    if sub_yaml:
        main_event["subevents"].append(sub_yaml)

# Usunięcie pustych
main_event = {k: v for k, v in main_event.items() if v and v != []}

# Zapis do pliku YAML
with open("b3272a7f6c9846ee1c5f17074e533ea2.yaml", "w", encoding="utf-8") as f:
    yaml.dump(main_event, f, allow_unicode=True, sort_keys=False)
