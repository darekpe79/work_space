import requests
from typing import Optional, Dict, Any, List


def _autosuggest_once(q: str, timeout: float) -> Optional[List[Dict[str, Any]]]:
    """Jedno wywołanie API – zwraca listę wyników lub None."""
    url = "https://viaf.org/viaf/AutoSuggest"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, params={"query": q}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("result") if isinstance(data, dict) else None


def viaf_autosuggest(
    query: str,
    *,
    timeout: float = 5.0,
    people_only: bool = True,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """
    Podpowiedzi VIAF z (domyślnie) filtrem do rekordów osobowych.

    Strategia zapytań (w tej kolejności):
      1. oryginalne hasło,
      2. „Nazwisko, Imię” (zakładamy nazwisko=1-szy wyraz),
      3. „Nazwisko, Imię” (zakładamy nazwisko=ostatni wyraz),
      4. wersja bez przecinka, gdy oryginał go miał
    """
    tried: set[str] = set()
    hits: List[Dict[str, Any]] = []

    def collect(q: str) -> None:
        if q in tried:
            return
        tried.add(q)
        res = _autosuggest_once(q, timeout)
        if res:
            hits.extend(res)

    # 1) oryginalne
    collect(query)

    # 2–3) dwa warianty z przecinkiem, jeśli go brak
    if "," not in query and " " in query:
        parts = query.split()
        if len(parts) >= 2:
            # a) nazwisko = pierwszy wyraz
            collect(f"{parts[0].rstrip(',')}, {' '.join(parts[1:])}")
            # b) nazwisko = ostatni wyraz
            collect(f"{parts[-1].rstrip(',')}, {' '.join(parts[:-1])}")

    # 4) wariant bez przecinka, jeśli oryginał go miał
    if "," in query:
        collect(query.replace(",", ""))

    # ---- filtr personal ---------------------------------------------------
    results = hits
    if people_only:
        personal = [h for h in results
                    if h.get("nametype", "").lower() == "personal"]
        if personal:          # bierzemy tylko gdy coś jest
            results = personal

    # ---- deduplikacja -----------------------------------------------------
    if deduplicate:
        uniq, seen = [], set()
        for h in results:
            fid = h.get("viafid")
            if fid not in seen:
                uniq.append(h)
                seen.add(fid)
        results = uniq

    return {"query": query, "result": results}



viaf=viaf_autosuggest('Henryk Sienkiewicz')
