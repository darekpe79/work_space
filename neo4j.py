# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:22:55 2025

@author: darek
"""

from neo4j import GraphDatabase, basic_auth

# 1. Dane dostępowe do Twojej lokalnej bazy
URI      = "bolt://localhost:7687"
USER     = "neo4j"
PASSWORD = "neo4j-demo123"   # <-- tutaj wpisz swoje hasło

# ── 2. Inicjalizacja drivera ────────────────────────────────────────────────────
driver = GraphDatabase.driver(URI, auth=basic_auth(USER, PASSWORD))

# ── 3. Otwieramy sesję ───────────────────────────────────────────────────────────
session = driver.session()

# ── 4. Czyścimy bazę (jeśli potrzebujesz) ───────────────────────────────────────
session.run("MATCH (n) DETACH DELETE n")
print("[INFO] Baza została wyczyszczona.")

# ── 5. Tworzymy przykładowe węzły i relacje ────────────────────────────────────
create_cypher = """
CREATE (a:Person {name: 'Alice', age: 30})
CREATE (b:Person {name: 'Bob', age: 25})
CREATE (m:Movie  {title: 'Matrix', released: 1999})
CREATE (a)-[:WATCHED {rating: 9}]->(m)
"""
session.run(create_cypher)
print("[INFO] Przykładowe dane zostały utworzone.")
session.run("""
CREATE (u:Person {name: 'Ewa', age: 28})
CREATE (f:Movie  {title: 'Inception', released: 2010})
CREATE (u)-[:WATCHED {rating: 8}]->(f)
""")

session.run("CREATE (u:Person {name: 'Adam', age: 15})")

session.run("CREATE (m:Movie {title: 'proba', released: 1999})")

session.run("CREATE (m:Movie {title: 'proba1', released: 1999})")

# ── 6. Wykonujemy prostą kwerendę i wypisujemy wynik ────────────────────────────
query_cypher = """
MATCH (p:Person)-[r:WATCHED]->(m:Movie {title: 'Inception'})
RETURN p.name AS personName, r.rating AS rating
"""
result = session.run(query_cypher)
print("[RESULT] Osoby, które oglądały 'Matrix':")
for record in result:
    print(f" - {record['personName']} (rating: {record['rating']})")

# ── 7. Zamykanie sesji i drivera ────────────────────────────────────────────────
session.close()
driver.close()
print("[INFO] Połączenie zamknięte.")


import pandas as pd
from neo4j import GraphDatabase, basic_auth

# 1. Wczytanie Excela
df = pd.read_excel('D:/Nowa_praca/neo4j_czarek/try_it.xlsx')

# 2. Parametry połączenia
URI      = "bolt://localhost:7687"
USER     = "neo4j"
PASSWORD = "neo4j-demo123"

driver = GraphDatabase.driver(URI, auth=basic_auth(USER, PASSWORD))

# 3. Funkcja importująca
def import_data(tx, row):
    # 3a. MERGE autora po jego unikalnym ID
    tx.run("""
        MERGE (a:Author {id: $id})
        ON CREATE SET a.name = $name,
                      a.birthdate = $birthdate,
                      a.deathdate = $deathdate,
                      a.viaf = $viaf_uri,
                      a.wikidata = $wikidata_uri,
                      a.sex = $sex,
                      a.occupation = $occupation,
                      a.historicalBackground = $hist
    """, {
        "id": row.author_id,
        "name": row.searchName,
        "birthdate": str(row.birthdate) if pd.notna(row.birthdate) else None,
        "deathdate": str(row.deathdate) if pd.notna(row.deathdate) else None,
        "viaf_uri": row.viaf_uri if pd.notna(row.viaf_uri) else None,
        "wikidata_uri": row.wikidata_uri if pd.notna(row.wikidata_uri) else None,
        "sex": row.sex,
        "occupation": row.occupation,
        "hist": row['historical background']
    })

    # 3b. Relacje do nagród (jeśli są)
    if pd.notna(row.prize_id):
        prize_ids = str(row.prize_id).split('|')
        for pid in prize_ids:
            # Tworzymy węzeł Prize i relację
            tx.run("""
                MERGE (p:Prize {id: $pid})
                MERGE (a:Author {id: $aid})
                MERGE (a)-[:RECEIVED]->(p)
            """, {"pid": pid, "aid": row.author_id})

# 4. Uruchomienie importu
with driver.session() as session:
    # (opcjonalnie) wyczyść starą bazę
    session.run("MATCH (n) DETACH DELETE n")
    for _, row in df.iterrows():
        session.write_transaction(import_data, row)

driver.close()
print("Import zakończony.")
query = """
MATCH (a1:Author), (a2:Author)
WHERE a1.id < a2.id AND a1.historicalBackground = a2.historicalBackground
RETURN a1.name AS Author1, a2.name AS Author2, a1.historicalBackground AS SharedBackground
ORDER BY SharedBackground
"""
with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) AS total_nodes")
    print("Liczba węzłów:", result.single()["total_nodes"])
    result = session.run(query)
    for record in result:
        print(f"{record['Author1']} 🧠 {record['Author2']} (epoka: {record['SharedBackground']})")
        
results = []

with driver.session() as session:
    result = session.run(query)
    for record in result:
        results.append({
            "author1": record["Author1"],
            "author2": record["Author2"],
            "shared_background": record["SharedBackground"]
        })

driver.close()


import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from neo4j import GraphDatabase, basic_auth

# 1. Wczytanie Excela


# 2. Parametry połączenia
URI      = "bolt://localhost:7687"
USER     = "neo4j"
PASSWORD = "neo4j-demo123"

driver = GraphDatabase.driver(URI, auth=basic_auth(USER, PASSWORD))
# ——— 1. Konfiguracja modelu Stable Cypher Instruct ——————————————
MODEL_HF = "ragraph-ai/stable-cypher-instruct-3b"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# 1a) HF Transformers
tokenizer = AutoTokenizer.from_pretrained(MODEL_HF, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_HF,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto"
)
model.generation_config.update(
    max_new_tokens=128,
    do_sample=True,
    top_p=0.9,
    temperature=0.2,
)

# (opcjonalnie) lub zamiast powyższego bloku, użyj llama_cpp GGUF:
# from llama_cpp import Llama
# model = Llama(model_path="stable-cypher-instruct-3b.Q4_K_M.gguf", n_ctx=512, n_gpu_layers=-1, verbose=False)

# ——— 2. Schemat bazy do promptu —————————————————————————————————————
schema = """
You are a Cypher assistant. Generate a Cypher query based on this schema:

Node labels:
 - Author {id, name, birthdate, deathdate, viaf, wikidata, sex, occupation, historicalBackground}
 - Prize  {id}

Relationships:
 - (Author)-[:RECEIVED]->(Prize)

Output ONLY the Cypher query, nothing else.
"""

# ——— 3. Zadaj pytanie ————————————————————————————————————————————
question = "Show authors who received more than one prize."
question = (
    "List pairs of authors who share the same historicalBackground. "
    "Return the two names and their shared background."
)
schema = """
You are a Cypher query generator. Given a natural language request and
the database schema, you must output a single valid Cypher query that
performs exactly what was asked—nothing more, nothing less.

Database schema:
  • Node labels:
    - Author {id, name, birthdate, deathdate, viaf, wikidata, sex, occupation, historicalBackground}
    - Prize  {id}
  • Relationships:
    - (Author)-[:RECEIVED]->(Prize)

Examples:
# 1) Natural language → Cypher
# “Show all movies released after 2000.”
MATCH (m:Movie) 
WHERE m.released > 2000 
RETURN m;

# 2) “Show authors who received more than one prize.”
MATCH (a:Author)-[:RECEIVED]->(p:Prize)
WITH a, count(p) AS prizeCount
WHERE prizeCount > 1
RETURN a.id AS authorId, a.name AS authorName, prizeCount AS numPrizes
ORDER BY prizeCount DESC;

# 3) “List pairs of authors who share the same historicalBackground.”
MATCH (a1:Author), (a2:Author)
WHERE a1.id < a2.id 
  AND a1.historicalBackground = a2.historicalBackground
RETURN 
  a1.name AS Author1, 
  a2.name AS Author2, 
  a1.historicalBackground AS SharedBackground
ORDER BY SharedBackground;

Now, generate a Cypher query for the user’s request below.
"""

# ——— 2. Natural language question —————————————————————————————————
question = "List authors grouped by their historicalBackground, showing only those backgrounds shared by more than one author, and include the list of author names for each."

# ——— 3. Kompletujemy prompt ——————————————————————————————————————
prompt = schema + "\nRequest: " + question + "\nCypher:"
prompt = schema + "\nQuestion: " + question + "\nCypher:"

# ——— 4. Generacja Cypher ——————————————————————————————————————————
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

# wyciągamy fragment po "Cypher:"
cypher_query = generated.split("Cypher:")[-1].strip()
print(">>> Generated Cypher:\n", cypher_query, "\n")
results = []

results = []
with driver.session() as session:
    result = session.run(cypher_query)
    for record in result:
        print(record)
        results.append({
            "authorId":   record["author"],
            "authorName": record["authorName"],
            "numPrizes":  record["numPrizes"]
        })
df = pd.DataFrame(results)
print(df)           # zobaczysz tabelę w konsoli
# W Spyderze możesz też po prostu wpisać `df` i w panelu zmiennych kliknąć dwukrotnie, by zobaczyć siatkę.
import matplotlib.pyplot as plt
# ——— 3. Wykres słupkowy —————————————————————————————
plt.figure()
plt.bar(df["authorName"], df["numPrizes"])
plt.xticks(rotation=90)
plt.xlabel("Author")
plt.ylabel("Number of Prizes")
plt.title("Number of Prizes per Author")
plt.tight_layout()
plt.show()

