# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:08:42 2024

@author: dariu
"""

import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

# Definiowanie baseuri, idsource oraz przestrzeni nazw dla BIBFRAME
baseuri = "http://example.org/resource/"
idsource = "http://id.loc.gov/vocabulary/organizations/dlc"
BF = Namespace("http://id.loc.gov/ontologies/bibframe/")
BFLc = Namespace("http://id.loc.gov/ontologies/bflc/")
DC = Namespace("http://purl.org/dc/elements/1.1/")

# Funkcja konwertująca wiersz z Dublin Core na BIBFRAME RDF/XML
def dc_to_bibframe(row):
    g = Graph()
    g.bind("bf", BF)
    g.bind("bflc", BFLc)
    g.bind("dc", DC)

    # Tworzenie głównego zasobu Work z użyciem baseuri
    work_uri = URIRef(f"{baseuri}{row.get('Identifier', 'Unknown')}")
    g.add((work_uri, RDF.type, BF.Work))

    # Dodanie identyfikatora zasobu i wskazanie źródła identyfikatora
    if "Identifier" in row and pd.notna(row["Identifier"]):
        identifier_bf = URIRef(f"{work_uri}/identifier")
        g.add((work_uri, BF.identifiedBy, identifier_bf))
        g.add((identifier_bf, RDF.type, BF.Local))
        g.add((identifier_bf, RDFS.label, Literal(row["Identifier"])))
        g.add((identifier_bf, BF.source, URIRef(idsource)))

    # Tytuł (Title)
    if "Title" in row and pd.notna(row["Title"]):
        title_bf = URIRef(f"{work_uri}/title")
        g.add((work_uri, BF.title, title_bf))
        g.add((title_bf, BF.mainTitle, Literal(row["Title"])))

    # Twórca (Creator) jako Contribution (Work)
    if "Creator" in row and pd.notna(row["Creator"]):
        contribution_bf = URIRef(f"{work_uri}/contribution")
        g.add((work_uri, BF.contribution, contribution_bf))
        agent_bf = URIRef(f"{contribution_bf}/agent")
        g.add((contribution_bf, BF.agent, agent_bf))
        g.add((agent_bf, RDF.type, BF.Agent))
        g.add((agent_bf, RDFS.label, Literal(row["Creator"])))

    # Dodatkowy współtwórca (Contributor)
    if "Contributor" in row and pd.notna(row["Contributor"]):
        contributor_bf = URIRef(f"{work_uri}/contributor")
        g.add((work_uri, BF.contribution, contributor_bf))
        agent_contributor_bf = URIRef(f"{contributor_bf}/agent")
        g.add((contributor_bf, BF.agent, agent_contributor_bf))
        g.add((agent_contributor_bf, RDF.type, BF.Agent))
        g.add((agent_contributor_bf, RDFS.label, Literal(row["Contributor"])))

    # Data publikacji (Date)
    if "Date" in row and pd.notna(row["Date"]):
        provision_activity = URIRef(f"{work_uri}/provisionActivity")
        g.add((work_uri, BF.provisionActivity, provision_activity))
        g.add((provision_activity, BF.date, Literal(row["Date"], datatype=XSD.date)))

    # Wydawca (Publisher)
    if "Publisher" in row and pd.notna(row["Publisher"]):
        publisher_bf = URIRef(f"{work_uri}/publisher")
        g.add((work_uri, BF.publisher, publisher_bf))
        g.add((publisher_bf, RDFS.label, Literal(row["Publisher"])))

    # Typ zasobu (Type)
    if "Type" in row and pd.notna(row["Type"]):
        genre_bf = URIRef(f"{work_uri}/genre")
        g.add((work_uri, BF.genreForm, genre_bf))
        g.add((genre_bf, RDFS.label, Literal(row["Type"])))

    # Format (Format)
    if "Format" in row and pd.notna(row["Format"]):
        format_bf = URIRef(f"{work_uri}/format")
        g.add((work_uri, BF.mediaType, format_bf))
        g.add((format_bf, RDFS.label, Literal(row["Format"])))

    # Język (Language)
    if "Language" in row and pd.notna(row["Language"]):
        language_bf = URIRef(f"{work_uri}/language")
        g.add((work_uri, BF.language, language_bf))
        g.add((language_bf, RDFS.label, Literal(row["Language"])))

    # Temat (Subject)
    if "Subject" in row and pd.notna(row["Subject"]):
        subject_bf = URIRef(f"{work_uri}/subject")
        g.add((work_uri, BF.subject, subject_bf))
        g.add((subject_bf, RDFS.label, Literal(row["Subject"])))

    # Opis (Description)
    if "Description" in row and pd.notna(row["Description"]):
        description_bf = URIRef(f"{work_uri}/description")
        g.add((work_uri, BF.note, description_bf))
        g.add((description_bf, RDFS.label, Literal(row["Description"])))

    # Relacja (Relation) z innym zasobem
    if "Relation" in row and pd.notna(row["Relation"]):
        relation_bf = URIRef(f"{work_uri}/relation")
        g.add((work_uri, BF.relatedTo, relation_bf))
        g.add((relation_bf, RDFS.label, Literal(row["Relation"])))

    # Zakres (Coverage)
    if "Coverage" in row and pd.notna(row["Coverage"]):
        coverage_bf = URIRef(f"{work_uri}/coverage")
        g.add((work_uri, BF.coverage, coverage_bf))
        g.add((coverage_bf, RDFS.label, Literal(row["Coverage"])))

    # Prawa (Rights)
    if "Rights" in row and pd.notna(row["Rights"]):
        rights_bf = URIRef(f"{work_uri}/rights")
        g.add((work_uri, BF.rights, rights_bf))
        g.add((rights_bf, RDFS.label, Literal(row["Rights"])))

    return g

# Ładowanie danych z Excela
data = pd.read_excel("dublin_core_data.xlsx")

# Przetwarzanie każdego rekordu na BIBFRAME
rdf_graph = Graph()
rdf_graph.bind("bf", BF)
rdf_graph.bind("bflc", BFLc)
rdf_graph.bind("dc", DC)

for _, row in data.iterrows():
    graph = dc_to_bibframe(row)
    rdf_graph += graph  # Dodanie danych do głównego grafu

# Zapisywanie do pliku RDF/XML
rdf_graph.serialize("output_bibframe.rdf", format="xml")

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

# Namespace
BF = Namespace("http://id.loc.gov/ontologies/bibframe/")
DC = Namespace("http://purl.org/dc/elements/1.1/")

# Konwerter danych
def convert_to_bibframe(row, baseuri="http://example.org/"):
    g = Graph()
    g.bind("bf", BF)
    g.bind("dc", DC)
    
    # Work (artykuł)
    article_work_uri = URIRef(f"{baseuri}{row['IDENTIFIER']}_work")
    g.add((article_work_uri, RDF.type, BF.Work))
    g.add((article_work_uri, BF.title, Literal(row["TITLE"])))
    g.add((article_work_uri, BF.genreForm, Literal(row["TYPE"])))
    
    # Twórca artykułu
    if row.get("CREATOR"):
        contribution_uri = URIRef(f"{article_work_uri}/contribution")
        g.add((article_work_uri, BF.contribution, contribution_uri))
        agent_uri = URIRef(f"{contribution_uri}/agent")
        g.add((contribution_uri, BF.agent, agent_uri))
        g.add((agent_uri, RDF.type, BF.Agent))
        g.add((agent_uri, RDFS.label, Literal(row["CREATOR"])))

    # Work (czasopismo)
    journal_work_uri = URIRef(f"{baseuri}{row['SOURCE']}_work")
    g.add((journal_work_uri, RDF.type, BF.Work))
    g.add((journal_work_uri, BF.title, Literal(row["SOURCE"])))
    if "ISSN" in row:
        issn_uri = URIRef(f"{journal_work_uri}/issn")
        g.add((journal_work_uri, BF.identifiedBy, issn_uri))
        g.add((issn_uri, RDF.type, BF.Issn))
        g.add((issn_uri, Literal(row["ISSN"])))

    # Instance (czasopismo)
    journal_instance_uri = URIRef(f"{journal_work_uri}/instance")
    g.add((journal_instance_uri, RDF.type, BF.Instance))
    g.add((journal_instance_uri, BF.title, Literal(row["SOURCE"])))
    g.add((journal_instance_uri, BF.part, Literal(row["SOURCE_PART"])))  # Zmiana na SOURCE_PART
    if row.get("SOURCE_SUBTITLE"):
        g.add((journal_instance_uri, BF.subtitle, Literal(row["SOURCE_SUBTITLE"])))  # Dodanie SOURCE_SUBTITLE
    g.add((journal_instance_uri, BF.instanceOf, journal_work_uri))

    # Relacja "partOf"
    relation_uri = URIRef(f"{article_work_uri}/relation")
    g.add((article_work_uri, BF.relation, relation_uri))
    g.add((relation_uri, RDF.type, BF.Relation))
    g.add((relation_uri, BF.relationship, URIRef("http://id.loc.gov/vocabulary/relationship/partof")))
    g.add((relation_uri, BF.associatedResource, journal_work_uri))

    # Instance (artykuł)
    article_instance_uri = URIRef(f"{article_work_uri}/instance")
    g.add((article_instance_uri, RDF.type, BF.Instance))
    g.add((article_instance_uri, BF.instanceOf, article_work_uri))
    if row.get("LINK"):
        g.add((article_instance_uri, BF.electronicLocator, URIRef(row["LINK"])))

    return g



# Obróbka danych w DataFrame
def process_source_number(row):
    if pd.notna(row["SOURCE_NUMBER"]):
        source_number = str(row["SOURCE_NUMBER"])  # Konwersja na string
        parts = source_number.split(":", 1)  # Rozdzielenie przy pierwszym ":"
        row["SOURCE_PART"] = parts[0].strip() if len(parts) > 0 else None
        row["SOURCE_SUBTITLE"] = parts[1].strip() if len(parts) > 1 else None
    else:
        row["SOURCE_PART"] = None
        row["SOURCE_SUBTITLE"] = None
    return row

# Ładowanie danych z Excela
data = pd.read_excel("D:/Nowa_praca/KDL.xlsx")

# Przetwarzanie kolumny SOURCE_NUMBER na SOURCE_PART i SOURCE_SUBTITLE
data = data.apply(process_source_number, axis=1)

# Generowanie RDF
rdf_graph = Graph()
rdf_graph.bind("bf", BF)
rdf_graph.bind("bflc", BFLc)
rdf_graph.bind("dc", DC)

for _, row in data.iterrows():
    graph = dc_to_bibframe(row)
    rdf_graph += graph  # Dodanie danych do głównego grafu

# Zapisywanie do pliku RDF/XML
rdf_graph.serialize("output_bibframe.rdf", format="xml")


from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD
from urllib.parse import quote
import pandas as pd

# Definicje przestrzeni nazw
BF = Namespace("http://id.loc.gov/ontologies/bibframe/")
BFLC = Namespace("http://id.loc.gov/ontologies/bflc/")
MADS = Namespace("http://www.loc.gov/mads/rdf/v1#")
MADSRDF = Namespace("http://www.loc.gov/mads/rdf/v1#")
DC = Namespace("http://purl.org/dc/elements/1.1/")
EX = Namespace("http://example.org/")

def convert_to_bibframe(row, baseuri="http://example.org/"):
    g = Graph()
    g.bind("bf", BF)
    g.bind("bflc", BFLC)
    g.bind("madsrdf", MADSRDF)
    g.bind("dc", DC)
    
    # Bezpieczne tworzenie URI
    identifier_encoded = quote(str(row['IDENTIFIER'])) if pd.notna(row.get('IDENTIFIER')) else None
    source_encoded = quote(str(row['SOURCE'])) if pd.notna(row.get('SOURCE')) else None

    # Work (artykuł)
    if identifier_encoded:
        article_work_uri = URIRef(f"{baseuri}{identifier_encoded}#Work")
        g.add((article_work_uri, RDF.type, BF.Work))
        
        # Dodanie tytułu artykułu
        if row.get("TITLE"):
            title_uri = URIRef(f"{article_work_uri}/Title")
            g.add((article_work_uri, BF.title, title_uri))
            g.add((title_uri, RDF.type, BF.Title))
            g.add((title_uri, BF.mainTitle, Literal(row["TITLE"])))
        
        # Dodanie twórcy artykułu (Agent)
        if row.get("CREATOR"):
            creator_encoded = quote(str(row["CREATOR"]))
            agent_uri = URIRef(f"{baseuri}agent/{creator_encoded}")
            g.add((agent_uri, RDF.type, BF.Person))
            g.add((agent_uri, RDFS.label, Literal(row["CREATOR"])))
            
            contribution_uri = URIRef(f"{article_work_uri}/Contribution")
            g.add((article_work_uri, BF.contribution, contribution_uri))
            g.add((contribution_uri, RDF.type, BF.Contribution))
            g.add((contribution_uri, RDF.type, BF.PrimaryContribution))
            g.add((contribution_uri, BF.agent, agent_uri))
            g.add((contribution_uri, BF.role, URIRef("http://id.loc.gov/vocabulary/relators/aut")))
        
        # Dodanie gatunku/formy
        if row.get("TYPE"):
            genre_form_uri = URIRef(f"{article_work_uri}/GenreForm")
            g.add((article_work_uri, BF.genreForm, genre_form_uri))
            g.add((genre_form_uri, RDF.type, MADS.GenreForm))
            g.add((genre_form_uri, RDFS.label, Literal(row["TYPE"])))
            g.add((genre_form_uri, MADSRDF.authoritativeLabel, Literal(row["TYPE"])))
        
        # Dodanie relacji do czasopisma
        if source_encoded:
            relation_uri = URIRef(f"{article_work_uri}/Relation")
            g.add((article_work_uri, BF.relation, relation_uri))
            g.add((relation_uri, RDF.type, BF.Relation))
            g.add((relation_uri, BF.relationship, URIRef("http://id.loc.gov/vocabulary/relationship/partOf")))
            
            # Tworzenie bf:associatedResource z bf:Work czasopisma
            associated_resource_uri = URIRef(f"{relation_uri}/AssociatedResource")
            g.add((relation_uri, BF.associatedResource, associated_resource_uri))
            
            journal_work_uri = URIRef(f"{baseuri}{source_encoded}#Work")
            g.add((associated_resource_uri, BF.resource, journal_work_uri))
            g.add((journal_work_uri, RDF.type, BF.Work))
            
            # Tytuł czasopisma
            if row.get("SOURCE"):
                journal_title_uri = URIRef(f"{journal_work_uri}/Title")
                g.add((journal_work_uri, BF.title, journal_title_uri))
                g.add((journal_title_uri, RDF.type, BF.Title))
                g.add((journal_title_uri, BF.mainTitle, Literal(row["SOURCE"])))
            
            # ISSN czasopisma (jeśli dostępne)
            if row.get("ISSN"):
                issn_uri = URIRef(f"{journal_work_uri}/ISSN")
                g.add((journal_work_uri, BF.identifiedBy, issn_uri))
                g.add((issn_uri, RDF.type, BF.Issn))
                g.add((issn_uri, RDF.value, Literal(row["ISSN"])))
            
            # Dodanie bf:hasInstance dla czasopisma
            journal_instance_uri = URIRef(f"{journal_work_uri}#Instance")
            g.add((journal_work_uri, BF.hasInstance, journal_instance_uri))
            g.add((journal_instance_uri, RDF.type, BF.Instance))
            g.add((journal_instance_uri, BF.instanceOf, journal_work_uri))
            
            # Tytuł instancji czasopisma
            if row.get("SOURCE"):
                instance_title_uri = URIRef(f"{journal_instance_uri}/Title")
                g.add((journal_instance_uri, BF.title, instance_title_uri))
                g.add((instance_title_uri, RDF.type, BF.Title))
                g.add((instance_title_uri, BF.mainTitle, Literal(row["SOURCE"])))
            
            # Numeracja, część, strony itd.
            if row.get("SOURCE_PART"):
                g.add((journal_instance_uri, BF.part, Literal(row["SOURCE_PART"])))
            if row.get("SOURCE_SUBTITLE"):
                g.add((journal_instance_uri, BF.subtitle, Literal(row["SOURCE_SUBTITLE"])))
            if row.get("PAGES"):
                g.add((journal_instance_uri, BF.extent, Literal(f"Strony {row['PAGES']}")))
            
            # Data i miejsce publikacji
            if row.get("PUBLICATION_DATE") or row.get("PUBLICATION_PLACE"):
                provision_activity_uri = URIRef(f"{journal_instance_uri}/ProvisionActivity")
                g.add((journal_instance_uri, BF.provisionActivity, provision_activity_uri))
                g.add((provision_activity_uri, RDF.type, BF.Publication))
                if row.get("PUBLICATION_PLACE"):
                    g.add((provision_activity_uri, BF.place, Literal(row["PUBLICATION_PLACE"])))
                if row.get("PUBLICATION_DATE"):
                    g.add((provision_activity_uri, BF.date, Literal(row["PUBLICATION_DATE"], datatype=XSD.date)))
        
        # Dodanie instancji artykułu
        article_instance_uri = URIRef(f"{article_work_uri}#Instance")
        g.add((article_work_uri, BF.hasInstance, article_instance_uri))
        g.add((article_instance_uri, RDF.type, BF.Instance))
        g.add((article_instance_uri, BF.instanceOf, article_work_uri))
        
        # Tytuł instancji artykułu
        if row.get("TITLE"):
            article_instance_title_uri = URIRef(f"{article_instance_uri}/Title")
            g.add((article_instance_uri, BF.title, article_instance_title_uri))
            g.add((article_instance_title_uri, RDF.type, BF.Title))
            g.add((article_instance_title_uri, BF.mainTitle, Literal(row["TITLE"])))
        
        # Elektroniczny lokalizator
        if row.get("LINK"):
            g.add((article_instance_uri, BF.electronicLocator, URIRef(row["LINK"])))
        
        # Inne właściwości instancji artykułu (np. strony)
        if row.get("PAGES"):
            g.add((article_instance_uri, BF.extent, Literal(f"Strony {row['PAGES']}")))
        
        # Polityka dostępu
        if row.get("open_access"):
            use_policy_uri = URIRef(f"{article_instance_uri}/UsePolicy")
            g.add((article_instance_uri, BF.usageAndAccessPolicy, use_policy_uri))
            g.add((use_policy_uri, RDF.type, BF.UsePolicy))
            g.add((use_policy_uri, RDFS.label, Literal("Open Access")))
        
        # Dodanie egzemplarza (Item)
        item_uri = URIRef(f"{article_instance_uri}#Item")
        g.add((article_instance_uri, BF.itemOf, item_uri))
        g.add((item_uri, RDF.type, BF.Item))
        g.add((item_uri, BF.itemOf, article_instance_uri))
        
        # Jeśli mamy informacje o lokalizacji egzemplarza (np. URL)
        if row.get("LINK"):
            held_item_uri = URIRef(f"{item_uri}/HeldItem")
            g.add((item_uri, BF.electronicLocator, URIRef(row["LINK"])))
            g.add((item_uri, BF.heldBy, URIRef(f"{baseuri}organization/your_library")))
        
        # Tematy (Subjects)
        if row.get("SUBJECTS"):
            subjects = str(row["SUBJECTS"]).split(";")
            for idx, subject in enumerate(subjects):
                subject_encoded = quote(subject.strip())
                subject_uri = URIRef(f"{baseuri}subject/{subject_encoded}")
                g.add((subject_uri, RDF.type, MADS.Topic))
                g.add((subject_uri, RDFS.label, Literal(subject.strip())))
                g.add((subject_uri, MADSRDF.authoritativeLabel, Literal(subject.strip())))
                g.add((article_work_uri, BF.subject, subject_uri))
    
    return g



# Funkcja do przetwarzania kolumny SOURCE_NUMBER
def process_source_number(row):
    if pd.notna(row["SOURCE_NUMBER"]):
        source_number = str(row["SOURCE_NUMBER"])
        parts = source_number.split(":", 1)
        row["SOURCE_PART"] = parts[0].strip() if len(parts) > 0 else None
        row["SOURCE_SUBTITLE"] = parts[1].strip() if len(parts) > 1 else None
    else:
        row["SOURCE_PART"] = None
        row["SOURCE_SUBTITLE"] = None
    return row

# Ładowanie danych z Excela
data = pd.read_excel("D:/Nowa_praca/KDL2.xlsx")

# Przetwarzanie kolumny SOURCE_NUMBER
data = data.apply(process_source_number, axis=1)

# Generowanie RDF
rdf_graph = Graph()
rdf_graph.bind("bf", BF)
rdf_graph.bind("bflc", BFLC)
rdf_graph.bind("madsrdf", MADSRDF)
rdf_graph.bind("dc", DC)

for _, row in data.iterrows():
    graph = convert_to_bibframe(row)
    rdf_graph += graph

# Zapisywanie do pliku RDF/XML
rdf_graph.serialize("output_bibframe.rdf", format="xml")
