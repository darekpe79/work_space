# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:30:15 2023

@author: dariu
"""

from rdflib import Graph, plugin
from rdflib.serializer import Serializer
from rdflib.namespace import DC, DCTERMS, DOAP, FOAF, SKOS, OWL, RDF, RDFS, VOID, XMLNS, XSD
from rdflib import Dataset
from rdflib import URIRef
from rdflib import Literal

g = Graph()
g.parse("D:/Nowa_praca/lit_bn_skos_27_04_23.ttl", format='ttl')


len(g)
v = g.serialize(format="json-ld")
y = json.loads(v)
#subject predicate object
words={} 
for word in tqdm(words655):
    
    objects=Literal("Apologia", lang='pl')
   # subject = URIRef("http://id.sgcb.mcu.es/Autoridades/LEM201014730/concept")
    predicate=URIRef("http:/www.w3.org/2004/02/skos/core#closeMatch")
    
    
    close_matches=[]
    loc_library=[]
    for sub, pred, obj in g.triples((None, None, objects)):  
        print(sub,pred,obj)
        
        
        for s,p,o in g.triples((sub, None, None)):
            print(s,p,o)
            my_close_matches=str(o)
            
from rdflib import Graph, Literal, Namespace, RDF, URIRef

graph = Graph()
skos = Namespace('http://www.w3.org/2004/02/skos/core#')
graph.bind('skos', skos)

graph.add((URIRef('URI'), RDF['type'], skos['Concept']))
graph.add((URIRef('URI'), skos['prefLabel'], Literal('Temp', lang='en')))
graph.add((URIRef('URI'), skos['related'], URIRef('URI-Related')))
print(graph.serialize(format='pretty-xml'))
#skos = Namespace('http://www.w3.org/2004/02/skos/core#')
graph.bind('skos', skos)
graph.add((URIRef('https:/bn-lit-skos.lab.dariah.pl/scheme/Agenci_literaccy'), RDF['type'], skos['Concept']))

graph.add((URIRef('https:/bn-lit-skos.lab.dariah.pl/scheme/Agenci_literaccy'), skos['prefLabel'], Literal('Agenci literaccy', lang='pl')))
graph.add((URIRef('https:/bn-lit-skos.lab.dariah.pl/scheme/Agenci_literaccy'), skos['closeMatch'], URIRef('http:/www.wikidata.org/entity/Q4764988')))
graph.add((URIRef('https:/bn-lit-skos.lab.dariah.pl/scheme/Agenci_literaccy'), skos['altLabel'], Literal('Agent literacki', lang='pl')))

print(graph.serialize(format='pretty-xml'))
#MARC to RDF
from pymarc import MARCReader
from rdflib import Graph, URIRef, Literal, RDF, Namespace

# Create an RDF graph
graph = Graph()

# Define namespaces
bibframe = Namespace("http://id.loc.gov/ontologies/bibframe/")
madsrdf = Namespace("http://www.loc.gov/mads/rdf/v1#")

# Open and read the MARC file
with open('input.mrc', 'rb') as marc_file:
    reader = MARCReader(marc_file)

    # Iterate over each MARC record
    for record in reader:
        # Extract relevant fields from the MARC record
        title = record['245']['a']
        creator = record['700']['a']
        url = record['856']['u']

        # Create a new Bibframe Work URI
        work_uri = URIRef(url)

        # Add triples for the Work
        graph.add((work_uri, RDF.type, bibframe.Work))
        graph.add((work_uri, bibframe.title, Literal(title)))
        graph.add((work_uri, bibframe.creator, URIRef(creator)))

        # Add triples for the Instance
        instance_uri = URIRef(url + '/instance')
        graph.add((instance_uri, RDF.type, bibframe.Instance))
        graph.add((instance_uri, bibframe.title, Literal(title)))
        graph.add((work_uri, bibframe.instance, instance_uri))

# Serialize the graph to RDF/XML
rdf_data = graph.serialize(format='xml')

# Write the RDF data to a file
with open('output.rdf', 'wb') as rdf_file:
    rdf_file.write(rdf_data)


from pymarc import MARCReader
from rdflib import Graph, URIRef, Literal, RDF, Namespace

# Create an RDF graph
graph = Graph()

# Define namespaces
bibframe = Namespace("http://id.loc.gov/ontologies/bibframe/")
madsrdf = Namespace("http://www.loc.gov/mads/rdf/v1#")

# Open and read the MARC file
with open('input.mrc', 'rb') as marc_file:
    reader = MARCReader(marc_file)

    # Iterate over each MARC record
    for record in reader:
        # Extract relevant fields from the MARC record
        title = record['245']['a']
        creator = record['700']['a']
        url = record['856']['u']
        language = record['041']['a']
        subjects = record.get_fields('655')
        format_type = record['380']['a']
        pbl = record['710']['a']
        pbl_uri = record['710']['4']
        pbl_uri = URIRef(pbl_uri)
        pbl_label = Literal(pbl)

        # Create a new Bibframe Work URI
        work_uri = URIRef(url)

        # Add triples for the Work
        graph.add((work_uri, RDF.type, bibframe.Work))
        graph.add((work_uri, bibframe.title, Literal(title)))
        graph.add((work_uri, bibframe.creator, URIRef(creator)))

        # Add triples for the Instance
        instance_uri = URIRef(url + '/instance')
        graph.add((instance_uri, RDF.type, bibframe.Instance))
        graph.add((instance_uri, bibframe.title, Literal(title)))
        graph.add((work_uri, bibframe.instance, instance_uri))

        # Add triples for Language
        lang_uri = URIRef('http://id.loc.gov/vocabulary/iso639-2/' + language)
        graph.add((instance_uri, bibframe.language, lang_uri))

        # Add triples for Subjects
        for subject in subjects:
            subject_uri = URIRef('http://id.loc.gov/authorities/subjects/' + subject['a'])
            graph.add((instance_uri, bibframe.subject, subject_uri))

        # Add triples for Format Type
        format_type_uri = URIRef('http://id.loc.gov/vocabulary/contentTypes/' + format_type)
        graph.add((instance_uri, bibframe.extent, format_type_uri))

        # Add triples for Publisher
        graph.add((instance_uri, bibframe.publisher, pbl_uri))
        graph.add((pbl_uri, RDF.type, bibframe.Agent))
        graph.add((pbl_uri, madsrdf.authoritativeLabel, pbl_label))

# Serialize the graph to RDF/XML
rdf_data = graph.serialize(format='xml')

# Write the RDF data to a file
with open('output.rdf', 'wb') as rdf_file:
    rdf_file.write(rdf_data)   





#FUNCTION MARC TO RDF

from pymarc import MARCReader
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS

def marc21_to_bibframe(marc_file):
    # Create an RDF graph
    graph = Graph()
    
    # Define namespaces
    bibframe = Namespace("http://id.loc.gov/ontologies/bibframe/")
    marcrel = Namespace("http://id.loc.gov/vocabulary/relators/")
    madsrdf = Namespace("http://www.loc.gov/mads/rdf/v1#")
    skos = Namespace("http://www.w3.org/2004/02/skos/core#")
    dcterms = Namespace("http://purl.org/dc/terms/")
    
    # Bind namespaces to the graph
    graph.bind("bf", bibframe)
    graph.bind("marcrel", marcrel)
    graph.bind("madsrdf", madsrdf)
    graph.bind("skos", skos)
    graph.bind("dcterms", dcterms)
    
    # Open and read the MARC file
    with open(marc_file, 'rb') as file:
        reader = MARCReader(file)
        
        # Process each MARC record
        for record in reader:
            # Create a new BIBFRAME resource
            work_uri = URIRef(f"http://example.org/work/{record['001'].value()}")
            graph.add((work_uri, RDF.type, bibframe.Work))
            
            # Extract and add relevant fields
            if '245' in record:
                title = record['245']['a']
                graph.add((work_uri, bibframe.title, Literal(title)))
            
            if '100' in record:
                creator = record['100']['a']
                graph.add((work_uri, bibframe.contributor, URIRef(f"http://example.org/agent/{creator}")))
                graph.add((URIRef(f"http://example.org/agent/{creator}"), RDF.type, bibframe.Agent))
                graph.add((URIRef(f"http://example.org/agent/{creator}"), RDFS.label, Literal(creator)))
            
            if '260' in record:
                publisher = record['260']['b']
                graph.add((work_uri, bibframe.publisher, Literal(publisher)))
                
            if '300' in record:
                physical_description = record['300']['a']
                graph.add((work_uri, bibframe.physicalDescription, Literal(physical_description)))
            
            if '650' in record:
                subjects = record.get_fields('650')
                for subject in subjects:
                    term = subject['a']
                    graph.add((work_uri, bibframe.subject, URIRef(f"http://example.org/subject/{term}")))
                    graph.add((URIRef(f"http://example.org/subject/{term}"), RDF.type, skos.Concept))
                    graph.add((URIRef(f"http://example.org/subject/{term}"), skos.prefLabel, Literal(term)))
                    
            # Add more fields as needed
            
    # Return the RDF graph
    return graph
     