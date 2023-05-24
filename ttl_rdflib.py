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
  