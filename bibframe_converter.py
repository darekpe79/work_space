# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:48:22 2024

@author: dariu
"""
from pymarc import Record, Field, Subfield

# Tworzenie nowego rekordu
record = Record()
record.add_field(
    Field(tag='001', data='123456789')
)
record.add_field(
    Field(
        tag='020',
        indicators=[' ', ' '],
        subfields=[
            Subfield(code='a', value='978-83-0000-000-0')
        ]
    )
)
record.add_field(
    Field(
        tag='100',
        indicators=['1', ' '],
        subfields=[
            Subfield(code='a', value='Kowalski, Jan.')
        ]
    )
)
record.add_field(
    Field(
        tag='245',
        indicators=['1', '0'],
        subfields=[
            Subfield(code='a', value='Przykładowa książka'),
            Subfield(code='c', value='Jan Kowalski.')
        ]
    )
)
record.add_field(
    Field(
        tag='260',
        indicators=[' ', ' '],
        subfields=[
            Subfield(code='a', value='Warszawa :'),
            Subfield(code='b', value='Wydawnictwo XYZ,'),
            Subfield(code='c', value='2020.')
        ]
    )
)
record.add_field(
    Field(
        tag='300',
        indicators=[' ', ' '],
        subfields=[
            Subfield(code='a', value='123 s. ;'),
            Subfield(code='c', value='21 cm.')
        ]
    )
)
record.add_field(
    Field(
        tag='650',
        indicators=[' ', '0'],
        subfields=[
            Subfield(code='a', value='Literatura polska.')
        ]
    )
)

# Zapis rekordu do pliku MARC
with open('probny.mrc', 'wb') as out:
    out.write(record.as_marc())

from pymarc import MARCReader
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS

# Definiowanie przestrzeni nazw
BF = Namespace('http://id.loc.gov/ontologies/bibframe/')
EX = Namespace('http://example.org/')

# Tworzenie grafu RDF
g = Graph()
g.bind('bf', BF)
g.bind('ex', EX)
g.bind('rdf', RDF)
g.bind('rdfs', RDFS)

# Otwieranie pliku MARC
with open('C:/Users/dariu/probny.mrc', 'rb') as fh:
    reader = MARCReader(fh)

    for record in reader:
        # Tworzenie URI dla bf:Work
        if record['001']:
            work_id = record['001'].value()
        else:
            continue  # Pomijanie rekordów bez pola 001

        work_uri = EX['work/' + work_id]
        g.add((work_uri, RDF.type, BF.Work))

        # Tworzenie bf:Instance
        instance_uri = EX['instance/' + work_id]
        g.add((instance_uri, RDF.type, BF.Instance))
        g.add((instance_uri, BF.instanceOf, work_uri))

        # Dodawanie tytułu jako bf:Title
        if record['245'] and 'a' in record['245']:
            title = record['245']['a']
            title_node = BNode()
            g.add((title_node, RDF.type, BF.Title))
            g.add((title_node, BF.mainTitle, Literal(title)))
            g.add((work_uri, BF.title, title_node))
            # Opcjonalnie, powiązanie tytułu z bf:Instance
            g.add((instance_uri, BF.title, title_node))

        # Dodawanie autora jako bf:Contribution
        if record['100'] and 'a' in record['100']:
            author_name = record['100']['a']
            author_uri = EX['agent/' + author_name.replace(' ', '_')]
            g.add((author_uri, RDF.type, BF.Agent))
            g.add((author_uri, RDFS.label, Literal(author_name)))

            contribution_node = BNode()
            g.add((contribution_node, RDF.type, BF.Contribution))
            g.add((contribution_node, BF.agent, author_uri))

            role_node = BNode()
            g.add((role_node, RDF.type, BF.Role))
            g.add((role_node, RDFS.label, Literal('Autor')))
            g.add((contribution_node, BF.role, role_node))

            g.add((work_uri, BF.contribution, contribution_node))

        # Dodawanie tematu (pole 650 $a)
        if record.get_fields('650'):
            for field in record.get_fields('650'):
                if 'a' in field:
                    subject = field['a']
                    subject_uri = EX['subject/' + subject.replace(' ', '_')]
                    g.add((subject_uri, RDF.type, BF.Topic))
                    g.add((subject_uri, RDFS.label, Literal(subject)))
                    g.add((work_uri, BF.subject, subject_uri))

        # Dodawanie ISBN do bf:Instance
        if record['020'] and 'a' in record['020']:
            isbn = record['020']['a']
            isbn_node = BNode()
            g.add((isbn_node, RDF.type, BF.ISBN))
            g.add((isbn_node, RDF.value, Literal(isbn)))
            g.add((instance_uri, BF.identifiedBy, isbn_node))

        # Dodawanie wydawcy do bf:Instance
        if record['260']:
            provision_node = BNode()
            g.add((provision_node, RDF.type, BF.ProvisionActivity))
            g.add((instance_uri, BF.provisionActivity, provision_node))

            if 'b' in record['260']:
                publisher_name = record['260']['b']
                publisher_node = BNode()
                g.add((publisher_node, RDF.type, BF.Agent))
                g.add((publisher_node, RDFS.label, Literal(publisher_name)))
                g.add((provision_node, BF.agent, publisher_node))

            if 'a' in record['260']:
                place_name = record['260']['a']
                place_node = BNode()
                g.add((place_node, RDF.type, BF.Place))
                g.add((place_node, RDFS.label, Literal(place_name)))
                g.add((provision_node, BF.place, place_node))

            if 'c' in record['260']:
                pub_date = record['260']['c']
                g.add((provision_node, BF.date, Literal(pub_date)))

        # Dodawanie formatu (pole 300 $a)
        if record['300'] and 'a' in record['300']:
            extent = record['300']['a']
            g.add((instance_uri, BF.extent, Literal(extent)))

        # Możesz dodać więcej mapowań pól MARC do BIBFRAME tutaj

# Zapis grafu do pliku w formacie RDF/XML
g.serialize(destination='output.rdf', format='xml')

