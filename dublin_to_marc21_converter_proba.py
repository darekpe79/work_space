# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:45:42 2023

@author: dariu
"""

from pymarc import Record, Field
import time
from datetime import datetime, timedelta

def convert_dc_to_marc21(dc_record):
    
    marc_record = Record(force_utf8=True)

    # Leader
    marc_record.leader = '     nam a22     a 4500'

    # Control fields
    marc_record.add_field(Field(tag='001', data='023034'))  # to be filled
    marc_record.add_field(Field(tag='003', data='OCoLC'))  # constant
    marc_record.add_field(Field(tag='005', data=datetime.utcnow().strftime('%Y%m%d%H%M%S.0')))  # timestamp
    marc_record.add_field(Field(tag='008', data='{}s{}\\\\\\\\\\\\{}\\\\\\\\\\\\{}\\und\\{}'.format(
        dc_record['date'][0], dc_record['language'][0], dc_record['format'][0], '0' * 17, dc_record['rights'][0][:3])))

    # Create a new MARC21 record
    marc_record = Record()

    # Add fields to the record based on the Dublin Core elements
    if "creator" in dc_record:
        for creator in dc_record["creator"]:
            marc_record.add_field(Field(tag="100", indicators=[" ", " "], subfields=["a", creator]))
    
    if "contributor" in dc_record:
        for contributor in dc_record["contributor"]:
            marc_record.add_field(Field(tag="700", indicators=[" ", " "], subfields=["a", contributor]))

    if "title" in dc_record:
        marc_record.add_field(Field(tag="245", indicators=[" ", " "], subfields=["a", dc_record["title"][0]]))

    if "publisher" in dc_record:
        marc_record.add_field(Field(tag="260", indicators=[" ", " "], subfields=["b", dc_record["publisher"][0]]))

    if "date" in dc_record:
        marc_record.add_field(Field(tag="260", indicators=[" ", " "], subfields=["c", dc_record["date"][0]]))

    if "format" in dc_record:
        marc_record.add_field(Field(tag="538", indicators=[" ", " "], subfields=["a", dc_record["format"][0]]))

    if "identifier" in dc_record:
        for identifier in dc_record["identifier"]:
            if identifier.startswith("http"):
                marc_record.add_field(Field(tag="856", indicators=["4", "0"], subfields=["u", identifier]))
            else:
                marc_record.add_field(Field(tag="024", indicators=["7", " "], subfields=["a", identifier]))

    if "description" in dc_record:
        marc_record.add_field(Field(tag="520", indicators=["3", " "], subfields=["a", dc_record["description"][0]]))

    if "language" in dc_record:
        marc_record.add_field(Field(tag="041", indicators=[" ", " "], subfields=["a", dc_record["language"][0]]))

    if "source" in dc_record:
        marc_record.add_field(Field(tag="773", indicators=["0", " "], subfields=["g", dc_record["source"][0]]))

    if "type" in dc_record:
        marc_record.add_field(Field(tag="655", indicators=[" ", "7"], subfields=["a", dc_record["type"][0]]))

    if "rights" in dc_record:
        marc_record.add_field(Field(tag="540", indicators=[" ", " "], subfields=["a", dc_record["rights"][0]]))

    # Return the completed MARC21 record
    return marc_record
import pymarc

def convert_dc_to_marc21(dc_record):
    # Create a new MARC record
    marc_record = pymarc.Record()

    # Add fields to the MARC record based on the DC record
    for key in dc_record.keys():
        if key == "contributor":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="700", 
                    indicators=[" ", "1"], 
                    subfields=["a", value]
                )
                marc_record.add_field(field)
        elif key == "date":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="260", 
                    indicators=[" ", " "], 
                    subfields=["c", value]
                )
                marc_record.add_field(field)
        elif key == "description":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="520", 
                    indicators=["3", " "], 
                    subfields=["a", value]
                )
                marc_record.add_field(field)
        elif key == "format":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="856", 
                    indicators=["4", "0"], 
                    subfields=["u", value]
                )
                marc_record.add_field(field)
        elif key == "identifier":
            for value in dc_record[key]:
                if value.startswith("http"):
                    field = pymarc.Field(
                        tag="856", 
                        indicators=["4", "0"], 
                        subfields=["u", value]
                    )
                    marc_record.add_field(field)
                else:
                    field = pymarc.Field(
                        tag="024", 
                        indicators=["7", " "], 
                        subfields=["a", value]
                    )
                    marc_record.add_field(field)
        elif key == "language":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="546", 
                    indicators=[" ", " "], 
                    subfields=["a", value]
                )
                marc_record.add_field(field)
        elif key == "rights":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="540", 
                    indicators=[" ", " "], 
                    subfields=["a", value]
                )
                marc_record.add_field(field)
        elif key == "source":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="773", 
                    indicators=["0", " "], 
                    subfields=["t", value.split(",")[0].strip(), "p", value.split(",")[1].strip(), "n", value.split(",")[2].strip(), "g", "q"]
                )
                marc_record.add_field(field)
        elif key == "title":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="245", 
                    indicators=["0", "0"], 
                    subfields=["a", value]
                )
                marc_record.add_field(field)
        elif key == "type":
            for value in dc_record[key]:
                field = pymarc.Field(
                    tag="245", 
                    indicators=[" ", " "], 
                    subfields=["h", value]
                )
                marc_record.add_field(field)

    return marc_record

dc_record = {
    "creator": ["Smith, John"],
    "title": ["Example Title"],
    "publisher": ["Example Publisher"],
    "date": ["2022"],
    "format": ["application/pdf"],
    "identifier": ["https://example.com/123456", "(URI) 123456"]
}
from pymarc import Record, Field

def convert_dc_to_marc21(dc_record):
    record = Record()
    record.leader = '     nam a22     4a 4500'
    if 'title' in dc_record:
        record.add_field(Field(tag='245', indicators=[' ', ' '], subfields=['a', dc_record['title'][0]]))
    if 'contributor' in dc_record:
        for contributor in dc_record['contributor']:
            record.add_field(Field(tag='700', indicators=['1', ' '], subfields=['a', contributor]))
    if 'description' in dc_record:
        record.add_field(Field(tag='520', indicators=['3', ' '], subfields=['a', dc_record['description'][0]]))
    if 'date' in dc_record:
        record.add_field(Field(tag='260', indicators=[' ', ' '], subfields=['c', dc_record['date'][0]]))
    if 'language' in dc_record:
        record.add_field(Field(tag='546', indicators=[' ', ' '], subfields=['a', dc_record['language'][0]]))
    if 'format' in dc_record:
        record.add_field(Field(tag='538', indicators=[' ', ' '], subfields=['a', dc_record['format'][0]]))
    if 'identifier' in dc_record:
        for identifier in dc_record['identifier']:
            if identifier.startswith('http'):
                record.add_field(Field(tag='856', indicators=['4', '0'], subfields=['u', identifier]))
            else:
                record.add_field(Field(tag='024', indicators=['7', ' '], subfields=['a', identifier]))
    if 'type' in dc_record:
        record.add_field(Field(tag='980', indicators=[' ', ' '], subfields=['a', dc_record['type'][0]]))
    if 'source' in dc_record:
        record.add_field(Field(tag='773', indicators=['0', ' '], subfields=['t', dc_record['source'][0].split(':')[0], 'p', dc_record['source'][0], 'y', dc_record['date'][0]]))
    if 'rights' in dc_record:
        record.add_field(Field(tag='540', indicators=[' ', ' '], subfields=['a', dc_record['rights'][0]]))
    return record


dc_record ={"contributor":
        [
        "Biosca Bas, Antoni"
        ],
    "date":
        [
        "2020"
        ],
    "description":
        [
        "Mercè Puig Rodríguez-Escalona (ed.), Projeccions de la lexicografia llatina medieval a Catalunya (Col·lecció IRCVM - Medieval Cultures), Roma, Viella, 2019, 243 pp. ISBN: 978-88-3313-131-3."
        ],
    "format":
        [
        "application/pdf"
        ],
    "identifier":
        [
        "https://dialnet.unirioja.es/servlet/oaiart?codigo=7701114",
        "(Revista) ISSN 1578-7486",
        "(Revista) ISSN 2255-5056"
        ],
    "language":
        [
        "spa"
        ],
    "rights":
        [
        "LICENCIA DE USO: Los documentos a texto completo incluidos en Dialnet son de acceso libre y propiedad de sus autores y/o editores. Por tanto, cualquier acto de reproducción, distribución, comunicación pública y/o transformación total o parcial requiere el consentimiento expreso y escrito de aquéllos. Cualquier enlace al texto completo de estos documentos deberá hacerse a través de la URL oficial de éstos en Dialnet. Más información: https://dialnet.unirioja.es/info/derechosOAI | INTELLECTUAL PROPERTY RIGHTS STATEMENT: Full text documents hosted by Dialnet are protected by copyright and/or related rights. This digital object is accessible without charge, but its use is subject to the licensing conditions set by its authors or editors. Unless expressly stated otherwise in the licensing conditions, you are free to linking, browsing, printing and making a copy for your own personal purposes. All other acts of reproduction and communication to the public are subject to the licensing conditions expressed by editors and authors and require consent from them. Any link to this document should be made using its official URL in Dialnet. More info: https://dialnet.unirioja.es/info/derechosOAI"
        ],
    "source":
        [
        "Revista de estudios latinos: RELat, ISSN 2255-5056, Nº. 20, 2020, pags. 207-209"
        ],
    "title":
        [
        "Mercè Puig Rodríguez-Escalona (ed.), Projeccions de la lexicografia llatina medieval a Catalunya, Roma 2019"
        ],
    "type":
        [
        "text (article)"
        ]
    }
new_rec=convert_dc_to_marc21(dc_record)
print (new_rec)

def convert_dc_to_marc21(dc_record):
    # create a new empty MARC21 record
    marc_record = pymarc.Record()

    # map fields from Dublin Core to MARC21
    # control fields
    marc_record.leader = "     nam a22     4a 4500"
    marc_record.add_field(pymarc.Field(
        tag='001',
        data=''.join(dc_record['identifier'])
    ))
    marc_record.add_field(pymarc.Field(
        tag='003',
        data='OCoLC'
    ))

    # main entry
    marc_record.add_field(pymarc.Field(
        tag='100',
        indicators=['1', ' '],
        subfields=[
            'a', dc_record['contributor'][0]
        ]
    ))

    # title
    marc_record.add_field(pymarc.Field(
        tag='245',
        indicators=['1', '0'],
        subfields=[
            'a', dc_record['title'][0],
            'h', '[electronic resource] :',
            'b', 'description'
        ]
    ))

    # edition
    marc_record.add_field(pymarc.Field(
        tag='250',
        indicators=[' ', ' '],
        subfields=[
            'a', 'Version 1.0'
        ]
    ))

    # publication, distribution, etc.
    marc_record.add_field(pymarc.Field(
        tag='260',
        indicators=[' ', ' '],
        subfields=[
            'a', 'Place of Publication',
            'b', 'Publisher',
            'c', dc_record['date'][0]
        ]
    ))

    # physical description
    marc_record.add_field(pymarc.Field(
        tag='300',
        indicators=[' ', ' '],
        subfields=[
            'a', '1 online resource (1 electronic text)',
            'b', 'text file',
            'c', dc_record['format'][0]
        ]
    ))

    # series statement
    marc_record.add_field(pymarc.Field(
        tag='490',
        indicators=['1', ' '],
        subfields=[
            'a', dc_record['description'][0]
        ]
    ))

    # general note
    marc_record.add_field(pymarc.Field(
        tag='500',
        indicators=[' ', ' '],
        subfields=[
            'a', dc_record['rights'][0]
        ]
    ))

    # language
    marc_record.add_field(pymarc.Field(
        tag='041',
        indicators=[' ', ' '],
        subfields=[
            'a', dc_record['language'][0]
        ]
    ))

    # identifier
    for identifier in dc_record['identifier']:
        if identifier.startswith('http'):
            marc_record.add_field(pymarc.Field(
                tag='856',
                indicators=['4', '2'],
                subfields=[
                    'u', identifier
                ]
            ))
        elif identifier.startswith('('):
            marc_record.add_field(pymarc.Field(
                tag='022',
                indicators=[' ', ' '],
                subfields=[
                    'a', identifier.strip('()')
                ]
            ))

    # source
    marc_record.add_field(pymarc.Field(
        tag='500',
        indicators=[' ', ' '],
        subfields=[
            'a', dc_record['source'][0]
        ]
    ))