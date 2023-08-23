# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 18:15:13 2023

@author: dariu
"""

from pymarc import Record, Field, Subfield
from datetime import datetime

from pymarc import Record, Field, Subfield
from datetime import datetime
from pymarc import Record, Field
import time
from datetime import datetime, timedelta
datetime.utcnow().strftime('%y%m%d')
import json
with open ('D:/Nowa_praca/Espana/periodcs/articles_espana.json', 'r', encoding='utf-8') as json_file:
    data_article=json.load(json_file)
def convert_dc_to_marc21(dc_record, id_001):
    marc_record = Record()
    marc_record.leader = '00000nab a22000000u 4500'

    # Control fields
    marc_record.add_ordered_field(Field(tag='001', data=id_001))  # to be filled
    marc_record.add_ordered_field(Field(tag='003', data='OCoLC'))  # constant
    marc_record.add_ordered_field(Field(tag='005', data=datetime.utcnow().strftime('%Y%m%d%H%M%S.0')))  # timestamp
    marc_record.add_ordered_field(Field(tag='008', data='{}s{}    {}                  {}  '.format(datetime.utcnow().strftime('%y%m%d'),dc_record['date'][0],'sp',dc_record['language'][0]
        )))

    list_ofauthors = dc_record.get('creator', []) + dc_record.get('contributor', [])
    if list_ofauthors:
        marc_record.add_ordered_field(Field(tag="100", indicators=[" ", " "], subfields=[
            Subfield(code='a', value=list_ofauthors.pop(0))
        ]))
        for author in list_ofauthors:
            marc_record.add_ordered_field(Field(tag="700", indicators=[" ", " "], subfields=[
                Subfield(code='a', value=author)
            ]))

    if "title" in dc_record:
        subfields_245 = [Subfield(code='a', value=dc_record["title"][0])]
        if "creator" in dc_record:
            subfields_245.append(Subfield(code='c', value=dc_record["creator"][0]))
        elif "contributor" in dc_record:
            subfields_245.append(Subfield(code='c', value=dc_record["contributor"][0]))

        marc_record.add_ordered_field(Field(tag="245", indicators=[" ", " "], subfields=subfields_245))

    if "publisher" in dc_record or "date" in dc_record:
        subfields_260 = []
        if "publisher" in dc_record:
            subfields_260.append(Subfield(code='b', value=dc_record["publisher"][0]))
        if "date" in dc_record:
            subfields_260.append(Subfield(code='c', value=dc_record["date"][0]))

        marc_record.add_ordered_field(Field(tag="260", indicators=[" ", " "], subfields=subfields_260))

    if "format" in dc_record:
        marc_record.add_ordered_field(Field(tag="538", indicators=[" ", " "], subfields=[
            Subfield(code='a', value=dc_record["format"][0])
        ]))

    if "identifier" in dc_record:
        for identifier in dc_record["identifier"]:
            if identifier.startswith("http"):
                marc_record.add_ordered_field(Field(tag="856", indicators=["4", "0"], subfields=[
                    Subfield(code='u', value=identifier)
                ]))
            else:
                marc_record.add_ordered_field(Field(tag="022", indicators=[" ", " "], subfields=[
                    Subfield(code='a', value=identifier)
                ]))

    if "description" in dc_record:
        marc_record.add_ordered_field(Field(tag="520", indicators=["3", " "], subfields=[
            Subfield(code='a', value=dc_record["description"][0])
        ]))

    if "language" in dc_record:
        marc_record.add_ordered_field(Field(tag="041", indicators=[" ", " "], subfields=[
            Subfield(code='a', value=dc_record["language"][0])
        ]))

    if "source" in dc_record:
        marc_record.add_ordered_field(Field(tag="773", indicators=["0", " "], subfields=[
            Subfield(code='t', value=dc_record["source"][0])
        ]))

    if "type" in dc_record:
        marc_record.add_ordered_field(Field(tag="655", indicators=[" ", "4"], subfields=[
            Subfield(code='a', value=dc_record["type"][0])
        ]))

    if "rights" in dc_record:
        marc_record.add_ordered_field(Field(tag="540", indicators=[" ", " "], subfields=[
            Subfield(code='a', value=dc_record["rights"][0])
        ]))

    if "subject" in dc_record:
        for subject in dc_record["subject"]:
            marc_record.add_ordered_field(Field(tag="650", indicators=["0", "4"], subfields=[
                Subfield(code='a', value=subject)
            ]))

    return marc_record


dc_record ={"contributor":
        [
        "Biosca Bas, Antoni","daro, pe"
        ],
    "date":
        [
        "2020"
        ],
    "publisher":
        ['abba'],
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
        ],
        "subject":
        [
        "Historia literaria",
        "Barroco español",
        "academias literarias",
        "Corte",
        "monarquía",
        "Felipe IV",
        "Literary History",
        "Spanish Baroque",
        "Literary Academies",
        "Court",
        "Monarchy",
        "Philip IV"
        ],
    }
dd=convert_dc_to_marc21(dc_record, 'id_001')
print(dd)





    #changing and spliting 773 field:
        
    new_field=[]
    numbers_info=[]
    
    for i, my_field in enumerate(dd['773']['t'].split(',')):
        new_subfields=my_field.strip()
        #print(i,new_subfields )
        if i==0:
            new_field.append(Subfield('t',new_subfields))
        if new_subfields.startswith("ISSN"):
            print(i)
            number=i
            new_field.append(Subfield('x',new_subfields.replace("ISSN ", "")))
        try:
            if number:
                if i>number:
                    numbers_info.append(my_field)
        except:
            pass
        
    
    
    
    rest_string=(',').join(numbers_info).strip()
    new_field.append(Subfield('g',rest_string))

    
    dd.remove_field(dd['773'])
    my_new_773_field = Field(
    
            tag = '773', 
    
            indicators = ['0',' '],
    
            subfields = new_field
            ) 
    
    dd.add_ordered_field(my_new_773_field)       
    print (dd)
    #Working with issn:
        
    if new_rec.get_fields('022'):
        my_022s = new_rec.get_fields('022')
        if len(my_022s)>1:
            new_issn_list=[]
            for my_500 in my_022s:
                
                new_issn=my_500.value().replace('(Revista) ISSN ', '')
                new_issn_list.append(new_issn)
                
                
            
            index_x=new_field.index('x')
            issn=new_field[index_x+1]
            new_issn_list.remove(issn)
            for my_500 in my_022s:
                new_rec.remove_field(my_500)
                
            my_new_022_field = Field(
            
                    tag = '022', 
            
                    indicators = [' ',' '],
            
                    subfields = ['a', issn,'l',new_issn_list[0]]
                    ) 
            
            new_rec.add_ordered_field(my_new_022_field)
        else:
                   
            new_rec['022']['a']=new_rec['022']['a'].replace('(Revista) ISSN ', '')
    data1.write(new_rec.as_marc())        