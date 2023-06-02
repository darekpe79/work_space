# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:48:11 2023

@author: dariu
"""

from pymarc import Record, Field
import time
from datetime import datetime, timedelta
datetime.utcnow().strftime('%y%m%d')
import json
from pymarc import TextWriter
from pymarc import XMLWriter
from pymarc import JSONWriter

with open ('D:/Nowa_praca/Espana/periodcs/articles_espana.json', 'r', encoding='utf-8') as json_file:
    data_article=json.load(json_file)
#dc_record['date'][0], dc_record['language'][0], dc_record['format'][0], '0' * 17, dc_record['rights'][0][:3]
#marc_record.add_ordered_field(Field(tag="245", indicators=[" ", " "], subfields=["c", dc_record["contributor"][0]]))
#marc_record.add_ordered_field(Field(tag="245", indicators=[" ", " "], subfields=["c", dc_record["creator"][0]]))
def convert_dc_to_marc21(dc_record, id_001):
    
   #marc_record = Record(force_utf8=True)
    marc_record = Record()
    # Leader
    marc_record.leader = '00000nab a22000000u 4500'

    # Control fields
    marc_record.add_ordered_field(Field(tag='001', data=id_001))  # to be filled
    marc_record.add_ordered_field(Field(tag='003', data='OCoLC'))  # constant
    marc_record.add_ordered_field(Field(tag='005', data=datetime.utcnow().strftime('%Y%m%d%H%M%S.0')))  # timestamp
    marc_record.add_ordered_field(Field(tag='008', data='{}s{}    {}                  {}  '.format(datetime.utcnow().strftime('%y%m%d'),dc_record['date'][0],'sp',dc_record['language'][0]
        )))

    # Create a new MARC21 record
    

    # Add fields to the record based on the Dublin Core elements
    list_ofauthors=dc_record.get('creator', [])+dc_record.get('contributor', [])
    if list_ofauthors:
        marc_record.add_ordered_field(Field(tag="100", indicators=[" ", " "], subfields=["a", list_ofauthors.pop(0)]))
        for author in list_ofauthors:
            marc_record.add_ordered_field(Field(tag="700", indicators=[" ", " "], subfields=["a", author]))
            
    # if "creator" not in dc_record:
  
    #     if "contributor" in dc_record:
            
    #         l=dc_record["contributor"]
    #         if len(l)>1:
                
                
    #             first, rest = l[0], l[1:]
    #             marc_record.add_ordered_field(Field(tag="100", indicators=[" ", " "], subfields=["a", first]))
    #             for contributor in rest:
    #                  marc_record.add_ordered_field(Field(tag="700", indicators=[" ", " "], subfields=["a", contributor]))
    #         else: 
    #             marc_record.add_ordered_field(Field(tag="100", indicators=[" ", " "], subfields=["a", l[0]]))
    # else:
        
    #     for creator in dc_record["creator"]:
    #         marc_record.add_ordered_field(Field(tag="100", indicators=[" ", " "], subfields=["a", creator]))
    #     if "contributor" in dc_record:
    #         for contributor in dc_record["contributor"]:
    #              marc_record.add_ordered_field(Field(tag="700", indicators=[" ", " "], subfields=["a", contributor]))    
    if "title" in dc_record and "creator" in dc_record:
        marc_record.add_ordered_field(Field(tag="245", indicators=[" ", " "], subfields=["a", dc_record["title"][0], "c", dc_record["creator"][0]]))
    elif "title"in dc_record and "contributor" in dc_record:
        marc_record.add_ordered_field(Field(tag="245", indicators=[" ", " "], subfields=["a", dc_record["title"][0], "c", dc_record["contributor"][0]]))
    elif "title" in dc_record:
        marc_record.add_ordered_field(Field(tag="245", indicators=[" ", " "], subfields=["a", dc_record["title"][0]]))
    if "publisher" in dc_record and "date" in dc_record:
        marc_record.add_ordered_field(Field(tag="260", indicators=[" ", " "], subfields=["b", dc_record["publisher"][0],"c", dc_record["date"][0]]))

    elif "date" in dc_record:
        marc_record.add_ordered_field(Field(tag="260", indicators=[" ", " "], subfields=["c", dc_record["date"][0]]))
    elif "publisher" in dc_record:
        marc_record.add_ordered_field(Field(tag="260", indicators=[" ", " "], subfields=["b", dc_record["publisher"][0]]))
        

    if "format" in dc_record:
        marc_record.add_ordered_field(Field(tag="538", indicators=[" ", " "], subfields=["a", dc_record["format"][0]]))

    if "identifier" in dc_record:
        for identifier in dc_record["identifier"]:
            if identifier.startswith("http"):
                marc_record.add_ordered_field(Field(tag="856", indicators=["4", "0"], subfields=["u", identifier]))
            else:
                marc_record.add_ordered_field(Field(tag="022", indicators=[" ", " "], subfields=["a", identifier]))

    if "description" in dc_record:
        marc_record.add_ordered_field(Field(tag="520", indicators=["3", " "], subfields=["a", dc_record["description"][0]]))

    if "language" in dc_record:
        marc_record.add_ordered_field(Field(tag="041", indicators=[" ", " "], subfields=["a", dc_record["language"][0]]))

    if "source" in dc_record:
        marc_record.add_ordered_field(Field(tag="773", indicators=["0", " "], subfields=["t", dc_record["source"][0]]))

    if "type" in dc_record:
        marc_record.add_ordered_field(Field(tag="655", indicators=[" ", "4"], subfields=["a", dc_record["type"][0]]))

    if "rights" in dc_record:
        marc_record.add_ordered_field(Field(tag="540", indicators=[" ", " "], subfields=["a", dc_record["rights"][0]]))
    if "subject" in dc_record:
        for subject in dc_record["subject"]:
             marc_record.add_ordered_field(Field(tag="650", indicators=["0", "4"], subfields=["a", subject]))    
    # Return the completed MARC21 record
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
convert_dc_to_marc21(dc_record, 'id_001')
with open('articles.mrc','wb')as data1:
    counter=0
    for article in data_article:
        counter+=1
        print(article)
        new_rec=convert_dc_to_marc21(article,'spart'+str(counter))
        print (new_rec)
        
        #changing and spliting 773 field:
            
        new_field=[]
        numbers_info=[]
        
        for i, my_field in enumerate(new_rec['773']['t'].split(',')):
            new_subfields=my_field.strip()
            #print(i,new_subfields )
            if i==0:
                new_field.extend(['t',new_subfields])
            if new_subfields.startswith("ISSN"):
                print(i)
                number=i
                new_field.extend(['x',new_subfields.replace("ISSN ", "")])
            try:
                if number:
                    if i>number:
                        numbers_info.append(my_field)
            except:
                pass
        numbers_info.insert(0, 'g') 
        first, rest = numbers_info[0], numbers_info[1:]
        rest_string=(', ').join(rest).strip()
        new_field.append(first)
        new_field.append(rest_string)
        
        new_rec.remove_field(new_rec['773'])
        my_new_773_field = Field(
        
                tag = '773', 
        
                indicators = ['0',' '],
        
                subfields = new_field
                ) 
        
        new_rec.add_ordered_field(my_new_773_field)       
        print (new_rec)    
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