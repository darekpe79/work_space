import regex as re
import json
import requests
class ViafSearcher:
    def __init__(self):
        pass

    def make_search_phrase(self, field):
        
        
        #musimy sprawdzić czy daje pole 100, 700
        
        subfield_a_regex='(?<=\$a).*?(?=\$|$)'
        subfield_d_regex='(?<=\$d).*?(?=\$|$)'
        name=re.search(subfield_a_regex, field).group(0).strip(', -')
        date=re.search(subfield_d_regex, field).group(0).strip(', .()[]')
        return name, date
    
    
    def get_response(self, searchphrase, startrecord, maxrecords):
        search_query = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{searchphrase}&maximumRecords={maxrecords}&startRecord={startrecord}&httpAccept=application/json".format(searchphrase = searchphrase, startrecord = startrecord, maxrecords=maxrecords  )
        try:
            r = requests.get(search_query)
            r.encoding = 'utf-8'
            response = r.json()
        except ConnectionError:
            return 'ConnectionError'
        except json.decoder.JSONDecodeError:
            return 'JSONDecodeError'
        
        return response
    
    def get_all_responses(self, searchphrase, startrecord=1, maxrecords=10, limit=100):
        if maxrecords not in range (1,11):
            raise ValueError('maxrecords must be in range (1-10)')
        else:    
                
                
            first_response=self.get_response(searchphrase, startrecord, maxrecords)
            number_of_records=int(first_response['searchRetrieveResponse']['numberOfRecords'])
            if number_of_records>maxrecords:
                for elem in range(number_of_records)[maxrecords+1:limit:maxrecords]:
                    search = "http://www.viaf.org//viaf/search?query=local.personalNames+=+{search}&maximumRecords=10&startRecord={number}&httpAccept=application/json".format(search=name.strip(), number = elem)
                    try:
                        r = requests.get(search)
                        r.encoding = 'utf-8'        
                        response['searchRetrieveResponse']['records'] = response['searchRetrieveResponse']['records'] + r.json()['searchRetrieveResponse']['records']
                    except:
                        continue
        
        
    
    
    def viaf_response(self, field):
        name, date= self.make_search_phrase(field)
        response=self.get_response(name+' '+date)
        
        return response
        
      
    def mark_to_dict(self,path):
        records = []
        with open(path, 'r', encoding = 'utf-8') as mrk:
            record = []
            for line in mrk.readlines():
                if line != '\n':
                    record.append(line)
                else:
                    records.append(record)
                    record = []
        for record in records:
            rec_dict = {}
            for field in record:
                if field[1:4] in rec_dict.keys():
                    rec_dict[field[1:4]] = '❦'.join([rec_dict[field[1:4]], field[6:].strip()])
                else:
                    rec_dict[field[1:4]] = field[6:].strip()
            recs2table.append(rec_dict)
            val700=[]
            for rekord in dictrec:
                for key, val in rekord.items():
                    if key=='700':
                        val100.append(val.split('❦'))
        return recs2table
 
        
        
        
    
        
searcher=ViafSearcher()

data=searcher.make_search_phrase('1\$aOesch, Lennart,$d1892-1978,$ekirjoittaja.$0(FIN11)000065427')
data2=searcher.get_response('1\$aOesch, Lennart,$d1892-1978,$ekirjoittaja.$0(FIN11)000065427', 1,12)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
1\$aOesch, Lennart,$d1892-1978,$ekirjoittaja.$0(FIN11)000065427
1\$aVäisänen, A. O.$q(Armas Otto),$d1890-1969,$ekirjoittaja.
1\$aVäisänen, A. O.$q(Armas Otto),$d1890-1969,$ekääntäjä.
1\$aJoutsen, Etti.

    