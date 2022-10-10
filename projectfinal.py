import requests
import json
import xlsxwriter
workbook   = xlsxwriter.Workbook('RakRadek.xlsx')
worksheet = workbook.add_worksheet()
url = "http://data.bn.org.pl/api/bibs.json?author=Rak"
data = requests.get(url).json()
nexturl=data['nextPage']
nexturl2=nexturl.replace("json", "marc")
print(nexturl2)
list=data['bibs']
while len(nexturl)>0:
    data = requests.get(data['nextPage']).json()
    nexturl=data['nextPage']
    nexturl2=nexturl.replace("json", "marc")
    print(nexturl)
    print(nexturl2)
    list.extend(data['bibs'])
autor=[d['author'] for d in list]
y=[d['title'] for d in list]
tytul=[i.split('/', 1)[0] for i in y]
zrodlo=[i.split('/', 1)[1] for i in y]
wydawca=[d['publisher'] for d in list]
nazwykolumn = ('Autor Publikacji','Tytuł Publikacji','Źródło','Wydawca','Autor Z pola 100 wraz viafid')
def kolumna(komorka,dane):
    return worksheet.write_column(komorka, dane)
worksheet.write_row('A1',nazwykolumn)
kolumna('A2', autor)
kolumna('B2',tytul)
kolumna('C2',zrodlo)
kolumna('D2',wydawca)
marc=[d['marc']['fields'][13] for d in list]
#print(marc)
autor100=[]
for m in list:
    z=m['marc']['fields']
    #print(type(z))
    for x in z:

        for k,v in x.items():

            if k=='100':
                print(k,v['subfields'][0]['a'])
                all100authors=v['subfields'][0]['a']
                if all100authors not in autor100:
                    autor100.append(v['subfields'][0]['a'])
#print(autor100)
viafid=[]
for autor in autor100:
    url = "http://www.viaf.org/viaf/AutoSuggest?query="+autor
    data = requests.get(url).json()
    viafid.append(data['result'][0]['viafid'])
#print (viafid)
autorViafid=[]
for pos,val in enumerate(autor100):
    if len(viafid[pos])>0:
        autorViafid.append('{} viafid: {}'.format(val,viafid[pos]))
#print(autorViafid)
kolumna('E2',autorViafid)
workbook.close()
