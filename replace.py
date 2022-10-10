

'śś'








strings=json.dumps(dictrec)

string_replaced=strings.replace("ś", "ś")

mydict=json.loads(string_replaced)
to_file('zagubiony_arto.mrk',mydict)
