from pymarc import MARCReader
from pymarc import parse_json_to_array
with open(r"F:\Nowa_praca\fennica\msplit00000002.mrc", 'rb') as fh:
    reader = MARCReader(fh)
    for record in reader:
        if record['100'] is not None:
          if record['100']['a'] is not None:
            print(record['100']['a'])


