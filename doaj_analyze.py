import time
from sickle import Sickle
import xml.etree.ElementTree as ET

# Initialize the Sickle object
sickle = Sickle('https://doaj.org/oai.article')

# Define the dictionary to store counts of the fields
field_counts = {
    'dc:title': 0,
    'dc:creator': 0,
    'dc:subject': 0,
    'dc:description': 0,
    'dc:publisher': 0,
    'dc:date': 0,
    'dc:type': 0,
    'dc:format': 0,
    'dc:identifier': 0,
    'dc:source': 0,
    'dc:language': 0,
    'dc:relation': 0,
    'dc:coverage': 0,
    'dc:rights': 0,
    'orcid_count': 0  # Counter for ORCID
}

# Counter for records fetched
counter = 0

# Limit of records to fetch
limit = 10000

# Namespaces for XML parsing
namespaces = {'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/', 'dc': 'http://purl.org/dc/elements/1.1/'}

try:
    records = sickle.ListRecords(metadataPrefix='oai_dc', raw=True)  # raw=True to retrieve raw XML records
    for record in records:
        #print(record)
        if counter >= limit:
            break
        tree = ET.ElementTree(ET.fromstring(record.raw))
        for tag in field_counts.keys():
            results = tree.findall(f".//{tag}", namespaces)
            if results:
                print(results)
                field_counts[tag] += len(results)

        # Extracting ORCID
        creators = tree.findall(".//dc:creator", namespaces)
        for creator in creators:
            orcid = creator.get('id')  # Get the id attribute which contains the ORCID
            if orcid and 'orcid.org' in orcid:
                #print(creator)
                field_counts['orcid_count'] += 1

        counter += 1

except Exception as e:
    print(f"Error while fetching records: {e}")

print("Counts of fields in harvested records:")
for field, count in field_counts.items():
    print(f"{field}: {count}")

# one count of subject by one record (even when we have many subjects in one record)


from sickle import Sickle
import xml.etree.ElementTree as ET

# Initialize the Sickle object
sickle = Sickle('https://doaj.org/oai.article')

# Define the dictionary to store counts of the fields
field_counts = {
    'dc:title': 0,
    'dc:creator': 0,
    'dc:subject': 0,
    'dc:description': 0,
    'dc:publisher': 0,
    'dc:date': 0,
    'dc:type': 0,
    'dc:format': 0,
    'dc:identifier': 0,
    'dc:source': 0,
    'dc:language': 0,
    'dc:relation': 0,
    'dc:coverage': 0,
    'dc:rights': 0,
    'orcid_count': 0  # Counter for ORCID
}

# Counter for records fetched
counter = 0

# Limit of records to fetch
limit = 1000

# Namespaces for XML parsing
namespaces = {'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/', 'dc': 'http://purl.org/dc/elements/1.1/'}

try:
    records = sickle.ListRecords(metadataPrefix='oai_dc', raw=True)  # raw=True to retrieve raw XML records
    for record in records:
        print(record)
        if counter >= limit:
            break
        tree = ET.ElementTree(ET.fromstring(record.raw))
        for tag in field_counts.keys():
            
            result = tree.find(f".//{tag}", namespaces)  # Use find() instead of findall() to get the first occurrence
            if result is not None:
                field_counts[tag] += 1

        # Extracting ORCID
        creator = tree.find(".//dc:creator", namespaces)
        if creator is not None:
            orcid = creator.get('id')  # Get the id attribute which contains the ORCID
            if orcid and 'orcid.org' in orcid:
                field_counts['orcid_count'] += 1

        counter += 1

except Exception as e:
    print(f"Error while fetching records: {e}")

print("Counts of fields in harvested records:")
for field, count in field_counts.items():
    print(f"{field}: {count}")
    
#%%        
# oai_doaj   
from sickle import Sickle
import xml.etree.ElementTree as ET

# Initialize the Sickle object
sickle = Sickle('https://doaj.org/oai.article')

# Define the dictionary to store counts of the fields
field_counts = {
    'language': 0,
    'publisher': 0,
    'journalTitle': 0,
    'issn': 0,
    'eissn': 0,
    'publicationDate': 0,
    'volume': 0,
    'issue': 0,
    'startPage': 0,
    'endPage': 0,
    'doi': 0,
    'publisherRecordId': 0,
    'title': 0,
    'orcid_count': 0,
    'abstract': 0,
    'fullTextUrl': 0,
    'keywords': 0
}

# Counter for records fetched
counter = 0

# Limit of records to fetch
limit = 1000

# Namespaces for XML parsing
namespaces = {
    'oai': 'http://www.openarchives.org/OAI/2.0/',
    'oai_doaj': 'http://doaj.org/features/oai_doaj/1.0/'
}

try:
    records = sickle.ListRecords(metadataPrefix='oai_doaj', raw=True)
    for record in records:
        #print(record)
        if counter >= limit:
            break
        tree = ET.ElementTree(ET.fromstring(record.raw))
        for tag in field_counts.keys():
            result = tree.find(f".//oai_doaj:{tag}", namespaces)
            if result is not None:
                field_counts[tag] += 1

        # Extracting ORCID from authors
        authors = tree.findall(".//oai_doaj:author", namespaces)
        for author in authors:
            orcid = author.find("oai_doaj:orcid_id", namespaces)
            if orcid is not None and 'orcid.org' in orcid.text:
                field_counts['orcid_count'] += 1

        counter += 1

except Exception as e:
    print(f"Error while fetching records: {e}")

print("Counts of fields in harvested records:")
for field, count in field_counts.items():
    print(f"{field}: {count}")
