# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:18:30 2025

@author: darek
"""

from pymarc import MARCReader, XMLWriter
from io import BytesIO

def convert_marc_to_marcxml(input_file, output_file):
    """
    Converts a MARC file (.mrc or .mrk) to MARCXML format.

    Args:
        input_file (str): Path to the MARC file (e.g., .mrc or .mrk).
        output_file (str): Path to save the MARCXML file.
    """
    try:
        with open(input_file, 'rb') as marc_file, open(output_file, 'wb') as xml_file:
            reader = MARCReader(marc_file)
            writer = XMLWriter(xml_file)

            for record in reader:
                writer.write(record)
            
            writer.close()
            print(f"Conversion complete. MARCXML saved to {output_file}.")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Example usage
input_marc_file = 'D:/Nowa_praca/marki_po_updatach 2025,2024/es_articles__08-02-2024.mrc'  # Replace with your .mrc or .mrk file path
output_marcxml_file = 'output.xml'
convert_marc_to_marcxml(input_marc_file, output_marcxml_file)


from lxml import etree

def marcxml_to_bibframe(input_marcxml, xslt_path, output_bibframe, baseuri=None, idsource=None):
    """
    Converts MARCXML to BIBFRAME using the LoC's XSLT stylesheets.

    Args:
        input_marcxml (str): Path to the input MARCXML file.
        xslt_path (str): Path to the marc2bibframe2.xsl file.
        output_bibframe (str): Path to save the BIBFRAME XML output.
        baseuri (str): Base URI for generated entities.
        idsource (str): Identifier source URI.
    """
    try:
        # Load MARCXML and XSLT files
        with open(input_marcxml, 'rb') as xml_file, open(xslt_path, 'rb') as xslt_file:
            xml_tree = etree.parse(xml_file)
            xslt_tree = etree.parse(xslt_file)

        # Configure parameters
        transform = etree.XSLT(xslt_tree)
        params = {}
        if baseuri:
            params['baseuri'] = f"'{baseuri}'"
        if idsource:
            params['idsource'] = f"'{idsource}'"

        # Apply transformation
        result_tree = transform(xml_tree, **params)

        # Save the output
        with open(output_bibframe, 'wb') as output_file:
            output_file.write(etree.tostring(result_tree, pretty_print=True, xml_declaration=True, encoding='UTF-8'))

        print(f"Conversion complete. Output saved to {output_bibframe}.")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_marcxml = 'output.xml'
xslt_file = 'D:/marc2bibframe2/xsl/marc2bibframe2.xsl'
output_file = 'output_bibframe.xml'

marcxml_to_bibframe(input_marcxml, xslt_file, output_file, baseuri='http://mylibrary.org/', idsource='http://id.loc.gov/vocabulary/organizations/dlc')


# Example usage
input_marcxml_file = 'output.xml'  # Path to your MARCXML file
xslt_file = 'D:/marc2bibframe2/xsl/marc2bibframe2.xsl'  # Path to your XSLT file
output_bibframe_file = 'bibframe_output.xml'  # Path to save the BIBFRAME file

marcxml_to_bibframe(input_marcxml_file, xslt_file, output_bibframe_file)



import pandas as pd
from lxml import etree

# Przygotowanie DataFrame
data = {
    "identifier": ["1_1_1867_Wislicki_Groch"],
    "link": ["https://drive.google.com/file/d/1QlZE1BY3S8EkMBV9VtxNiuSljzdvXQ8B"],
    "type": ["chapter"],
    "title": ["Groch na ścianę. Parę słów do całej plejady zapoznanych wieszczów naszych"],
    "creator": ["Adam Wyślicki"],
    "author_gender": ["mężczyzna"],
    "journal_title": ["Programy i dyskusje literackie pozytywizmu"],
    "source_number": ["32"],
    "source_place": [""],
    "source_date": [""],
    "date": ["1867"],
    "publication_place": [""],
    "pages": ["44638"],
    "open_access": ["FAŁSZ"]
}

df = pd.DataFrame(data)

# Wyświetlenie DF dla podglądu
print(df)
def df_to_dublin_core(df, output_xml):
    nsmap = {"dc": "http://purl.org/dc/elements/1.1/"}
    root = etree.Element("articles", nsmap=nsmap)
    
    for _, row in df.iterrows():
        article = etree.SubElement(root, "article")
        
        # Obsługa obowiązkowych pól
        etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}identifier").text = str(row["identifier"])
        etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}title").text = str(row["title"])
        etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}creator").text = str(row["creator"])
        etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}type").text = str(row["type"])
        etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}source").text = str(row["journal_title"])
        etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}date").text = str(row["date"])
        etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}relation").text = str(row["link"])
        
        # Obsługa pól opcjonalnych (tylko jeśli nie są puste i kolumna istnieje)
        if "pages" in df.columns and pd.notna(row["pages"]) and row["pages"].strip():
            etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}pages").text = str(row["pages"])
        
        if "publication_place" in df.columns and pd.notna(row["publication_place"]) and row["publication_place"].strip():
            etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}publication_place").text = str(row["publication_place"])
        
        if "open_access" in df.columns and pd.notna(row["open_access"]) and row["open_access"].strip():
            etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}open_access").text = str(row["open_access"])
        
        if "source_number" in df.columns and pd.notna(row["source_number"]) and row["source_number"].strip():
            etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}source_number").text = str(row["source_number"])
        
        if "issue" in df.columns and pd.notna(row["issue"]) and row["issue"].strip():
            etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}issue").text = str(row["issue"])
        
        if "volume" in df.columns and pd.notna(row["volume"]) and row["volume"].strip():
            etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}volume").text = str(row["volume"])
    
    tree = etree.ElementTree(root)
    tree.write(output_xml, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"Dublin Core XML saved to {output_xml}")



# Wygenerowanie XML
df_to_dublin_core(df, "dublin_core_example.xml")


def transform_to_bibframe(input_xml, xslt_file, output_rdf):
    dom = etree.parse(input_xml)
    xslt = etree.parse(xslt_file)
    transform = etree.XSLT(xslt)
    bibframe = transform(dom)
    
    with open(output_rdf, "wb") as f:
        f.write(etree.tostring(bibframe, pretty_print=True, xml_declaration=True, encoding="UTF-8"))
    print(f"BIBFRAME RDF saved to {output_rdf}")

# Pliki wejściowe
input_xml = "dublin_core_example.xml"
xslt_file = "D:/Nowa_praca/bibframe/DC_to_bibframe.xsl"  # Zamień na ścieżkę do pliku XSLT
output_rdf = "output_bibframe.rdf"

# Transformacja
transform_to_bibframe(input_xml, xslt_file, output_rdf)

