# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:54:52 2025

@author: darek
"""




import pandas as pd
import json
from lxml import etree

def df_to_dublin_core_json(df, mapping_file, output_xml):
    """
    Z DataFrame tworzony jest XML Dublin Core
    wykorzystując mapowanie z pliku JSON.
    """
    # 1) Wczytujemy mapowanie
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
        # np. mapping = {
        #   "title": "dc:title",
        #   "link": "dc:relation",
        #   ...
        # }

    # 2) Tworzymy główny element <articles> z namespace DC
    nsmap = {"dc": "http://purl.org/dc/elements/1.1/"}
    root = etree.Element("articles", nsmap=nsmap)

    # 3) Iterujemy po każdym wierszu DataFrame
    for _, row in df.iterrows():
        article = etree.SubElement(root, "article")

        # 4) Sprawdzamy wszystkie kolumny DF
        for col in df.columns:
            value = row[col]
            # pomijamy None, NaN, puste stringi
            if pd.notna(value) and str(value).strip():
                # sprawdzamy, czy jest w mapowaniu
                if col in mapping:
                    # np. "dc:creator"
                    dc_field = mapping[col]  
                    # Rozbijamy "dc:creator" na prefix "dc" i lokalną nazwę "creator"
                    prefix, localname = dc_field.split(":")
                    # Tworzymy element w przestrzeni DC
                    etree.SubElement(article, "{http://purl.org/dc/elements/1.1/}" + localname).text = str(value)

                # jeśli kolumna nie jest w mapowaniu, ignorujemy

    # 5) Zapis do XML
    tree = etree.ElementTree(root)
    tree.write(output_xml, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"Dublin Core XML saved to {output_xml}")


def transform_to_bibframe(input_xml, xslt_file, output_rdf):
    """
    Transformuje DC XML na BIBFRAME RDF za pomocą pliku XSLT.
    """
    dom = etree.parse(input_xml)       # wczytujemy plik DC
    xslt = etree.parse(xslt_file)      # wczytujemy plik XSLT
    transform = etree.XSLT(xslt)       # tworzymy obiekt transformacji

    bibframe = transform(dom)          # wykonujemy transformację

    with open(output_rdf, "wb") as f:
        f.write(etree.tostring(bibframe, pretty_print=True, 
                               xml_declaration=True, encoding="UTF-8"))
    print(f"BIBFRAME RDF saved to {output_rdf}")


# ============ PRZYKŁADOWE UŻYCIE ============

if __name__ == "__main__":
    # Przykładowy DataFrame (mógłby być wczytany z Excela)
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

    # 1) Wygenerowanie pliku DC XML
    input_xml = "dublin_core_example.xml"
    mapping_file = "D:/Nowa_praca/bibframe/mapowanie_DC_Bib.json"     # plik z mapowaniem
    df_to_dublin_core_json(df, mapping_file, input_xml)

    # 2) Transformacja do BIBFRAME
    xslt_file = "D:/Nowa_praca/bibframe/DC_to_bibframe.xsl"
    output_rdf = "output_bibframe.rdf"
    transform_to_bibframe(input_xml, xslt_file, output_rdf)

