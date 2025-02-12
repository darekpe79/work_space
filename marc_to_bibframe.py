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

