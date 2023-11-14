import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
import tkinter as tk
from tkinter import filedialog, simpledialog, Toplevel, Label, Entry, Button, messagebox

class ColumnMappingDialog(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Column Mapping")
        self.mapping = {}
        skos_properties = [
            "concept",
            "prefLabel",
            "altLabel",
            "hiddenLabel",
            "definition",
            "notation",
            "broader",
            "narrower",
            "related",
            "inScheme",
            "hasTopConcept",
            "topConceptOf",
            "exactMatch",
            "closeMatch",
            "broadMatch",
            "narrowMatch",
            "relatedMatch",
            "example",
            "note",
            "editorialNote",
            "changeNote",
            "historyNote",
            "scopeNote",
            "semanticRelation",
            "mappingRelation",
            "broaderTransitive",
            "narrowerTransitive",
            "member",
            "memberList"
        ]
        self.entries = {}

        info_button = Button(self, text="instructions", command=self.show_info)
        info_button.pack()

        frame = tk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        entry_frame = tk.Canvas(frame, yscrollcommand=scrollbar.set)
        entry_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=entry_frame.yview)

        inner_frame = tk.Frame(entry_frame)
        entry_frame.create_window((0, 0), window=inner_frame, anchor=tk.N)

        for prop in skos_properties:
            prop_frame = tk.Frame(inner_frame)
            prop_frame.pack(fill=tk.X)
            label = Label(prop_frame, text=f"skos:{prop}")
            label.pack(side=tk.LEFT)
            entry = Entry(prop_frame)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
            self.entries[prop] = entry

        inner_frame.update_idletasks()  
        entry_frame.config(scrollregion=entry_frame.bbox("all"))  

        ok_button = Button(self, text="OK", command=self.ok)
        ok_button.pack()

    def show_info(self):
        info_text = (
      "1. Enter the column names from the Excel sheet for each SKOS property.\n"
      "2. Leave the field empty if there is no corresponding column for a SKOS property.\n"
      "3. For columns with multiple values in a single row (e.g., multiple altLabels in different languages), "
      "enter each value separately with comma in row. If you wish to add a language tag to a value, append it to the value, separated by an \"@\" symbol. For example: \"Literature @en, Literatura @es, Littérature @fr\" for different language versions of a label. If you don’t add a language tag, the default version without a language tag will be used.\n"
      
      "4. Ensure that the 'skos:concept' field is filled out as it is used to generate the URI for the concept.\n"
      "5. When specifying URIs in the Excel sheet (e.g., for skos:broadMatch, skos:narrowMatch, etc.), ensure they "
      "are complete and well-formed URIs.\n"
      "6. Click OK when done."
  )

        messagebox.showinfo("Instructions", info_text)

    def ok(self):
        for prop, entry in self.entries.items():
            column_name = entry.get().strip()
            if column_name:
                self.mapping[f'skos:{prop}'] = column_name
        self.destroy()

def load_excel_file():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    filepath = filedialog.askopenfilename(title="Select Excel file",
                                          filetypes=[("Excel files", "*.xlsx;*.xls")])
    root.attributes('-topmost', False)
    root.destroy()
    return pd.read_excel(filepath)

def get_column_mapping():
    root = tk.Tk()
    root.withdraw()
    dialog = ColumnMappingDialog(root)
    root.lift()
    root.attributes('-topmost', True)
    root.wait_window(dialog)
    root.attributes('-topmost', False)
    return dialog.mapping

def get_output_format():
    return simpledialog.askstring("Output Format", "Enter output format (xml, turtle, n3, nt, pretty-xml, trix, trig):")

def generate_rdf(df, column_mapping):
    g = Graph()
    skos = Namespace("http://www.w3.org/2004/02/skos/core#")
    for index, row in df.iterrows():
        concept_uri = URIRef(row[column_mapping['skos:concept']])
        for skos_property, column in column_mapping.items():
            if column in df.columns and pd.notna(row[column]) and column != column_mapping['skos:concept']:
                values = str(row[column]).split(',')
                for value in values:
                    value = value.strip()
                    lang_tag = None
                    if '@' in value:
                        value, lang_tag = value.rsplit('@', 1)
                    if skos_property in ['skos:broadMatch', 'skos:narrowMatch', 'skos:relatedMatch',
                                         'skos:closeMatch', 'skos:exactMatch']:
                        g.add((concept_uri, skos[skos_property.split(':')[1]], URIRef(value.strip())))
                    else:
                        if lang_tag:
                            g.add((concept_uri, skos[skos_property.split(':')[1]], Literal(value.strip(), lang=lang_tag)))
                        else:
                            g.add((concept_uri, skos[skos_property.split(':')[1]], Literal(value.strip())))
    return g

def save_rdf(g, format):
    filepath = filedialog.asksaveasfilename(title="Save RDF file",
                                            defaultextension=f".{format}",
                                            filetypes=[(f"{format.upper()} files", f"*.{format}")])
    g.serialize(destination=filepath, format=format)

def main():
    root = tk.Tk()
    root.withdraw()
    df = load_excel_file()
    column_mapping = get_column_mapping()
    output_format = get_output_format()
    g = generate_rdf(df, column_mapping)
    save_rdf(g, output_format)

if __name__ == "__main__":
    main()

