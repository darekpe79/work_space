# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:20:57 2023

@author: dariu
"""

import requests

def get_wikidata_items_by_labels(labels):
    url = "https://www.wikidata.org/w/api.php"
    
    results = {}
    
    for label in labels:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": label
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("search"):
            # Taking the first match for simplicity. You can modify this to handle multiple matches if needed.
            matched_item = data["search"][0]
            results[label] = {"ID": matched_item['id'], "Label": matched_item['label']}
        else:
            results[label] = {"ID": None, "Label": None}
    
    return results

# Example usage
labels = ["Albanian literature", "Polish literature"]
results = get_wikidata_items_by_labels(labels)

for label, result in results.items():
    print(f"Search Label: {label}, ID: {result['ID']}, Label: {result['Label']}")
    
    
    
import pandas as pd
import re

# Load the Excel file
file_path = 'D:/Nowa_praca/650_dokumenty/Slownik_ELB_G_N.xlsx'
df = pd.read_excel(file_path)

# Extract URLs from the 'desk_650' column
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
df['links'] = df['desk_650'].apply(lambda x: re.search(url_pattern, str(x)).group() if re.search(url_pattern, str(x)) else None)

# Remove URLs from the 'desk_650' column
df['desk_650'] = df['desk_650'].str.replace(url_pattern, '', regex=True)

# Save the modified dataframe to a new Excel file
output_file_path = 'Modified_Slownik_ELB_G_N.xlsx'
df.to_excel(output_file_path, index=False)



#%% comparing two excels 
import pandas as pd
import ast
from tqdm import tqdm

# Load the Excel files
df1 = pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/Modified_Slownik_ELB_G_N.xlsx')
df2 = pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/10082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze,esp)_FINAL_with_broader_help (1).xlsx')

# Convert the 'fi_id' column into a list
df2['fi_id'] = df2['fi_id'].apply(lambda x: ast.literal_eval(str(x)))

# Iterate through the rows of df1 and compare the links with df2
for index1, row1 in df1.iterrows():
    link1 = row1['links']
    for index2, row2 in df2.iterrows():
        if link1 in row2['fi_id']:
            for col in df2.columns:
                df1.at[index1, col] = row2[col]

# Save the modified dataframe to a new Excel file
df1.to_excel('Merged_Slownik_ELB_G_N.xlsx', index=False)

import pandas as pd

# Load the Excel file
xl = pd.ExcelFile('D:/Nowa_praca/650_dokumenty/all_650_new_karolina.xlsx')

# Read the 'fin' and 'bn_do_łączenia' sheets
fin_df = xl.parse('fin')
bn_do_łączenia_df = xl.parse('bn_do_łączenia')

# Merge the two dataframes on the 'desk_650' column
merged_df = bn_do_łączenia_df.merge(fin_df[['desk_650', 'genre']], on='desk_650', how='left')

# Rename the 'genre' column from 'fin' sheet to 'genre_from_fin'
merged_df.rename(columns={'genre': 'genre_from_fin'}, inplace=True)

# Save the updated 'bn_do_łączenia' sheet back to the Excel file
with pd.ExcelWriter('all_650_new_karolina_updated.xlsx') as writer:
    merged_df.to_excel(writer, sheet_name='bn_do_łączenia', index=False)
    # You can also save other sheets if needed
    fin_df.to_excel(writer, sheet_name='fin', index=False)

print("File 'all_650_new_karolina_updated.xlsx' has been saved with the updated data.")
#%% dzielenie wiadomosci z kolumn:
pattern=r'(?<=\$7).*?(?=\$|$)' 
patterna=r'(?<=\$a).*?(?=\$|$)' 

xl = pd.ExcelFile('D:/Nowa_praca/650_dokumenty/20102023all_650_new_karolina_spain_do_uzupełnienia_mrc.xlsx')

# Read the 'fin' and 'bn_do_łączenia' sheets
fin_df = xl.parse('fin')
czech_do_łączenia_df = xl.parse('czech_do_łaczenia')
fin_do_laczenia = xl.parse('fin_do_laczenia')
bn_do_łączenia=xl.parse('bn_do_łączenia')


czech_do_łączenia_df['links'] = czech_do_łączenia_df['desk_650'].apply(lambda x: re.search(pattern, str(x)).group() if re.search(pattern, str(x)) else None)
czech_do_łączenia_df['field650'] = czech_do_łączenia_df['desk_650'].apply(lambda x: re.search(patterna, str(x)).group() if re.search(patterna, str(x)) else None)

fin_do_laczenia['field650'] = fin_do_laczenia['desk_650'].apply(lambda x: re.search(patterna, str(x)).group() if re.search(patterna, str(x)) else None)
bn_do_łączenia['field650'] = bn_do_łączenia['desk_650'].apply(lambda x: re.search(patterna, str(x)).group() if re.search(patterna, str(x)) else None)
# Remove URLs from the 'desk_650' column
#czech_do_łączenia_df['desk_650'] = czech_do_łączenia_df['desk_650'].str.replace(pattern, '', regex=True)

with pd.ExcelWriter('D:/Nowa_praca/650_dokumenty/20102023all_650_new_karolina_spain_do_uzupełnienia_mrc.xlsx', engine='xlsxwriter') as writer:
    fin_df.to_excel(writer, sheet_name='fin', index=False)
    czech_do_łączenia_df.to_excel(writer, sheet_name='czech_do_łaczenia', index=False)
    fin_do_laczenia.to_excel(writer, sheet_name='fin_do_laczenia', index=False)
    bn_do_łączenia.to_excel(writer, sheet_name='bn_do_łączenia', index=False)

#vlookup
import pandas as pd
import ast
from tqdm import tqdm
xl = pd.ExcelFile('D:/Nowa_praca/650_dokumenty/20102023all_650_new_karolina_spain_do_uzupełnienia_mrc.xlsx')
czech_do_łączenia_df = xl.parse('czech_do_łaczenia')

df2 = pd.read_excel('D:/Nowa_praca/650_dokumenty/10082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze,esp)_FINAL_with_broader_help (2).xlsx')
#fin_do_laczenia = xl.parse('fin_do_laczenia')
# Convert the 'fi_id' column into a list
df2['Cze_ID/exactMatch?close?'] = df2['Cze_ID/exactMatch?close?'].apply(lambda x: ast.literal_eval(str(x)))

for index1, row1 in tqdm(czech_do_łączenia_df.iterrows()):
    link1 = row1['links']
    for index2, row2 in df2.iterrows():
        (row2['Cze_ID/exactMatch?close?'])
        if link1 in row2['Cze_ID/exactMatch?close?']:
            for col in df2.columns:
                czech_do_łączenia_df.at[index1, col] = row2[col]
czech_do_łączenia_df.to_excel('Merged_Czech.xlsx', index=False)

import pandas as pd
import ast
from tqdm import tqdm
xl = pd.ExcelFile('C:/Users/dariu/Merged_Sp.xlsx')
czech_do_łączenia_df = xl.parse('Sheet1')

df2 = pd.read_excel('D:/Nowa_praca/650_dokumenty/10082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze,esp)_FINAL_with_broader_help (2).xlsx')
#df2 = pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/12062023_odpytka_slownik_hiszpanski_loc_wiki_ver1.0.xlsx')

#fin_do_laczenia = xl.parse('fin_do_laczenia')
# Convert the 'fi_id' column into a list
df2['esp_ID_exactMatch?close?'] = df2['esp_ID_exactMatch?close?'].apply(lambda x: ast.literal_eval(str(x)))

for index1, row1 in tqdm(czech_do_łączenia_df.iterrows()):
    link1 = row1['sp_id']
    for index2, row2 in df2.iterrows():
        
        if link1 in row2['esp_ID_exactMatch?close?']:
            for col in df2.columns:
                czech_do_łączenia_df.at[index1, col] = row2[col]
czech_do_łączenia_df.to_excel('Merged_cze-pomoc.xlsx', index=False)
#inne podejscie:




import pandas as pd
import ast

xl = pd.ExcelFile('D:/Nowa_praca/650_dokumenty/20102023all_650_new_karolina_spain_do_uzupełnienia_mrc.xlsx')
czech_do_łączenia_df = xl.parse('czech_do_łaczenia')

df2 = pd.read_excel('D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/10082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze,esp)_FINAL_with_broader_help (1).xlsx')

# Convert the 'cze_id' column into a list
df2['cze_id'] = df2['cze_id'].apply(lambda x: ast.literal_eval(str(x)))

# Explode the 'cze_id' column so that each link has its own row
df2_exploded = df2.explode('cze_id')

# Usuwanie wierszy z pustymi komórkami w kolumnie 'cze_id' w df2_exploded
df2_exploded = df2_exploded[df2_exploded['cze_id'].notna()]

# Wyodrębnij wiersze z pustymi komórkami z 'czech_do_łączenia_df'
rows_with_na = czech_do_łączenia_df[czech_do_łączenia_df['links'].isna()]

# Usuń te wiersze z 'czech_do_łączenia_df'
czech_do_łączenia_df = czech_do_łączenia_df[czech_do_łączenia_df['links'].notna()]

# Merge the dataframes on the links
merged_df = czech_do_łączenia_df.merge(df2_exploded, left_on='links', right_on='cze_id', how='left')

# Drop the 'cze_id' column as it's now redundant
merged_df.drop('cze_id', axis=1, inplace=True)

# Dodaj wiersze z pustymi komórkami z powrotem do 'merged_df'
merged_df = pd.concat([merged_df, rows_with_na], ignore_index=True)

# The merged_df now contains the updated data
czech_do_łączenia_df = merged_df

czech_do_łączenia_df.to_excel('Merged_Czech2.xlsx', index=False)

#lematyzacje, porównania
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from difflib import SequenceMatcher

# Ensure you have the required NLTK data
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load the Excel file
df = pd.read_excel("C:/Users/dariu/31102023Modified_ratio2.xlsx")
df['all'] = df['all'].str.lower()
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to convert nltk tag to first character used by WordNetLemmatizer
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# Function to lemmatize a sentence
def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

# Lemmatize the 'all' column and store in 'lemmatize_all'
df['lemmatize_all'] = df['all'].apply(lambda x: lemmatize_sentence(str(x)))

# Calculate the sequence match ratio between 'lemmatize_all' and 'elb_concept_2'
df['ratio1_notlemattized'] = df.apply(lambda row: SequenceMatcher(None, str(row['all']).lower(), str(row['prefLabel']).lower()).ratio(), axis=1)

# Save the modified dataframe to a new Excel file
df.to_excel("31102023Modified_ratio2_ratio1.xlsx", index=False)


lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("running")
lemmatizer.lemmatize('running', 'v')
word, nltk_tag = 'running', 'VBG'

# Konwersja tagu nltk na format WordNet
wordnet_tag = nltk_tag_to_wordnet_tag(nltk_tag)

# Lematyzacja słowa z uwzględnieniem jego części mowy
lemmatized_word = lemmatizer.lemmatize(word, wordnet_tag)

print(lemmatized_word)
nltk.pos_tag(["running"])

#%%
# alt labels to one row


import pandas as pd

# Load the Excel file
df = pd.read_excel("C:/Users/dariu/31102023Modified_ratio2_ratio1.xlsx")

# Group by 'elb_concept_2' and aggregate unique 'field650' values

df['altLabel'] = df['altLabel'].str.strip(' .')
agg_data = df.groupby('prefLabel')['altLabel'].unique().reset_index()


# Rename the aggregated column to 'alt_label'
agg_data.rename(columns={'altLabel': 'altlabel_combined'}, inplace=True)

# Merge the aggregated data back to the original DataFrame
df = df.merge(agg_data, on='prefLabel', how='left')
print(df)
# Convert the 'alt_label' column from list to string format, filtering out non-string values
#df['alt_label'] = df['alt_label'].apply(lambda x: ', '.join([item for item in x if isinstance(item, str)]))

df['alt_label_combined'] = df['altlabel_combined'].apply(lambda x: str([item for item in x if isinstance(item, str)]))


# Save the modified DataFrame to a new Excel file
df.to_excel("Modified_Merged__with_Alt_Labels.....xlsx", index=False)


## gropu many columns

import pandas as pd
from ast import literal_eval

# Read the file
df = pd.read_excel('C:/Users/dariu/Modified_Merged__with_Alt_Labels.....xlsx')

def add_prefix(id_val):
    if str(id_val).startswith('ph'):
        return f"https://aleph.nkp.cz/F/?func=find-c&local_base=aut&ccl_term=ica={id_val}"
    elif str(id_val).startswith('XX'):
        return f"https://datos.bne.es/tema/{id_val}"
    else:
        return id_val

# Apply the custom function to the 'exactMatch' column
df['exactMatch'] = df['exactMatch'].apply(add_prefix)
# Convert string representation of list in 'exactMatch2' to actual list
def safe_literal_eval(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError):
        return []

# Convert string representation of list in 'exactMatch2' to actual list
df['exactMatch2'] = df['exactMatch2'].apply(safe_literal_eval)
# Group by 'elb_concept_2' and aggregate unique values from 'altLabel' and 'exactMatch2'
def flatten_and_unique(x):
    return list(set(item for sublist in x if isinstance(sublist, (list, tuple)) for item in sublist))

# ... [rest of the code remains the same]

# Group by 'elb_concept_2' and aggregate unique values from 'altLabel' and 'exactMatch2'
agg_data = df.groupby('prefLabel').agg({
    'exactMatch': lambda x: list(set(x)),
    'exactMatch2': flatten_and_unique
}).reset_index()


# Combine the unique values from 'altLabel' and 'exactMatch2' into a single list
agg_data['combinedexact'] = agg_data.apply(lambda row: row['exactMatch'] + row['exactMatch2'], axis=1)
import numpy as np
# Filter out non-string items
agg_data['combinedexact'] = agg_data['combinedexact'].apply(lambda x: [item for item in x if isinstance(item, str) and item == item] if x != [] else np.nan)


# Merge the combined list back to the original dataframe
df = df.merge(agg_data[['prefLabel', 'combinedexact']], on='prefLabel', how='left')

# Drop duplicates based on 'elb_concept_2'
#df = df.drop_duplicates(subset='elb_concept_2')

# Save to Excel
df.to_excel('agg_data_related_exact_with_original_columns.xlsx', index=False)


#%%polish ids
import pandas as pd

# Load the Excel file into a pandas DataFrame
file_path = 'C:/Users/dariu/Modified_Merged_Czech_with_Alt_Labels.....xlsx'
sheet_name = 'pl_concepts'
df = pd.read_excel(file_path, sheet_name=sheet_name)

df['field650'] = df['field650'].str.strip(' .')
unique_values = df['field650'].unique()
unique_list = unique_values.tolist()

#get ids from api

subject_id_dict = {}

for term in tqdm(unique_list):
    # Replace spaces with %20 to ensure the URL is formatted correctly
    formatted_term = term.replace(' ', '%20')
   # formatted_term='Literatura%20elektroniczna'
    url = f"https://data.bn.org.pl/api/institutions/authorities.json?subject={formatted_term}"
    
    response = requests.get(url)
    response_data = response.json()

    # Flag to indicate if the term was found
    term_found = False
    
    # Check if the response contains any authorities
    if response_data['authorities']:
        for authority in response_data['authorities']:
            # Iterate through the fields array
            for field in authority.get('marc', {}).get('fields', []):
                # Check if the field object has the key 155
                if '155' in field:
                    # Iterate through the subfields array
                    for subfield in field['155'].get('subfields', []):
                        # Check if the subfield object has the key a
                        if 'a' in subfield:
                            # Access field 155 subfield a and convert it to lowercase
                            field_155a = subfield['a'].lower()
                            print(field_155a)
            
            # Compare with the term (case-insensitive)
                            if field_155a == term.lower():
                                subject_id_dict[term] = 'https://data.bn.org.pl/api/institutions/authorities.json?id='+ str(authority['id'])
                                term_found = True
                                break  # Exit loop once a match is found
    
    # If the term was not found, set 'No ID found'
    if not term_found:
        subject_id_dict[term] = 'No ID found'

# Print the dictionary
print(subject_id_dict)
# Get ids from file:
    
    
from pymarc import MARCReader



# Dictionary to store the terms and corresponding values from the first 035 field
term_value_dict = {}

# Open your MARC file
with open('D:/Nowa_praca/650_dokumenty/authorities-all (1).marc', 'rb') as fh:
    reader = MARCReader(fh)
    for record in tqdm(reader):
        # Access all 155 fields
        fields_155 = record.get_fields('155')
        for field_155 in fields_155:
            # Access subfield a
            subfield_a = field_155.get_subfields('a')
            if subfield_a:
                subfield_a_lower = subfield_a[0].lower()  # Assuming there's at least one subfield a
                for term in unique_list:
                    if subfield_a_lower == term.lower():
                        # If a match is found, get the value from the first 035 field
                        field_035 = record['035']
                        if field_035:
                            value_035 = field_035['a'] if 'a' in field_035 else None
                            if value_035:
                                # Store the original term and value in your dictionary
                                term_value_dict[term] = value_035
df = pd.DataFrame.from_dict(term_value_dict, orient='index', columns=['Value1'])
df.to_excel('ids_polish.xlsx')

#combined polish links with other in pandas:
import pandas as pd

# Load the Excel file
df = pd.read_excel('C:/Users/dariu/Modified_Merged_Czech_with_Alt_Labels.....xlsx')


import ast  # Use ast.literal_eval instead of eval for safer evaluation of string representations
def aggregate_unique_non_na(series):
    # This function aggregates unique non-na values into a list
    return [x for x in series.unique() if pd.notna(x)]  # Filter out nan values

# Group by 'prefLabel' and aggregate unique values in 'related_match_polish' into lists
agg_data = df.groupby('prefLabel')['related_match_polish'].apply(aggregate_unique_non_na).reset_index()


# Group by 'prefLabel' and aggregate unique values in 'related_match_polish' into lists


# Merge the aggregated data back to the original DataFrame
df = pd.merge(df, agg_data, on='prefLabel', suffixes=('', '_aggregated'))

def combine_links(row):
    # Convert the string representation of a list to an actual list
    # Use ast.literal_eval for safer evaluation of string representations
    combined2_related_list = ast.literal_eval(row['combined2_related']) if row['combined2_related'] else []
    related_match_polish_list = row['related_match_polish_aggregated']  # This is already a list
    if related_match_polish_list:  # Check for non-NA/null values using an if statement
    # Extend 'combined2_related' list with the 'related_match_polish' list
        combined2_related_list.extend(related_match_polish_list)

    # Convert the list back to a string representation
    return str(combined2_related_list)

# Apply the function row-wise
df['combined2_related'] = df.apply(combine_links, axis=1)

# Drop the extra 'related_match_polish_aggregated' column
df = df.drop(columns=['related_match_polish_aggregated'])
##SAME FOR EXact
agg_data = df.groupby('prefLabel')['exact_polish'].apply(aggregate_unique_non_na).reset_index()


# Merge the aggregated data back to the original DataFrame
df = pd.merge(df, agg_data, on='prefLabel', suffixes=('', '_aggregated'))

def combine_links(row):
    # Convert the string representation of a list to an actual list
    # Use ast.literal_eval for safer evaluation of string representations
    combined_exact = ast.literal_eval(row['combined_exact']) if row['combined_exact'] else []
    exact_polish_list = row['exact_polish_aggregated']  # This is already a list
    if exact_polish_list:  # Check for non-NA/null values using an if statement
    # Extend 'combined2_related' list with the 'related_match_polish' list
        combined_exact.extend(exact_polish_list)

    # Convert the list back to a string representation
    return str(combined_exact)

# Apply the function row-wise
df['combined_exact'] = df.apply(combine_links, axis=1)

# Drop the extra 'related_match_polish_aggregated' column
df = df.drop(columns=['exact_polish_aggregated'])

# Save the modified DataFrame back to an Excel file
df.to_excel('modified_file.xlsx', index=False)

#%% RDF making





# Iterate through the DataFrame rows and add the concepts to the graph
import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import SKOS
import ast
import hashlib

# Function to create a hash from a string
def create_hash(string):
    return hashlib.md5(string.encode()).hexdigest()

# Load the Excel file into a pandas DataFrame
file_path = 'C:/Users/dariu/agg_data_related_exact_with_original_columns.xlsx'
df = pd.read_excel(file_path)

# Create the SKOS graph
g = Graph()

# Define the namespace for your SKOS dictionary
namespace = Namespace('http://example.org/skos/')

# Iterate through the DataFrame rows and add the concepts to the graph
for index, row in df.iterrows():
    concept = URIRef(namespace + row['concept'])
    label = Literal(row['prefLabel'])
    
    g.add((concept, SKOS.prefLabel, label))
    
    # Convert the string representation of the lists into actual lists
    related_matches = ast.literal_eval(row['relatedMatch'])
    exact_matches = ast.literal_eval(row['exactMatch'])
    alt_labels = ast.literal_eval(row['altLabel'])
    
    # Add the related matches to the graph
    for related_match in related_matches:
        related_match_encoded = related_match.replace(" ", "")
        related_concept = URIRef(related_match_encoded)
        g.add((concept, SKOS.relatedMatch, related_concept))
    
    # Add the exact matches to the graph
    for exact_match in exact_matches:
        exact_match_encoded = exact_match.replace(" ", "")
        exact_concept = URIRef(exact_match_encoded)
        g.add((concept, SKOS.closeMatch, exact_concept))
    
    # Add the alternative labels to the graph
    for alt_label in alt_labels:
        alt_label_literal = Literal(alt_label)
        g.add((concept, SKOS.altLabel, alt_label_literal))

# Serialize the graph to a file
g.serialize('skos_dictionary_genre.rdf', format='xml')

#bigger rdf
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal

# Wczytaj arkusz kalkulacyjny Excela do ramki danych pandas
df = pd.read_excel('C:/Users/dariu/major_genre.xlsx')

# Utwórz nowy graf RDF
g = Graph()

# Zdefiniuj przestrzeń nazw SKOS
skos = Namespace("http://www.w3.org/2004/02/skos/core#")

# Iteruj przez każdy wiersz w ramce danych
for index, row in df.iterrows():
    # Utwórz URI dla konceptu na podstawie wartości w kolumnie skos:concept
    concept_uri = URIRef(f"http://example.org/concepts/{row['skos:concept']}")
    for column in df.columns:
        # Sprawdź, czy komórka zawiera dane, a także czy kolumna nie jest skos:concept
        if pd.notna(row[column]) and column != 'skos:concept':
            # Rozdziel dane w komórce po przecinku
            values = str(row[column]).split(',')
            for value in values:
                value = value.strip()
                # Dodaj trójki do grafu w zależności od nazwy kolumny
                if column == 'skos:prefLabel':
                    g.add((concept_uri, skos.prefLabel, Literal(value)))
                elif column == 'skos:altLabel':
                    g.add((concept_uri, skos.altLabel, Literal(value)))
                elif column == 'skos:broader':
                    g.add((concept_uri, skos.broader, Literal(value)))
                elif column == 'skos:narrower':
                    g.add((concept_uri, skos.narrower, Literal(value)))
                elif column == 'skos:related':
                    g.add((concept_uri, skos.related, Literal(value)))
                elif column == 'skos:broadMatch':
                    g.add((concept_uri, skos.broadMatch, Literal(value)))
                elif column == 'skos:narrowMatch':
                    g.add((concept_uri, skos.narrowMatch, URIRef(value)))
                elif column == 'skos:relatedMatch':
                    g.add((concept_uri, skos.relatedMatch, URIRef(value)))
                elif column == 'skos:closeMatch':
                    g.add((concept_uri, skos.closeMatch, URIRef(value)))
                elif column == 'skos:exactMatch':
                    g.add((concept_uri, skos.exactMatch, URIRef(value)))

# Zapisz graf do pliku RDF
g.serialize(destination='outputXML.rdf', format='xml')



#translate words once again:
from deep_translator import GoogleTranslator
import pandas as pd
from tqdm import tqdm
    
file_path = 'C:/Users/dariu/modified_file.xlsx'
df = pd.read_excel(file_path)
lista=df['alt_label'].tolist()


def translate_to_english(words_list):
    translated_dict = {}
    for word in tqdm(words_list):
        try:
            translation = GoogleTranslator(source='auto', target='en').translate(word)
            translated_dict[word] = translation
        except Exception as e:
            print(f"Could not translate {word}: {e}")
            translated_dict[word] = None  # or some other value indicating failure
    return translated_dict

english_words = translate_to_english(lista)


df = pd.DataFrame.from_dict(english_words, orient='index', columns=['Value1'])
df.to_excel('tlumaczenie_labele.xlsx')
