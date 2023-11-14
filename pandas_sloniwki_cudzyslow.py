# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:04:39 2023

@author: dariu
"""

# Since the user specified that all the values are in one column and provided more context,
# I will adjust the function to handle a single column of data where each cell can have multiple entries
# separated by commas, and we want to remove quotes only from the start and end of each entry.
import pandas as pd
from openpyxl import Workbook
# Simulate the user's data structure in a single DataFrame column
df = pd.read_excel ("C:/Users/dariu/genre.xlsx", sheet_name='to_rdf')

# Create a DataFrame


# Define a function to remove quotes from the start and end of each entry in the cell
def clean_cell_quotes(cell):
    # Check if the cell is a string, because only strings have the 'split' attribute
    if isinstance(cell, str):
        # Split the cell by comma to process each entry separately
        entries = cell.split(', ')
        cleaned_entries = []
        for entry in entries:
            # Strip quotes from both ends of each entry, leaving middle quotes intact
            cleaned_entry = entry.strip('\'"')
            cleaned_entries.append(cleaned_entry)
        # Rejoin the cleaned entries into a single string
        return ', '.join(cleaned_entries)
    else:
        # If the cell is not a string (e.g., NaN/None), return it as is
        return cell

# Apply the cleaning function to the DataFrame column
df['CleanedrelatedMatch'] = df['relatedMatch'].apply(clean_cell_quotes)
df['CleanedexactMatch'] = df['exactMatch'].apply(clean_cell_quotes)

# Display the original and cleaned data for comparison
df[['altLabel', 'Cleaned']]
df.to_excel("C:/Users/dariu/genre_corrected14112023.xlsx", engine='openpyxl', index=False)