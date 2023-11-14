import polars as pl
import pandas as pd
import ast

# Krok 1: Załadowanie danych z pliku Excel
df = pl.read_excel("D:/Nowa_praca/słowniki_praca_fin_cze_sp_pl/10082023_matched_fi_cze_sp_pl_(broader_narrower-yso,cze,esp)_FINAL_with_broader_help (3).xlsx")
prefix = "http://example.org/concepts/"

# Using `apply` for DataFrame-wide application:
df = df.with_columns(
    (pl.lit(prefix) + pl.col("Concept")).alias("Concept_transform"))
def select_prefix(s):
    if s.startswith('n'):
        return "https://id.loc.gov/authorities/names/"
    elif s.startswith('sh'):
        return "https://id.loc.gov/authorities/subjects/"
    elif s.startswith('ph'):
        return "https://aleph.nkp.cz/F/?func=find-c&local_base=aut&ccl_term=ica="
    elif s.startswith('XX'):
        return "https://datos.bne.es/tema/"
    elif s.startswith('libri_thesauri'):
        return "http://example.org/concepts/"
    else:
        return ""  # Domyślny prefix

# Funkcja do przetwarzania kolumn
def transform_column(col_expr):
    # Przetwarzanie elementów listy z dodaniem odpowiedniego prefixu
    def transform_elements(cell):
        # Usuwamy nawiasy kwadratowe i konwertujemy na listę
        items = ast.literal_eval(cell)
        # Dodajemy prefix do każdego elementu listy
        return ','.join(select_prefix(item.strip()) + item.strip() for item in items)

    return col_expr.map_elements(transform_elements, return_dtype=pl.Utf8)

def transform_column2(col_expr, append_str=""):
    # Funkcja do przekształcania pojedynczej komórki
    def transform_elements(cell):
        # Usuwamy nawiasy kwadratowe i konwertujemy na listę
        items = ast.literal_eval(cell)
        # Łączymy elementy listy w jeden ciąg i dodajemy ciąg 'append_str' na końcu każdego elementu
        return ','.join(item.strip() + append_str for item in items)
    
    # Używamy funkcji map_elements do zastosowania transform_elements do każdego elementu w kolumnie
    return col_expr.map_elements(transform_elements, return_dtype=pl.Utf8)


# Przetwarzanie każdej z kolumn z danymi i dodawanie odpowiedniego prefixu
df = df.with_columns([
    transform_column(pl.col('exactMatch')).alias('exactMatch_transformed'),
    transform_column(pl.col('LOC_ID/exactMatch?close?')).alias('LOC_ID_exactMatch_close_transformed'),
    transform_column(pl.col('Fi_Id/exactMatch?close?')).alias('Fi_Id_exactMatch_close_transformed'),
    transform_column(pl.col('esp_ID_exactMatch?close?')).alias('esp_ID_exactMatch_close_transformed'),
    transform_column(pl.col('slownik_Hubar/exactMatch?close?')).alias('slownik_Hubar_exactMatch_close_transformed'),
    transform_column(pl.col('Cze_ID/exactMatch?close?')).alias('Cze_ID/exactMatch?close?_transformed'),
    transform_column(pl.col('narrowMatch')).alias('narrowMatch_transformed'),
    transform_column(pl.col('broadMatch')).alias('broadMatch_transformed'),
    # Repeat the same process for other columns as needed...
])

df = df.with_columns([
    transform_column2(pl.col('altLabel fin_a'), "@fi").alias('altLabel fin_a_transformed'),
    transform_column2(pl.col('altLabel cze_a'), '@cs').alias('altLabel cze_a_transformed'),
    transform_column2(pl.col('altLabel field_a_esp'), '@es').alias('altLabel field_a_esp_transformed'),
    transform_column2(pl.col('altLabel_pl'), '@pl').alias('altLabel_pl_transformed'),

    # Repeat the same process for other columns as needed...
])

# Dodanie połączonej kolumny 'polaczone'
df = df.with_columns(
    pl.concat_str(
        [
            pl.col('exactMatch_transformed').fill_null(""),
            pl.col('LOC_ID_exactMatch_close_transformed').fill_null(""),
            pl.col('Fi_Id_exactMatch_close_transformed').fill_null(""),
            pl.col('esp_ID_exactMatch_close_transformed').fill_null(""),
            pl.col('slownik_Hubar_exactMatch_close_transformed').fill_null(""),
            pl.col('Cze_ID/exactMatch?close?_transformed').fill_null("")
        ],
        separator=','
    ).alias('polaczone_exact')
)

df = df.with_columns(
    pl.concat_str(
        [
            pl.col('altLabel fin_a_transformed').fill_null(""),
            pl.col('altLabel cze_a_transformed').fill_null(""),
            pl.col('altLabel field_a_esp_transformed').fill_null(""),
            pl.col('altLabel_pl_transformed').fill_null(""),
            
        ],
        separator=','
    ).alias('polaczone_altlabel')
)
# Usunięcie wielokrotnych przecinków i pozostawienie pojedynczego
df = df.with_columns(
    pl.col('polaczone_exact').str.replace_all(r",+", ",").str.strip(', ').alias('polaczone_exact')
)
df = df.with_columns(
    pl.col('polaczone_altlabel').str.replace_all(r",+", ",").str.strip(', ').alias('polaczone_altlabel')
)

# Usunięcie kolumn tymczasowych
df = df.drop([
    'exactMatch_transformed',
    'LOC_ID_exactMatch_close_transformed',
    'Fi_Id_exactMatch_close_transformed',
    'esp_ID_exactMatch_close_transformed',
    'slownik_Hubar_exactMatch_close_transformed'
])

# Konwersja DataFrame'u Polars do pandas
pandas_df = df.to_pandas()


# Save the pandas DataFrame to an Excel file
pandas_df.to_excel("C:/Users/dariu/subject_corrected14112023.xlsx", engine='openpyxl', index=False)