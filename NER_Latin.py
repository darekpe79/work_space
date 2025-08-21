# # -*- coding: utf-8 -*-
# """
# Created on Mon Aug  4 08:57:46 2025

# @author: darek
# """

# import os
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# from tqdm import tqdm
# # Ścieżka do katalogu z plikami
# folder_path = "C:/pdf_llm_do_roboty/single_pages_simple_text-20250804T064354Z-1-001/single_pages_simple_text/"  # <- Zmień to!

# # Ładowanie modelu NER
# model_name = "magistermilitum/roberta-multilingual-medieval-ner"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# # Lista na wyniki
# results = []

# # Lista plików
# file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

# # Pasek postępu
# for filename in tqdm(file_list, desc="Przetwarzanie plików"):
#     filepath = os.path.join(folder_path, filename)
#     with open(filepath, 'r', encoding='utf-8') as f:
#         text = f.read()
#         ner_results = ner_pipeline(text)
#         for ent in ner_results:
#             results.append({
#                 "plik": filename,
#                 "tekst": ent['word'],
#                 "ent": ent['entity_group'],
#                 "score": round(ent['score'], 3)
#             })

# # Tworzenie DataFrame
# df = pd.DataFrame(results)
# excel_output_path = "ner_wyniki_monumenta.xlsx"
# df.to_excel(excel_output_path, index=False)


# import fitz  # PyMuPDF
# import nltk
# import torch
# import pandas as pd
# from tqdm import tqdm
# from transformers import pipeline

# # === KONIEC Z BŁĘDEM punkt_tab ===
# from nltk.tokenize import sent_tokenize
# nltk.download('punkt')

# # === MODEL ===
# pipe = pipeline(
#     "token-classification",
#     model="magistermilitum/roberta-multilingual-medieval-ner",
#     device=0 if torch.cuda.is_available() else -1
# )

# # === FUNKCJA DO AGREGACJI TOKENÓW ===
# def merge_entities(entities):
#     merged = []
#     current_word = ''
#     current_label = None
#     current_start = None
#     current_end = None

#     for ent in entities:
#         word = ent['word']
#         if word.startswith("▁"):
#             if current_word:
#                 merged.append((current_word, current_label, current_start, current_end))
#             word = word[1:]
#             current_word = word
#             current_label = ent['entity']
#             current_start = ent['start']
#             current_end = ent['end']
#         else:
#             current_word += word
#             current_end = ent['end']

#     if current_word:
#         merged.append((current_word, current_label, current_start, current_end))

#     return merged

# # === ŚCIEŻKI I ZAKRES STRON ===
# pdf_path = "C:/Users/darek/Downloads/Monumenta_Peruana_1.pdf"  # ← Zmień jeśli inna nazwa
# start_page = 77  # strona 78
# end_page = 97    # strona 97 włącznie

# # === GŁÓWNA LISTA WYNIKÓW ===
# results = []

# # === EKSTRAKCJA TEKSTU I ANALIZA NER ===
# with fitz.open(pdf_path) as doc:
#     for i in tqdm(range(start_page, end_page)):
#         page = doc[i]
#         text = page.get_text()
#         sentences = sent_tokenize(text)

#         for sent in sentences:
#             try:
#                 ents = pipe(sent)
#                 merged = merge_entities(ents)
#                 for word, label, start, end in merged:
#                     results.append({
#                         "page": i + 1,
#                         "sentence": sent,
#                         "entity": word,
#                         "label": label
#                     })
#             except Exception as e:
#                 print(f"Błąd na stronie {i+1}: {e}")

# # === ZAPIS DO EXCELA ===
# df = pd.DataFrame(results)
# df.to_excel("Monumenta_Peruana_NER_78-97.xlsx", index=False)
# print("✅ Zapisano do: Monumenta_Peruana_NER_78-97.xlsx")

# from transformers import pipeline, AutoTokenizer
# import torch
# import pandas as pd

# model_name = "magistermilitum/roberta-multilingual-medieval-ner"
# ner_pipeline = pipeline(
#     "ner",
#     model=model_name,
#     tokenizer=AutoTokenizer.from_pretrained(model_name),
#     aggregation_strategy="simple",
#     device=0 if torch.cuda.is_available() else -1
# )

# text = """78 P. H1ERONYMUS RUIZ DE PORTILLO P. FRANCISCO BORGIAE
# Generated through HathiTrust on 2025-07-31 02:18 GMT
# https://hdl.handle.net/2027/mdp.39015O12899749 / Creative Commons Attribution-NonCommercial-NoDerivatives
# 3
# PATER HIERONYMUS RUIZ DE PORTILLO PATRI FRANCISCO BORGIAE, VIC. GEN.
# Methymna Campi 7 Augusti 1565 — Romam
# Ex autogr. in cod. Hisp. 102, ff. 232-233v. (prius 186).
# Praefatio: Haec est prima ex lis quas nunc edimus epistolis Hieronymi Ruiz de Portillo, viri optime de missione peruana meriti. Natus Lucroni, calagurritanae dioeceseos, a. 1532 vel 1533, studiis Grammaticae et Artium absolutis Lucroni et Salmanticae, duode-vicennis Societati initiatus est a. 1551 Salmanticae, a P. doctore Torres. Novitiatu expleto Methymnae Campi et curriculo Phiioso-phiae, in salmanticensi athenaeo Theoiogiae operam navavit aa. 1553 et 54. Adductus in domum probationis septimancensem (Simancas), ibi rector fuit et simul magister novitiorum annis 1555-59. Die 11 lunii 1556, cum sollemnis celebraretur sacra actio Yallisoleti, in aula regia, adstantibus principe Carolo, regina gubematrice loanna, plerisąue optimatibus, et verba faciente P. Araoz, vota tria coadiu-torum spiritualium Portillo dixit. PCo VI 40; Alcazar, Chro no historia, 1296. Cum autem Comes de Nieva ad Peruam prorex profectu-tus esset a. 1559, rogavit missionarios Societatis sibi adiutores; tunc Borgia Patri Lainez, generali, sic recensens destinatos: «E1 segundo es el Padre Portillo, rector de la probación de Simancas, que en todas estas partes ya dichas es no menos quallficado que el doctor Rodriguez, y tiene buen talento de predicar » SFB III 501.
# Exinde vero a. 1560 factus est rector vallisoletani collegii S. Antonii, quo relicto, a. 1562 regimen domus methymnensis suscepit, ubi sollemnem emisit professionem die 19 Februarii 1565. A postre-mo Maio eiusdem anni, durante Congregatione generali, Provin-ciam castellanam ut Viceprovincialis rexit, posteaquam iam Indias petendi desiderium ostendisset, prout eruitur ex litteris Borgiae, adhuc vicarii generalis, 12 Mai 1565 : «El primero, que sera superior de los otros [missionariorum /loridensium], sera el P. Portillo, porque ultra de sus antiguos deseos, es professo, como por las ulti-mas letras entendemos ». MAF 9. Re quidem vera Portillo mis-siones exteras Borgiae petivit, Salmantica 26 Augusti 1565, et tem-pus ad se praeparandum virtute, oratione, mortiflcatione et studio. Arch. Prov. Tol., Astr. 40. Missionarius porro erat electus pro Florida.
# Interim tamen praedicationi evangelicae incumbebat; de eo enim in memoria rerum collegii methymnensis a. 1566 dicitur : « En este ano partió de aqui el Padre Gerónimo de Portillo por predica-dor a Valladolid, y desde pocos dias fuć embiado por Provincial del Perń, y fuć el primero que allf passó eon Don Francisco de Toledo, virrey. Ha hecho y hace grandę fructo en aąuellas partes et adhuc vi-vit. (Archiu. Prov. Tol. fasc. 826, Relación de lo que haclan en Medina del Campo, t. 1) Ubi tamen corrigendom est eum comitem fuisse iti-neris proregis Toledo.
# □ riginal from
# UNIYERSITY OF MICHIGAN"""
# aggregated = ner_pipeline(text)

# # Jako DataFrame
# df = pd.DataFrame([
#     (e["word"], e["entity_group"], e["start"], e["end"])
#     for e in aggregated
# ], columns=["text", "label", "start", "end"])

# print(df)

# from transformers import pipeline, AutoTokenizer
# import torch
# import pandas as pd

# # Ustawienia modelu i tokenizera
# model_name = "magistermilitum/roberta-multilingual-medieval-ner"
# pipe = pipeline("token-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Przykładowy tekst
# text = """78 P. H1ERONYMUS RUIZ DE PORTILLO P. FRANCISCO BORGIAE
# Generated through HathiTrust on 2025-07-31 02:18 GMT
# https://hdl.handle.net/2027/mdp.39015O12899749 / Creative Commons Attribution-NonCommercial-NoDerivatives
# 3
# PATER HIERONYMUS RUIZ DE PORTILLO PATRI FRANCISCO BORGIAE, VIC. GEN.
# Methymna Campi 7 Augusti 1565 — Romam
# Ex autogr. in cod. Hisp. 102, ff. 232-233v. (prius 186).
# Praefatio: Haec est prima ex lis quas nunc edimus epistolis Hieronymi Ruiz de Portillo, viri optime de missione peruana meriti. Natus Lucroni, calagurritanae dioeceseos, a. 1532 vel 1533, studiis Grammaticae et Artium absolutis Lucroni et Salmanticae, duode-vicennis Societati initiatus est a. 1551 Salmanticae, a P. doctore Torres. Novitiatu expleto Methymnae Campi et curriculo Phiioso-phiae, in salmanticensi athenaeo Theoiogiae operam navavit aa. 1553 et 54. Adductus in domum probationis septimancensem (Simancas), ibi rector fuit et simul magister novitiorum annis 1555-59. Die 11 lunii 1556, cum sollemnis celebraretur sacra actio Yallisoleti, in aula regia, adstantibus principe Carolo, regina gubematrice loanna, plerisąue optimatibus, et verba faciente P. Araoz, vota tria coadiu-torum spiritualium Portillo dixit. PCo VI 40; Alcazar, Chro no historia, 1296. Cum autem Comes de Nieva ad Peruam prorex profectu-tus esset a. 1559, rogavit missionarios Societatis sibi adiutores; tunc Borgia Patri Lainez, generali, sic recensens destinatos: «E1 segundo es el Padre Portillo, rector de la probación de Simancas, que en todas estas partes ya dichas es no menos quallficado que el doctor Rodriguez, y tiene buen talento de predicar » SFB III 501.
# Exinde vero a. 1560 factus est rector vallisoletani collegii S. Antonii, quo relicto, a. 1562 regimen domus methymnensis suscepit, ubi sollemnem emisit professionem die 19 Februarii 1565. A postre-mo Maio eiusdem anni, durante Congregatione generali, Provin-ciam castellanam ut Viceprovincialis rexit, posteaquam iam Indias petendi desiderium ostendisset, prout eruitur ex litteris Borgiae, adhuc vicarii generalis, 12 Mai 1565 : «El primero, que sera superior de los otros [missionariorum /loridensium], sera el P. Portillo, porque ultra de sus antiguos deseos, es professo, como por las ulti-mas letras entendemos ». MAF 9. Re quidem vera Portillo mis-siones exteras Borgiae petivit, Salmantica 26 Augusti 1565, et tem-pus ad se praeparandum virtute, oratione, mortiflcatione et studio. Arch. Prov. Tol., Astr. 40. Missionarius porro erat electus pro Florida.
# Interim tamen praedicationi evangelicae incumbebat; de eo enim in memoria rerum collegii methymnensis a. 1566 dicitur : « En este ano partió de aqui el Padre Gerónimo de Portillo por predica-dor a Valladolid, y desde pocos dias fuć embiado por Provincial del Perń, y fuć el primero que allf passó eon Don Francisco de Toledo, virrey. Ha hecho y hace grandę fructo en aąuellas partes et adhuc vi-vit. (Archiu. Prov. Tol. fasc. 826, Relación de lo que haclan en Medina del Campo, t. 1) Ubi tamen corrigendom est eum comitem fuisse iti-neris proregis Toledo.
# □ riginal from
# UNIYERSITY OF MICHIGAN"""

# # Token-classification (daje wyniki token po tokenie)
# entities = pipe(text)

# # Funkcja do ręcznego łączenia bytów BIO
# def merge_entities_bio(entities):
#     merged = []
#     current_tokens = []
#     current_label = None
#     current_start = None
#     current_end = None

#     for ent in entities:
#         word = ent["word"].lstrip("▁Ġ#")
#         label = ent["entity"]
#         start = ent["start"]
#         end = ent["end"]

#         if "-" in label:
#             prefix, typ = label.split("-")
#         else:
#             prefix, typ = "O", label

#         if prefix == "B":
#             if current_tokens:
#                 merged.append((" ".join(current_tokens), current_label, current_start, current_end))
#             current_tokens = [word]
#             current_label = typ
#             current_start = start
#             current_end = end

#         elif prefix in {"I", "L"} and current_label == typ:
#             current_tokens.append(word)
#             current_end = end
#             if prefix == "L":
#                 merged.append((" ".join(current_tokens), current_label, current_start, current_end))
#                 current_tokens = []
#                 current_label = None

#         else:
#             if current_tokens:
#                 merged.append((" ".join(current_tokens), current_label, current_start, current_end))
#             if prefix != "O":
#                 current_tokens = [word]
#                 current_label = typ
#                 current_start = start
#                 current_end = end
#             else:
#                 current_tokens = []
#                 current_label = None

#     if current_tokens:
#         merged.append((" ".join(current_tokens), current_label, current_start, current_end))

#     return merged

# # Wykonanie łączenia
# merged_entities_manual = merge_entities_bio(entities)

# # Podgląd jako DataFrame
# df_manual = pd.DataFrame(merged_entities_manual, columns=["text", "label", "start", "end"])
# print("🔹 Wynik z merge_entities_bio():")
# print(df_manual)


# #%%
# import torch
# from transformers import pipeline
# import re
# import pandas as pd

# class TextProcessorFromPages:
#     def __init__(self, pages):
#         self.pages = pages  # <- lista stringów, np. ['strona 1...', 'strona 2...']
#         self.new_sentences = []
#         self.results = []
#         self.new_sentences_token_info = []
#         self.new_sentences_bio = []
#         self.BIO_TAGS = []
#         self.stripped_BIO_TAGS = []

#     def process_pages(self):
#         # Rozbij na "pseudo-zdania", np. fragmenty po 250 tokenów
#         for page in self.pages:
#             chunks = re.split(r'(?<=[.?!])\s+', page.strip())
#             temp = ""
#             for chunk in chunks:
#                 if len(temp.split()) + len(chunk.split()) < 250:
#                     temp += " " + chunk
#                 else:
#                     self.new_sentences.append(temp.strip())
#                     temp = chunk
#             if temp:
#                 self.new_sentences.append(temp.strip())

#     def apply_model(self, pipe):
#         self.results = list(map(pipe, self.new_sentences))
#         self.results = [[[y["entity"], y["word"], y["start"], y["end"]] for y in x] for x in self.results]

#     def tokenize_sentences(self):
#         for n_s in self.new_sentences:
#             tokens = n_s.split()
#             token_info = []
#             char_index = 0
#             for token in tokens:
#                 start = char_index
#                 end = char_index + len(token)
#                 token_info.append((token, start, end))
#                 char_index += len(token) + 1
#             self.new_sentences_token_info.append(token_info)

#     def process_results(self):
#         for result in self.results:
#             merged_bio_result = []
#             current_word = ""
#             current_label = None
#             current_start = None
#             current_end = None
#             for entity, subword, start, end in result:
#                 if subword.startswith("▁"):
#                     subword = subword[1:]
#                     merged_bio_result.append([current_word, current_label, current_start, current_end])
#                     current_word = "" ; current_label = None ; current_start = None ; current_end = None
#                 if current_start is None:
#                     current_word = subword
#                     current_label = entity
#                     current_start = start+1
#                     current_end = end
#                 else:
#                     current_word += subword
#                     current_end = end
#             if current_word:
#                 merged_bio_result.append([current_word, current_label, current_start, current_end])
#             self.new_sentences_bio.append(merged_bio_result[1:])

#     def match_tokens_with_entities(self):
#         for i, ss in enumerate(self.new_sentences_token_info):
#             for word in ss:
#                 for ent in self.new_sentences_bio[i]:
#                     if word[1] == ent[2]:
#                         if ent[1] == "L-PERS":
#                             self.BIO_TAGS.append([word[0], "I-PERS", "B-LOC"])
#                             break
#                         else:
#                             if "LOC" in ent[1]:
#                                 self.BIO_TAGS.append([word[0], "O", ent[1]])
#                             else:
#                                 self.BIO_TAGS.append([word[0], ent[1], "O"])
#                             break
#                 else:
#                     self.BIO_TAGS.append([word[0], "O", "O"])

#     def separate_dots_and_comma(self):
#         signs = [",", ";", ":", "."]
#         for bio in self.BIO_TAGS:
#             if any(bio[0][-1] == sign for sign in signs) and len(bio[0]) > 1:
#                 self.stripped_BIO_TAGS.append([bio[0][:-1], bio[1], bio[2]])
#                 self.stripped_BIO_TAGS.append([bio[0][-1], "O", "O"])
#             else:
#                 self.stripped_BIO_TAGS.append(bio)

#     def save_BIO_to_csv(self, filename="output.csv"):
#         df = pd.DataFrame(self.stripped_BIO_TAGS, columns=["TOKEN", "PERS", "LOCS"])
#         df.to_csv(filename, index=False)


# from transformers import AutoTokenizer

# model_name = "magistermilitum/roberta-multilingual-medieval-ner"
# pipe = pipeline("token-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)



# # Przygotuj testową stronę jako listę
# # pages = ["""
# # 78 P. H1ERONYMUS RUIZ DE PORTILLO P. FRANCISCO BORGIAE
# # Generated through HathiTrust on 2025-07-31 02:18 GMT
# # https://hdl.handle.net/2027/mdp.39015O12899749 / Creative Commons Attribution-NonCommercial-NoDerivatives
# # 3
# # PATER HIERONYMUS RUIZ DE PORTILLO PATRI FRANCISCO BORGIAE, VIC. GEN.
# # Methymna Campi 7 Augusti 1565 — Romam
# # Ex autogr. in cod. Hisp. 102, ff. 232-233v. (prius 186).
# # Praefatio: Haec est prima ex lis quas nunc edimus epistolis Hieronymi Ruiz de Portillo, viri optime de missione peruana meriti. Natus Lucroni, calagurritanae dioeceseos, a. 1532 vel 1533, studiis Grammaticae et Artium absolutis Lucroni et Salmanticae, duode-vicennis Societati initiatus est a. 1551 Salmanticae, a P. doctore Torres. Novitiatu expleto Methymnae Campi et curriculo Phiioso-phiae, in salmanticensi athenaeo Theoiogiae operam navavit aa. 1553 et 54. Adductus in domum probationis septimancensem (Simancas), ibi rector fuit et simul magister novitiorum annis 1555-59. Die 11 lunii 1556, cum sollemnis celebraretur sacra actio Yallisoleti, in aula regia, adstantibus principe Carolo, regina gubematrice loanna, plerisąue optimatibus, et verba faciente P. Araoz, vota tria coadiu-torum spiritualium Portillo dixit. PCo VI 40; Alcazar, Chro no historia, 1296. Cum autem Comes de Nieva ad Peruam prorex profectu-tus esset a. 1559, rogavit missionarios Societatis sibi adiutores; tunc Borgia Patri Lainez, generali, sic recensens destinatos: «E1 segundo es el Padre Portillo, rector de la probación de Simancas, que en todas estas partes ya dichas es no menos quallficado que el doctor Rodriguez, y tiene buen talento de predicar » SFB III 501.
# # Exinde vero a. 1560 factus est rector vallisoletani collegii S. Antonii, quo relicto, a. 1562 regimen domus methymnensis suscepit, ubi sollemnem emisit professionem die 19 Februarii 1565. A postre-mo Maio eiusdem anni, durante Congregatione generali, Provin-ciam castellanam ut Viceprovincialis rexit, posteaquam iam Indias petendi desiderium ostendisset, prout eruitur ex litteris Borgiae, adhuc vicarii generalis, 12 Mai 1565 : «El primero, que sera superior de los otros [missionariorum /loridensium], sera el P. Portillo, porque ultra de sus antiguos deseos, es professo, como por las ulti-mas letras entendemos ». MAF 9. Re quidem vera Portillo mis-siones exteras Borgiae petivit, Salmantica 26 Augusti 1565, et tem-pus ad se praeparandum virtute, oratione, mortiflcatione et studio. Arch. Prov. Tol., Astr. 40. Missionarius porro erat electus pro Florida.
# # Interim tamen praedicationi evangelicae incumbebat; de eo enim in memoria rerum collegii methymnensis a. 1566 dicitur : « En este ano partió de aqui el Padre Gerónimo de Portillo por predica-dor a Valladolid, y desde pocos dias fuć embiado por Provincial del Perń, y fuć el primero que allf passó eon Don Francisco de Toledo, virrey. Ha hecho y hace grandę fructo en aąuellas partes et adhuc vi-vit. (Archiu. Prov. Tol. fasc. 826, Relación de lo que haclan en Medina del Campo, t. 1) Ubi tamen corrigendom est eum comitem fuisse iti-neris proregis Toledo.
# # □ riginal from
# # UNIYERSITY OF MICHIGAN
# # """]

# # Przetwarzanie
# processor = TextProcessorFromPages(pages)
# processor.process_pages()
# processor.apply_model(pipe)
# processor.tokenize_sentences()
# processor.process_results()
# processor.match_tokens_with_entities()
# processor.separate_dots_and_comma()
# processor.save_BIO_to_csv("test_output.csv")


# import pandas as pd
# from pathlib import Path

# # ---------- parametry ----------
# # ścieżka do pliku wygenerowanego wcześniej przez TextProcessorFromPages
# SOURCE_CSV = Path(r"C:\Users\darek\test_output.csv")

# # gdzie zapisać gotowy arkusz
# OUT_XLSX   = SOURCE_CSV.with_suffix(".entities.xlsx")
# # --------------------------------


# def merge_bio(df: pd.DataFrame, tag_col: str, keep_page: bool = True):
#     """
#     Łączy tokeny wg konwencji BIO (B-, I-, L-) dla jednej kolumny tagów.
#     Zwraca listę słowników: {"entity": ..., "type": ..., "page": ...}
#     """
#     merged = []
#     current_tokens, current_type, current_page = [], None, None

#     # iterujemy po wierszach w kolejności występowania (index rośnie)
#     for _, row in df.iterrows():
#         token = str(row["TOKEN"])
#         tag   = str(row[tag_col])
#         page  = row["PAGE"] if keep_page and "PAGE" in df.columns else None

#         # ignoruj puste / brakujące
#         if pd.isna(tag) or tag == "O":
#             if current_tokens:
#                 merged.append(
#                     {"entity": " ".join(current_tokens),
#                      "type": current_type,
#                      "page": current_page})
#                 current_tokens, current_type, current_page = [], None, None
#             continue

#         # rozbij prefiks (B-, I-, L-) od etykiety właściwej
#         if "-" in tag:
#             prefix, etype = tag.split("-", 1)
#         else:           # fallback gdyby etykieta była bez prefiksu
#             prefix, etype = "B", tag

#         # --- logika łączenia ---
#         if prefix == "B":                     # początek nowego bytu
#             # jeśli coś „się ciągnęło” - zamknij poprzedni
#             if current_tokens:
#                 merged.append(
#                     {"entity": " ".join(current_tokens),
#                      "type": current_type,
#                      "page": current_page})
#             current_tokens, current_type, current_page = [token], etype, page

#         elif prefix in ("I", "L") and current_type == etype:
#             current_tokens.append(token)
#             # przy L- zamykamy byt natychmiast
#             if prefix == "L":
#                 merged.append(
#                     {"entity": " ".join(current_tokens),
#                      "type": current_type,
#                      "page": current_page})
#                 current_tokens, current_type, current_page = [], None, None

#         else:  # pojedynczy token z innym typem lub błędnym prefiksem
#             if current_tokens:
#                 merged.append(
#                     {"entity": " ".join(current_tokens),
#                      "type": current_type,
#                      "page": current_page})
#             current_tokens, current_type, current_page = [token], etype, page

#     # dopisujemy ewentualny „wiszący” byt z końca pliku
#     if current_tokens:
#         merged.append(
#             {"entity": " ".join(current_tokens),
#              "type": current_type,
#              "page": current_page})

#     return merged


# def merge_all_tags(df, tag_columns=("PERS", "LOCS")):
#     """
#     Zbiera unikalne byty ze wszystkich wskazanych kolumn tagów
#     i zwraca DataFrame gotowy do eksportu.
#     """
#     all_entities = []
#     for tag_col in tag_columns:
#         if tag_col in df.columns:
#             all_entities.extend(merge_bio(df, tag_col))

#     # usuwamy duplikaty (ten sam string + typ + strona)
#     out = pd.DataFrame(all_entities).drop_duplicates()
#     # sortowanie wg strony i kolejności wystąpienia ułatwia nawigację
#     if "page" in out.columns:
#         out = out.sort_values(["page", "entity"]).reset_index(drop=True)
#     else:
#         out = out.sort_values("entity").reset_index(drop=True)
#     return out


# def main():
#     # 1️⃣ wczytanie
#     df_tokens = pd.read_csv(SOURCE_CSV, sep=",")

#     # 2️⃣ łączenie bytów
#     entities_df = merge_all_tags(df_tokens, tag_columns=("PERS", "LOCS"))

#     # 3️⃣ eksport do Excela
#     with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
#         entities_df.to_excel(writer, index=False, sheet_name="Entities")
#         # bonus: wrzućmy jeszcze oryginalne tokeny do drugiego arkusza
#         df_tokens.to_excel(writer, index=False, sheet_name="Tokens_RAW")

#     print(f"Gotowe! Wynik zapisano do:\n{OUT_XLSX}")


# if __name__ == "__main__":
#     main()

import torch
from transformers import pipeline
import re
import pandas as pd
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ===== 1. Ładowanie tekstu z pliku TXT lub PDF =====
def load_text_from_txt(filepath):
    with open(filepath, encoding='utf8') as f:
        return [f.read()]

# ===== 2. Przetwarzanie NER =====
class TextProcessorFromPages:
    def __init__(self, pages):
        self.pages = pages  # lista stringów, np. ['strona 1...', 'strona 2...']
        self.new_sentences = []
        self.results = []
        self.new_sentences_token_info = []
        self.new_sentences_bio = []
        self.BIO_TAGS = []
        self.stripped_BIO_TAGS = []

    def process_pages(self):
        # Rozbij na "pseudo-zdania", np. fragmenty po 250 tokenów
        for page in self.pages:
            chunks = re.split(r'(?<=[.?!])\s+', page.strip())
            temp = ""
            for chunk in chunks:
                if len(temp.split()) + len(chunk.split()) < 250:
                    temp += " " + chunk
                else:
                    self.new_sentences.append(temp.strip())
                    temp = chunk
            if temp:
                self.new_sentences.append(temp.strip())

    def apply_model(self, pipe):
        # ZBIERAJ również SCORE!
        self.results = []
        for sent in self.new_sentences:
            preds = pipe(sent)
            self.results.append([
                [y["entity"], y["word"], y["start"], y["end"], y.get("score", None)] for y in preds
            ])

    def tokenize_sentences(self):
        for n_s in self.new_sentences:
            tokens = n_s.split()
            token_info = []
            char_index = 0
            for token in tokens:
                start = char_index
                end = char_index + len(token)
                token_info.append((token, start, end))
                char_index += len(token) + 1
            self.new_sentences_token_info.append(token_info)

    def process_results(self):
        for result in self.results:
            merged_bio_result = []
            current_word = ""
            current_label = None
            current_start = None
            current_end = None
            current_scores = []
            for entity, subword, start, end, score in result:
                if subword.startswith("▁"):
                    subword = subword[1:]
                    if current_word:  # jeśli już coś się zebrało, zapisujemy
                        merged_bio_result.append([
                            current_word, current_label, current_start, current_end,
                            np.mean(current_scores) if current_scores else None
                        ])
                    current_word = "" ; current_label = None ; current_start = None ; current_end = None ; current_scores = []
                if current_start is None:
                    current_word = subword
                    current_label = entity
                    current_start = start+1
                    current_end = end
                    current_scores = [score]
                else:
                    current_word += subword
                    current_end = end
                    current_scores.append(score)
            if current_word:
                merged_bio_result.append([
                    current_word, current_label, current_start, current_end,
                    np.mean(current_scores) if current_scores else None
                ])
            self.new_sentences_bio.append(merged_bio_result[1:])

    def match_tokens_with_entities(self):
        for i, ss in enumerate(self.new_sentences_token_info):
            for word in ss:
                for ent in self.new_sentences_bio[i]:
                    if word[1] == ent[2]:
                        if ent[1] == "L-PERS":
                            self.BIO_TAGS.append([word[0], "I-PERS", "B-LOC", ent[4]])
                            break
                        else:
                            if "LOC" in ent[1]:
                                self.BIO_TAGS.append([word[0], "O", ent[1], ent[4]])
                            else:
                                self.BIO_TAGS.append([word[0], ent[1], "O", ent[4]])
                            break
                else:
                    self.BIO_TAGS.append([word[0], "O", "O", None])

    def separate_dots_and_comma(self):
        signs = [",", ";", ":", "."]
        for bio in self.BIO_TAGS:
            if any(bio[0][-1] == sign for sign in signs) and len(bio[0]) > 1:
                self.stripped_BIO_TAGS.append([bio[0][:-1], bio[1], bio[2], bio[3]])
                self.stripped_BIO_TAGS.append([bio[0][-1], "O", "O", None])
            else:
                self.stripped_BIO_TAGS.append(bio)

    def save_BIO_to_csv(self, filename="output.csv"):
        df = pd.DataFrame(self.stripped_BIO_TAGS, columns=["TOKEN", "PERS", "LOCS", "SCORE"])
        df.to_csv(filename, index=False)

# ===== 3. Łączymy byty BIO + agregujemy score =====
def merge_bio(df: pd.DataFrame, tag_col: str, keep_page: bool = False):
    merged = []
    current_tokens, current_type, current_page, current_scores = [], None, None, []
    for _, row in df.iterrows():
        token = str(row["TOKEN"])
        tag   = str(row[tag_col])
        score = float(row.get("SCORE", 0)) if "SCORE" in row and pd.notna(row["SCORE"]) else None
        page  = row["PAGE"] if keep_page and "PAGE" in df.columns else None

        if pd.isna(tag) or tag == "O":
            if current_tokens:
                merged.append(
                    {"entity": " ".join(current_tokens),
                     "type": current_type,
                     "page": current_page,
                     "score": np.mean(current_scores) if current_scores else None})
                current_tokens, current_type, current_page, current_scores = [], None, None, []
            continue

        if "-" in tag:
            prefix, etype = tag.split("-", 1)
        else:
            prefix, etype = "B", tag

        if prefix == "B":
            if current_tokens:
                merged.append(
                    {"entity": " ".join(current_tokens),
                     "type": current_type,
                     "page": current_page,
                     "score": np.mean(current_scores) if current_scores else None})
            current_tokens, current_type, current_page, current_scores = [token], etype, page, [score]
        elif prefix in ("I", "L") and current_type == etype:
            current_tokens.append(token)
            current_scores.append(score)
            if prefix == "L":
                merged.append(
                    {"entity": " ".join(current_tokens),
                     "type": current_type,
                     "page": current_page,
                     "score": np.mean(current_scores) if current_scores else None})
                current_tokens, current_type, current_page, current_scores = [], None, None, []
        else:
            if current_tokens:
                merged.append(
                    {"entity": " ".join(current_tokens),
                     "type": current_type,
                     "page": current_page,
                     "score": np.mean(current_scores) if current_scores else None})
            current_tokens, current_type, current_page, current_scores = [token], etype, page, [score]
    if current_tokens:
        merged.append(
            {"entity": " ".join(current_tokens),
             "type": current_type,
             "page": current_page,
             "score": np.mean(current_scores) if current_scores else None})

    return merged

def merge_all_tags(df, tag_columns=("PERS", "LOCS")):
    all_entities = []
    for tag_col in tag_columns:
        if tag_col in df.columns:
            all_entities.extend(merge_bio(df, tag_col))
    out = pd.DataFrame(all_entities).drop_duplicates()
    if "page" in out.columns:
        out = out.sort_values(["page", "entity"]).reset_index(drop=True)
    else:
        out = out.sort_values("entity").reset_index(drop=True)
    return out

# ===== 4. GŁÓWNA CZĘŚĆ PIPELINE =====


def load_texts_from_folder(folder):
    folder = Path(folder)
    txt_files = sorted(folder.glob("*.txt"))
    return txt_files

# ---- Pipeline główny ----
def main(mode="single"):
    model_name = "magistermilitum/roberta-multilingual-medieval-ner"
    pipe = pipeline("token-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)

    if mode == "single":
        # Pojedynczy plik
        txt_path = r"C:/pdf_llm_do_roboty/single_pages_simple_text-20250804T064354Z-1-001/single_pages_simple_text/MonumentaPeruana_1 - 0776.txt"
        pages = load_text_from_txt(txt_path)
        processor = TextProcessorFromPages(pages)
        processor.process_pages()
        processor.apply_model(pipe)
        processor.tokenize_sentences()
        processor.process_results()
        processor.match_tokens_with_entities()
        processor.separate_dots_and_comma()
        processor.save_BIO_to_csv("output_with_scores.csv")

        df_tokens = pd.read_csv("output_with_scores.csv", sep=",")
        entities_df = merge_all_tags(df_tokens, tag_columns=("PERS", "LOCS"))
        entities_df['page'] = Path(txt_path).name
        entities_df.to_excel("entities_with_scores.xlsx", index=False)
        print("Gotowe! Sprawdź plik: entities_with_scores.xlsx")

    elif mode == "folder":
        # Przetwarzanie wszystkich plików txt w folderze
        folder_path = r"C:/pdf_llm_do_roboty/single_pages_simple_text-20250804T064354Z-1-001/probka/"
        txt_files = load_texts_from_folder(folder_path)
        print(f"Znaleziono {len(txt_files)} plików TXT do przetworzenia.")

        all_entities = []

        for txt_file in tqdm(txt_files, desc="NER na plikach"):
            pages = load_text_from_txt(txt_file)
            processor = TextProcessorFromPages(pages)
            processor.process_pages()
            processor.apply_model(pipe)
            processor.tokenize_sentences()
            processor.process_results()
            processor.match_tokens_with_entities()
            processor.separate_dots_and_comma()
            tmp_csv = "tmp_bio.csv"
            processor.save_BIO_to_csv(tmp_csv)
            df_tokens = pd.read_csv(tmp_csv, sep=",")
            entities_df = merge_all_tags(df_tokens, tag_columns=("PERS", "LOCS"))
            entities_df["page"] = txt_file.name
            all_entities.append(entities_df)

        final_df = pd.concat(all_entities, ignore_index=True)
        final_df = final_df.sort_values(["page", "entity", "type"]).reset_index(drop=True)
        out_xlsx = Path(folder_path) / "entities_all_pages.xlsx"
        final_df.to_excel(out_xlsx, index=False)
        print(f"Wynik zapisany do: {out_xlsx}")

    else:
        print("Nieznany tryb działania: wybierz 'single' albo 'folder'.")

if __name__ == "__main__":
    main(mode="single")   # Jeśli chcesz pojedynczy plik
    main(mode="folder")    # Jeśli chcesz cały katalog
    
    
    
    
from PyPDF2 import PdfReader, PdfWriter

# Ścieżka do oryginału:
input_pdf = "C:/Users/darek/Downloads/Monumenta_Peruana_1.pdf"

# Od której strony zaczynamy nowy plik (np. 776 = 775, bo liczymy od zera)
start_page = 775

# Otwieramy PDF
reader = PdfReader(input_pdf)
writer = PdfWriter()

# Dodajemy kolejne strony od start_page do końca
for page_num in range(start_page, len(reader.pages)):
    writer.add_page(reader.pages[page_num])

# Zapisujemy do nowego pliku
output_pdf = f"Monumenta_Peruana_1_od_{start_page+1}.pdf"
with open(output_pdf, "wb") as f:
    writer.write(f)

print(f"Zapisano: {output_pdf}")




