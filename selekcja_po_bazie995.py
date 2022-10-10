from definicje import*
"""
Created on Wed Jun  8 15:46:48 2022

@author: darek
"""
journal=[]
for plik in [r"F:\Nowa_praca\24.05.2022Marki\arto.mrk"]:
    lista=mark_to_list(plik)
    dictrec=list_of_dict_from_list_of_lists(lista)
    for rekord in tqdm(dictrec):
        for key, val in rekord.items():
            
            
            if key=='995':
                
                
                #print(val)
                v=val.split('❦')
                for value in v:
                    if 'Journal.fi' in value:
                        journal.append(rekord)
                        