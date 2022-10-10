import regex as re
from tqdm import tqdm

#%%
paths = [
    r"F:\Nowa_praca\libri\Iteracja 2021-07\20022Owiafowane_do_wysłania\pbl_marc_books2021-8-4_good995_ALL_good_VIAF_700.mrk",
    r"F:\Nowa_praca\libri\Iteracja 2021-07\20022Owiafowane_do_wysłania\pbl_marc_articles2021-8-4_ISSN_good995_ALL_good_VIAF_700.mrk",
    r"F:\Nowa_praca\libri\Iteracja 2021-07\20022Owiafowane_do_wysłania\libri_marc_bn_chapters2_2021-08-05!100_700_600z_VIAF_i_bez_viaf.mrk",
    r"F:\Nowa_praca\libri\Iteracja 2021-07\20022Owiafowane_do_wysłania\libri_marc_bn_chapters_2021-08-05!100_600_700z_VIAF_i_bez_viaf_good995.mrk",
    r"F:\Nowa_praca\libri\Iteracja 2021-07\20022Owiafowane_do_wysłania\libri_marc_bn_books_2021-08-05!100_700_600z_VIAF_i_bez_viaf_good995.mrk",
    r"F:\Nowa_praca\libri\Iteracja 2021-07\20022Owiafowane_do_wysłania\libri_marc_bn_articles_2021-08-05!100_700_600z_VIAF_i_bez_viaf_good995_issns.mrk"
         ]

counter=1
for path in tqdm(paths):
    
    output_lines = []
    path2=path.split('\\') 
    print(path2)
    with open(path, 'r', encoding='utf-8') as mrk:
        for line in mrk.readlines():
            if line[1:4] in ['600', '610', '611', '630', '648', '650', '651', '655'] and 'Miejscowości w Polsce, ' in line:
                #print('-'*20, '\n', line)
                line = re.sub('Miejscowości w Polsce, ', '', line)
                #print(line, '\n','-'*20)
            output_lines.append(line)
    counter+=1        
    out_path = r"F:\Nowa_praca\libri\Iteracja 2021-07\zmienione_miejscowości\\" +path2[-1] 
    with open(out_path, 'w', encoding='utf-8') as mrk:
        mrk.writelines(output_lines)
    
#['600', '610', '611', '630', '648', '650', '651', '655']
