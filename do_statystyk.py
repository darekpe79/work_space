from tqdm import tqdm
import os

folderpath = r"/home/darek/Nowa_praca /fennica/fennica_odsiane_24082021" 
os.listdir(folderpath)
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
all_files =[]
all_files1=[]
for path in filepaths:
    with open(path, 'r', encoding='utf-8') as f:
        file = f.readlines()
        all_files.append(file)
        all_files1.extend(file)
records=[]
record=[]        
for line in all_files1:
    
    if line != '\n':
        record.append(line)
    else:
        records.append(record)
        record = []
print(records[3])
