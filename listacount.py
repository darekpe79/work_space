list=['lala','lala','reks']
count={}
for x in list:
    if x in count.keys():
        count[x]+=1
        
    else:
        count[x]=1