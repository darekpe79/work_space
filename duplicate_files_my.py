



# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:13:02 2022

@author: dariu
"""
import os
from pathlib import Path
from filecmp import cmp
from tqdm import tqdm

DATA_DIR = Path('C:/Users/dariu/compare')
files = sorted(os.listdir(DATA_DIR))

duplicate={}
for i in range(len(files)):
    print()
    for j in range(i + 1, len(files)):
        print(files[i], files[j])
        
        
        comp = cmp(DATA_DIR/files[i], DATA_DIR/files[j], shallow = False)
        if comp:
            if any([files[i] in x for x in duplicate.values()]):
                continue
            else:
                
            
                
                    if files[i] not in duplicate:
                        
                        duplicate[files[i]]=[files[j]]
                    else:
                        duplicate[files[i]].append(files[j])
            
            
comp = cmp(files[i], files[j], shallow = False)
dictionary={'lala':['komp','lala'], 'kot':['kom','jak']}      
print(any(['komp' in x for x in dictionary.values()])) 
import regex as re
from tkinter import *
from tkinter import ttk
import os
from pathlib import Path
from filecmp import cmp
from tqdm import tqdm
#Create an instance of tkinter frame or window
win= Tk()
#Set the geometry of tkinter frame
win.geometry("750x250")
def get_value():
   e_text=entry.get()
   DATA_DIR = Path(re.escape(e_text))
   files = sorted(os.listdir(DATA_DIR))
   duplicate={}
   for i in range(len(files)):
       print()
       for j in range(i + 1, len(files)):
           print(files[i], files[j])
           
           
           comp = cmp(DATA_DIR/files[i], DATA_DIR/files[j], shallow = False)
           if comp:
               if any([files[i] in x for x in duplicate.values()]):
                   continue
               else:
                   
               
                   
                       if files[i] not in duplicate:
                           
                           duplicate[files[i]]=[files[j]]
                       else:
                           duplicate[files[i]].append(files[j])
    
   listbox = Listbox(root, width=40, height=10, selectmode=MULTIPLE) 
   for i,x in duplicate.items():
       
       Label(myFrame, text = "● "+i +'=  '+' '.join(x) ).pack() #you can use a bullet point emoji.
   counter=0
   for v in x:
       counter+=1
        
       listbox.insert(counter, v)
   return duplicate, listbox 
    
def selected_item():
    for i in listbox.curselection():
        print(listbox.get(i))   
    
   

entry= ttk.Entry(win,font=('Century 12'),width=100)
entry.pack(pady= 30)
#Create a button to display the text of entry widget
button= ttk.Button(win, text="Enter", command= get_value)
btn = ttk.Button(root, text='Print Selected', command=selected_item)
listbox.pack()
button.pack()
win.mainloop()




from tkinter import *
root = Tk()
myFrame = Frame(root).place()
myList = {"a":'c', "b":'d', "c":'f', 'g':'edwerew'}
for i,x in myList.items():
    Label(myFrame, text = "● "+i +'='+x ).pack() #you can use a bullet point emoji.
root.mainloop()
########
from tkinter import *

# Create the root window
root = Tk()
root.geometry('180x200')

# Create a listbox
listbox = Listbox(root, width=40, height=10, selectmode=MULTIPLE)

# Inserting the listbox items
listbox.insert(1, "Data Structure")
listbox.insert(2, "Algorithm")
listbox.insert(3, "Data Science")
listbox.insert(4, "Machine Learning")
listbox.insert(5, "Blockchain")

# Function for printing the
# selected listbox value(s)
def selected_item():#traverse the tuple returned by
# curselection method and print
# corresponding value(s) in the listbox
    for i in listbox.curselection():
        print(listbox.get(i))

# Create a button widget and
# map the command parameter to
# selected_item function
btn = Button(root, text='Print Selected', command=selected_item)

# Placing the button and listbox
btn.pack(side='bottom')
listbox.pack()

root.mainloop()



################
from tkinter import *
from tkinter import ttk
#Create an instance of tkinter frame or window
win= Tk()
#Set the geometry of tkinter frame
win.geometry("750x250")
def get_value():
   e_text=entry.get()
   Label(win, text=e_text, font= ('Century 15 bold')).pack(pady=20)
#Create an Entry Widget
entry= ttk.Entry(win,font=('Century 12'),width=40)
entry.pack(pady= 30)
#Create a button to display the text of entry widget
button= ttk.Button(win, text="Enter", command= get_value)
button.pack()
win.mainloop()


#######################################




import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

# create the root window
root = tk.Tk()
root.title('Listbox')

entry= ttk.Entry(root,font=('Century 12'),width=100)
entry.pack(pady= 30)
# create a list box
langs = ['Java', 'C#', 'C', 'C++', 'Python',
         'Go', 'JavaScript', 'PHP', 'Swift']

var = tk.Variable(value=langs)

listbox = tk.Listbox(
    root,
    listvariable=var,
    height=6,
    selectmode=tk.EXTENDED
)

listbox.pack(expand=True, fill=tk.BOTH)


def items_selected(event):
    # get all selected indices
    selected_indices = listbox.curselection()
    # get selected items
    selected_langs = ",".join([listbox.get(i) for i in selected_indices])
    msg = f'You selected: {selected_langs}'
    showinfo(title='Information', message=msg)


listbox.bind('<<ListboxSelect>>', items_selected)

root.mainloop()
####


#####
root.mainloop()

###Działające
import regex as re
from tkinter import *
from tkinter import ttk
import os
from pathlib import Path
from filecmp import cmp
from tqdm import tqdm
#Create an instance of tkinter frame or window
win= Tk()
#myFrame = Frame(win).place()
#Set the geometry of tkinter frame
win.geometry("750x250")

def get_value():
   e_text=entry.get()
   DATA_DIR = Path(re.escape(e_text))
   files = sorted(os.listdir(DATA_DIR))
   duplicate={}
   list_values=[]
   for i in range(len(files)):
       
       for j in range(i + 1, len(files)):
           print(files[i], files[j])
           
           
           comp = cmp(DATA_DIR/files[i], DATA_DIR/files[j], shallow = False)
           if comp:
               if any([files[i] in x for x in duplicate.values()]):
                   continue
               else:
                   
               
                   
                       if files[i] not in duplicate:
                           
                           duplicate[files[i]]=[files[j]]
                       else:
                           duplicate[files[i]].append(files[j])
   for x in duplicate.values():
           list_values.append(x)
    
    
    
   for i,x in duplicate.items():
       Label(win, text = "● "+i +'=  '+', '.join(x) ).pack() #you can use a bullet point emoji.
########


entry= ttk.Entry(win,font=('Century 12'),width=100)
entry.pack(pady= 30)
#Create a button to display the text of entry widget
button= ttk.Button(win, text="Enter", command= get_value)
button.pack()
#get_value()
win.mainloop()




#####PROBA
import regex as re
from tkinter import *
from tkinter import ttk
import os
from pathlib import Path
from filecmp import cmp
from tqdm import tqdm
#Create an instance of tkinter frame or window
win= Tk()
#myFrame = Frame(win).place()
#Set the geometry of tkinter frame
win.geometry("750x250")

def get_value():
   global list_values
   e_text=entry.get()
   DATA_DIR = Path(re.escape(e_text))
   files = sorted(os.listdir(DATA_DIR))
   duplicate={}
   list_values=[]
   for i in range(len(files)):
       
       for j in range(i + 1, len(files)):
           print(files[i], files[j])
           
           
           comp = cmp(DATA_DIR/files[i], DATA_DIR/files[j], shallow = False)
           if comp:
               if any([files[i] in x for x in duplicate.values()]):
                   continue
               else:
                   
               
                   
                       if files[i] not in duplicate:
                           
                           duplicate[files[i]]=[files[j]]
                       else:
                           duplicate[files[i]].append(files[j])
   for x in duplicate.values():
       for y in x:
           list_values.append(y)
    
    
    
   for i,x in duplicate.items():
       Label(win, text = "● "+i +'=  '+', '.join(x) ).pack() #you can use a bullet point emoji.
########


entry= ttk.Entry(win,font=('Century 12'),width=100)
entry.pack(pady= 30)
#Create a button to display the text of entry widget
button= ttk.Button(win, text="Enter", command= get_value)
button.pack()

#get_value()
win.mainloop()





