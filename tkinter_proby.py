# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:27:52 2022

@author: dariu
"""
import tkinter as tk
window= tk.Tk()
window.title('my app')

label=tk.Label(text="insert ")



window.mainloop()







import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo




# create the root window
root = tk.Tk()
root.title('Listbox')


entry= ttk.Entry(root,font=('Century 12'),width=100)
entry.pack(pady= 30)
# create a list box
langs = [entry.get()]

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







root = tk.Tk()
root.title("List App")
root.geometry("400x400")
 
def retrievedata():
    ''' get data stored '''
    global list_data
    list_data = []
    try:
      with open("save.txt", "r", encoding="utf-8") as file:
       for f in file:
        listbox.insert(tk.END, f.strip())
        list_data.append(f.strip())
        print(list_data)
    except:
        pass
 
def reload_data():
    listbox.delete(0, tk.END)
    for d in list_data:
        listbox.insert(0, d)
 
 
def add_item(event=1):
    global list_data
    if content.get() != "":
        listbox.insert(tk.END, content.get())
        list_data.append(content.get())
        content.set("")
 
 
def delete():
    global list_data
    listbox.delete(0, tk.END)
    list_data = []
 
 
def delete_selected():
 
    try:
        selected = listbox.get(listbox.curselection())
        listbox.delete(listbox.curselection())
        list_data.pop(list_data.index(selected))
        # reload_data()
        # # listbox.selection_clear(0, END)
        listbox.selection_set(0)
        listbox.activate(0)
        listbox.event_generate("&lt;&lt;ListboxSelect>>")
        print(listbox.curselection())
    except:
        pass
 
 
 
def quit():
 global root
 with open("save.txt", "w", encoding="utf-8") as file:
  for d in list_data:
   file.write(d + "\n")
 root.destroy()
 
# LISTBOX
 
content = tk.StringVar()
entry = tk.Entry(root, textvariable=content)
entry.pack()
 
button = tk.Button(root, text="Add Item", command=add_item)
button.pack()
 
button_delete = tk.Button(text="Delete", command=delete)
button_delete.pack()
 
button_delete_selected = tk.Button(text="Delete Selected", command=delete_selected)
button_delete_selected.pack()
 
listbox = tk.Listbox(root)
listbox.pack()
entry.bind("&lt;Return>", add_item)
 
bquit = tk.Button(root, text="Quit and save", command=quit)
bquit.pack()
 
retrievedata()
root.mainloop()