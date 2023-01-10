# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:15:19 2023

@author: darek
"""

class  Employee:
    def __init__(self, first, last, pay):
        self.first=first    #atrributes of class
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@company.com'     
        
    def fullname(self):  #method
        return '{} {}'.format(self.first,self.last)

emp_1=(Employee('Darek', "Perla", 10000))
emp_2=(Employee('User', "Test", 20000))
print(emp_1.email)
emp_1.fullname() #method with ()
