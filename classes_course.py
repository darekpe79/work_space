# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:15:19 2023

@author: darek
"""
import datetime
class  Employee:
    num_of_emps = 0
    raise_amount=1.04 #class variable (Atributte?)
    def __init__(self, first, last, pay):
        self.first=first    #atrributes of class
        self.last=last
        self.pay=pay
        self.email=first+'.'+last+'@company.com'  
        Employee.num_of_emps+=1 #class value
        
    def fullname(self):  #method (regular method with instance as variable)
        return '{} {}'.format(self.first,self.last)
    def apply_raise(self):
        self.pay=int(self.pay*self.raise_amount)
    #Classmethods
    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amount=amount
    @classmethod
    #alternative constructor
    def from_string(cls,emp_str):
        first,last,pay=emp_str.split('-')
        return cls(first,last,pay) #to samo co Employee(first,last,pay)
    
    #staticmethods
    @staticmethod
    def is_workday(day): 
        #no self (instance) no cls (class)as variable 
        # weekday method in python monday==0, sunday==6   
        if day.weekday()==5 or day.weekday() == 6:
            return False
        return True
class Developer (Employee):
    raise_amount=1.10
    def __init__(self, first, last, pay,prog_lang):
        super().__init__(first, last, pay) # same as Employee.__init__(self,first,last,pay)- but super better
        self.prog_lang=prog_lang

class Manager(Employee):
    def __init__(self, first, last, pay,employees=None):
        super().__init__(first, last, pay) # same as Employee.__init__(self,first,last,pay)- but super better
        if employees is None:
            self.employees=[]
        else:
            self.employees=employees
    def add_emp(self,emp):
        if emp not in self.employees:
            self.employees.append(emp)
    def remove_emp (self, emp):
        if emp in self.employees:
            self.employees.remove(emp)
        
    def print_emps(self):
        for emp in self.employees:
            print(emp.fullname())

my_date=datetime.date(2016, 7, 11)
Employee.is_workday(my_date)
my_date.weekday()
        
        
        
emp_1=(Employee('Darek', "Perla", 10000)) # instance
emp_2=(Employee('User', "Test", 20000))
print(emp_1.email)
emp_1.fullname() #method == Employee.fullname(emp_1)
print(emp_1.raise_amount)
Employee.set_raise_amt(1.05)
print(emp_1.raise_amount)
print(emp_2.raise_amount)
emp_1.raise_amount=1.07
print(emp_1.raise_amount)
print(emp_2.raise_amount)
Employee.raise_amount=1.07 # - to to samo co- Employee.set_raise_amt(1.05)
print(emp_1.raise_amount)
print(emp_2.raise_amount)
emp_2.apply_raise()
print(emp_2.pay)


print(emp_1.__dict__)
print(Employee.__dict__)

print(emp_1.raise_amount) #w metodzie mam self (nie Employee) - dlatego mogę zmienić dla pojedyńczego instance sklaę podwyżki i użyć jako metody (użwyam wtedy zmiennej nie dla klasy,ale dla instancji)
print(emp_1.__dict__)
print(emp_2.__dict__)
emp_1.apply_raise()
print(emp_1.pay)
print(emp_2.raise_amount) 
print(Employee.__dict__)
emp_str='Jhon-Doe-10000'
emp_3=Employee.from_string(emp_str)
print(Employee.num_of_emps)
emp_3.__dict__

dev_1=(Developer('User', "Test", 20000, 'python'))
print(dev_1.email)
print(dev_1.prog_lang)
#print(help(Developer))
dev_1.apply_raise()
print(dev_1.pay)





