# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:58:12 2022

@author: dariu
"""
import json
from tqdm import tqdm
import requests
from definicje import *
import pandas as pd
import os
from tqdm import tqdm
#from googletrans import Translator
from itertools import zip_longest
import regex as re
from pprint import pprint
import pprint
from time import time



    
def centuryFromYear(year):
    return (year + 99) // 100
centuryFromYear(1001)