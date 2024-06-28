# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:55:46 2024

@author: dariu
"""

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import HerbertTokenizerFast

# Wybierz model
model_name = 'allegro/herbert-large-cased'
tokenizer = HerbertTokenizerFast.from_pretrained('allegro/herbert-large-cased')

# Sprawdź maksymalną długość sekwencji tokenów
max_length = tokenizer.model_max_length
print(f'Maksymalna liczba tokenów dla modelu {model_name}: {max_length}')

from transformers import AutoConfig

# Wybierz model
model_name = 'allegro/herbert-large-cased'
config = AutoConfig.from_pretrained(model_name)

# Sprawdź maksymalną długość sekwencji tokenów
max_length = config.max_position_embeddings
print(f'Maksymalna liczba tokenów dla modelu {model_name}: {max_length}')