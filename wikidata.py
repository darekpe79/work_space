# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:22:27 2023

@author: dariu
"""

from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api
# create an item representing "Douglas Adams"
Q_DOUGLAS_ADAMS = "Q79822"
q42_dict = get_entity_dict_from_api(Q_DOUGLAS_ADAMS)
q42 = WikidataItem(q42_dict)
q42.get_description()
q42.json_dump()
claim_groups = q42.get_truthy_claim_groups()
p31_claim_group = claim_groups["P31"]

for x in p31_claim_group:
    print(x)
