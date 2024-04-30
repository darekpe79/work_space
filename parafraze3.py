# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:10:09 2024

@author: dariu
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("shrishail/t5_paraphrase_msrp_paws")
model = AutoModelForSeq2SeqLM.from_pretrained("shrishail/t5_paraphrase_msrp_paws")
sentence = "This is something which i cannot understand at all"
text =  "paraphrase: " + sentence + " </s>"
encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    do_sample=True,
    top_k=120,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=5
)
for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(line)