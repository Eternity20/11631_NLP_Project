#!pip install torch
#!pip install transformers
#!pip install pytorch_lightning==0.7.5
#!pip install sentencepiece

import sentencepiece
from transformers import T5ForConditionalGeneration,T5Tokenizer
import torch
import sys
import warnings
warnings.filterwarnings('ignore')


argument_lst = sys.argv

filename = argument_lst[1]
nquestion = int(argument_lst[2])


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

set_seed(42)



model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)


def greedy_decoding (inp_ids,attn_mask):
    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
    return Question.strip().capitalize()

def beam_search_decoding (inp_ids,attn_mask):
    beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               num_beams=10,
                               num_return_sequences=nquestion,
                               no_repeat_ngram_size=2,
                               early_stopping=True
                               )
    Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               beam_output]
    return [Question.strip().capitalize() for Question in Questions]


def topkp_decoding (inp_ids,attn_mask):
    topkp_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               do_sample=True,
                               top_k=40,
                               top_p=0.80,
                               num_return_sequences=nquestion,
                                no_repeat_ngram_size=2,
                                early_stopping=True
                               )
    Questions = [tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]
    return [Question.strip().capitalize() for Question in Questions]

with open(filename,"r") as f:
    passage = f.read()
truefalse ="yes"

text = "truefalse: %s passage: %s </s>" % (passage, truefalse)


max_len = 256

encoding = tokenizer.encode_plus(text, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)



# print ("Context: ",passage)
#
# output = beam_search_decoding(input_ids,attention_masks)
# print ("\nBeam decoding [Most accurate questions] ::\n")
# for out in output:
#     print(out)


output = topkp_decoding(input_ids,attention_masks)
# print ("\nTopKP decoding [Not very accurate but more variety in questions] ::\n")
for out in output:
    print (out)

print ("\n")







