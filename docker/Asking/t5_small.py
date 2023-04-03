#!/usr/bin/python3 -W ignore::DeprecationWarning
# -*- coding:utf8 -*-

import sys
import logging
import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer
)
from datasets import load_dataset,utils
logging.basicConfig(level=logging.CRITICAL)


class T5SmallQuestionGenerator:

    def __init__(self,wiki_file_path,nquestions):
        self.QA_MODEL = 'deepset/tinyroberta-squad2'
        self.QG_MODEL = 'allenai/t5-small-squad2-question-generation'
        self.wiki_file_path = wiki_file_path
        self.nquestions = nquestions

    def generate_questions(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        logging.getLogger('datasets').setLevel(logging.CRITICAL)
        utils.disable_progress_bar()
        tokenizer = T5Tokenizer.from_pretrained(self.QG_MODEL)
        model = T5ForConditionalGeneration.from_pretrained(self.QG_MODEL).to(device)
        qg_dataset = load_dataset('text', data_files={'test': [self.wiki_file_path]}, sample_by='paragraph')
            # this will load one paragraph at a time
        qg_dataloader = DataLoader(qg_dataset['test'], batch_size=1, num_workers=1)
        questions_generated = []
        for input_string in qg_dataloader:
            if self.nquestions <= len(questions_generated):
                break
            else:
                if len(input_string['text'][0]) < 20:  # minimum context to generate question
                    continue
                else:
                    input_ids = tokenizer.encode(input_string['text'][0], return_tensors="pt").to(device)
                    res = model.generate(input_ids, max_length=40)
                    output = tokenizer.batch_decode(res, skip_special_tokens=True)
                    questions_generated.extend(output)
        return questions_generated


if __name__ == '__main__':
    input_file = sys.argv[1]
    N = int(sys.argv[2])
    qg = T5SmallQuestionGenerator(input_file,N)
    generated_questions = qg.generate_questions()
    for question in generated_questions:
        print(f'{question}')