import json
import sys
import argparse
import logging
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
	RobertaTokenizerFast,
	RobertaForQuestionAnswering
)
from mydatasets.QAProjectDataset import QAProjectDataset
from mymodels.QAProjectModel import QAProjectModel
from utils import set_device

#QA_MODEL = 'deepset/tinyroberta-squad2'
QA_MODEL = 'deepset/roberta-base-squad2-distilled'

def answer_questions(wiki_doc, questions, loaded_conf_dict):
	device = torch.device('cpu')#set_device()
	model = RobertaForQuestionAnswering.from_pretrained(QA_MODEL).to(device)
	qa_model = QAProjectModel.from_config_dict(model, device, loaded_conf_dict)
	tokenizer = RobertaTokenizerFast.from_pretrained(QA_MODEL)
	qa_dataset = QAProjectDataset.from_config_dict(wiki_doc,questions, tokenizer, loaded_conf_dict)
	# qa_dataset = load_dataset('text', data_files={'test': [wiki_file_path]}, sample_by='document')
	# this will load one paragraph at a time
	#qa_dataloader = DataLoader(qa_dataset['test'], batch_size=1, num_workers=1)
	qa_dataloader = DataLoader(qa_dataset, batch_size=qa_model.batch_size, num_workers=2, pin_memory=True)

	start_logits, end_logits = qa_model.qa_inference(qa_dataloader)
	pred_answers = qa_model.post_processing(questions, qa_dataset.wiki_doc_str, qa_dataset.tokenized_data, start_logits, end_logits)
	return pred_answers


def read_files(wiki_file_path:str, questions_file_path:str, config_file_path:str) -> Union[List[str],List[str],Dict]:
	config_dict = None
	with open(config_file_path, 'r') as f:
		config_dict = json.load(f)

	wiki_doc = []
	with open(wiki_file_path, 'r') as f:
		for line in f:
			wiki_doc.append(line.strip())

	questions = []
	with open(questions_file_path, 'r') as f:
		for line in f:
			questions.append(line.strip())
	return wiki_doc, questions, config_dict


if __name__ == '__main__':
	logger = logging.getLogger(__name__)
	parser = argparse.ArgumentParser()
	parser.add_argument('--wiki_file_path', type=str, required=True)
	parser.add_argument('--questions_file_path', type=str, required=True)
	parser.add_argument('--config_file_path', type=str, default='config_qa.json', required=False)
	args = parser.parse_args()

	wiki_doc, questions, config_dict = \
		read_files(args.wiki_file_path, args.questions_file_path, args.config_file_path)

	predicted_answers = answer_questions(wiki_doc, questions, config_dict)
	for answer in predicted_answers:
		print(f'{answer}')
	sys.exit(0)
