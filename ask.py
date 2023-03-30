import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader
#logging.basicConfig(level=logging.INFO)
from transformers import (
	T5TokenizerFast,
	T5ForConditionalGeneration, T5Tokenizer
)
from datasets import load_dataset

QA_MODEL = 'deepset/tinyroberta-squad2'
QG_MODEL = 'allenai/t5-small-squad2-question-generation'

def generate_questions(wiki_file_path, nr_questions):
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	#logger.info(f'Using Device: {device}')

	tokenizer = T5Tokenizer.from_pretrained(QG_MODEL)
	model = T5ForConditionalGeneration.from_pretrained(QG_MODEL).to(device)

	qg_dataset = load_dataset('text', data_files={'test': [wiki_file_path]}, sample_by='paragraph')
	# this will load one paragraph at a time
	qg_dataloader = DataLoader(qg_dataset['test'], batch_size=1, num_workers=1)
	questions_generated = []
	for input_string in qg_dataloader:
		if nr_questions < len(questions_generated):
			break
		else:
			if len(input_string['text'][0]) < 20: # minimum context to generate question
				continue
			else:
				input_ids = tokenizer.encode(input_string['text'][0], return_tensors="pt").to(device)
				res = model.generate(input_ids, max_length=40)
				output = tokenizer.batch_decode(res, skip_special_tokens=True)
				questions_generated.extend(output)
	return questions_generated

if __name__ == '__main__':
	logger = logging.getLogger(__name__)
	parser = argparse.ArgumentParser()
	parser.add_argument('--wiki_file_path', type=str, required=True)
	parser.add_argument('--nquestions', type=int, required=True)
	args = parser.parse_args()

	generated_questions = generate_questions(args.wiki_file_path, args.nquestions)
	for question in generated_questions:
		print(f'{question}')
	sys.exit(0)