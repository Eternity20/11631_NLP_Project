import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random
import collections
from tqdm import tqdm
import os
import json
import torch
from torch.utils.data import (
	DataLoader,
	Dataset
)
from datasets import load_dataset
from evaluate import load
from transformers import (
	AutoTokenizer,
	AutoModelForQuestionAnswering
)
from .datasets.QADataset import QADataset

def set_random_seed(seed: int = 0):
	"""
	Helper function to seed experiment for reproducibility.
	If -1 is provided as seed, experiment uses random seed from 0~9999
	Args:
		seed (int): integer to be used as seed, use -1 to randomly seed experiment
	"""
	print("Seed: {}".format(seed))

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.deterministic = True

	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)



class QAModel:
	def __init__(self, model, device, max_length=386, truncation='only_second', padding='max_length',
	             return_overflowing_tokens=True, return_offset_mappings=True, stride=128,
	             n_best_size=20, max_answer_length=30, batch_size=16):
		self.model_name = model
		self.device = device
		self.max_length = max_length
		self.truncation = truncation
		self.padding = padding
		self.return_overflowing_tokens = return_overflowing_tokens
		self.return_offset_mappings = return_offset_mappings
		self.stride = stride
		self.n_best_size = n_best_size
		self.max_answer_length = max_answer_length
		self.batch_size = batch_size
		set_random_seed()
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device)

	@classmethod
	def from_config_dict(cls, model, device, config):
		return cls(model, device, **config)

	def load_dataset_and_dataloaders(self, wiki_file_path=None, wiki_file_dir=None, paragraph=True, config=None):
		if wiki_file_path:
			# It should be test below but i didnt find how to change it for both cases
			_dataset = load_dataset('text', data_files={'train': [wiki_file_path]},
			                        sample_by='paragraph' if paragraph else 'document')
		if wiki_file_dir:
			_dataset = load_dataset('text', data_dir=wiki_file_dir,
			                        sample_by='paragraph' if paragraph else 'document')

		if config:
			eval_dataset = QADataset.from_config_dict(
				data=datasets['train'],
				tokenizer=self.tokenizer,
				config=config
			)
		else:
			eval_dataset = QADataset(
				data=datasets['train'],
				tokenizer=self.tokenizer,
				max_length=self.max_answer_length,
				stride=self.stride,
				truncation=self.truncation,
				padding=self.padding,
				return_overflowing_tokens=self.return_overflowing_tokens,
				return_offsets_mapping=self.return_offset_mappings,
			)
		eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size)

#		eval_data = eval_dataset.data
#		eval_features = eval_dataset.tokenized_data
		return eval_dataset, eval_dataloader

	def qa_inference(self, data_loader):
		self.qa_model.eval()
		start_logits = []
		end_logits = []
		for step, batch in enumerate(tqdm(data_loader, desc="Inference Iteration")):
			with torch.no_grad():
				model_kwargs = {
					'input_ids': batch['input_ids'].to(self.device, dtype=torch.long),
					'attention_mask': batch['attention_mask'].to(self.device, dtype=torch.long)
				}
				# TODO 6: pass the model arguments to the model and store the output
				outputs = self.qa_model(**model_kwargs)
				# TODO 7: Extract the start and end logits by extending `start_logits` and `end_logits`
				start_logits_cpu = outputs['start_logits'].cpu().detach().numpy()
				start_logits.extend(start_logits_cpu)
				end_logits_cpu = outputs['end_logits'].cpu().detach().numpy()
				end_logits.extend(end_logits_cpu)
		# TODO 8: Convert the start and end logits to a numpy array (by passing them to `np.array`)
		start_logits, end_logits = np.array(start_logits), np.array(end_logits)
		# TODO 9: return start and end logits
		return start_logits, end_logits

	def post_processing(self, raw_dataset, tokenized_dataset, start_logits, end_logits):
		# Map each example to its features. This is done because an example can have multiple features
		# as we split the context into chunks if it exceeded the max length
		data2features = collections.defaultdict(list)
		for idx, feature_id in enumerate(tokenized_dataset['ID']):
			data2features[feature_id].append(idx)

		# Decode the answers for each datapoint
		predictions = []
		for k, data in enumerate(tqdm(raw_dataset)):
			answers = []
			data_id = data["id"]
			context = data["context"]

			for feature_index in data2features[data_id]:
				# TODO 10: Get the start logit, end logit, and offset mapping for each index.
				start_logits_for_feature_idx, end_logits_for_feature_idx, offset_mapping_for_feature_idx = \
					start_logits[feature_index], end_logits[feature_index], tokenized_dataset['offset_mapping'][
						feature_index]

				# TODO 10.1:
				# min_logit_score = start_logits_for_feature_idx[0] + end_logits_for_feature_idx[0]
				# add CLS - no answer
				# answers.append({"text": "", "logit_score": min_logit_score})

				# TODO 11: Sort the start and end logits and get the top n_best_size logits.
				# Hint: look at other QA pipelines/tutorials.
				start_indexes = np.argsort(start_logits_for_feature_idx).tolist()[::-1][:config['n_best_size']]
				end_indexes = np.argsort(end_logits_for_feature_idx).tolist()[::-1][:config['n_best_size']]

				for start_index in start_indexes:
					for end_index in end_indexes:
						# TODO 12: Exclde answers that are not in the context
						_offset_mapping_start_idx, _offset_mapping_end_idx = \
							offset_mapping_for_feature_idx[start_index][0], offset_mapping_for_feature_idx[end_index][1]
						if _offset_mapping_start_idx == -1 or _offset_mapping_end_idx == -1:
							continue
						# TODO 13: Exclude answers if (answer length < 0) or (answer length > max_answer_length)
						if end_index - start_index < 0 or \
								end_index - start_index + 1 > config['max_answer_length']:
							continue
						# TODO 14: collect answers in a list.
						answers.append(
							{
								"text": context[_offset_mapping_start_idx:_offset_mapping_end_idx],
								"logit_score": end_logits_for_feature_idx[end_index] + \
								               start_logits_for_feature_idx[start_index],
							}
						)
			best_answer = max(answers, key=lambda x: x["logit_score"])
			predictions.append(
				{
					"id": data_id,
					"prediction_text": best_answer["text"],
					"no_answer_probability": 0.0 if len(best_answer["text"]) > 0 else 1.0
				}
			)
		return predictions


if __name__ == '__main__':
	if torch.cuda.is_available():
		DEVICE = torch.device("cuda")
		print("Using GPU: ", DEVICE)
	else:
		DEVICE = torch.device("cpu")
		print("Using CPU: ", DEVICE)

	set_random_seed(SEED)

	# TODO 1: fill in the values for all the hyper-paramters mentioned in the config dictionary.
	config = {
		'model_checkpoint': 'deepset/roberta-base-squad2',
		"max_length": 386,
		"truncation": 'only_second',
		"padding": 'max_length',
		"return_overflowing_tokens": True,
		"return_offsets_mapping": True,
		"stride": 128,
		"n_best_size": 20,
		"max_answer_length": 30,
		"batch_size": 16
	}

	datasets = load_dataset("squad_v2")
	# TODO 2: Define the tokenizer and QA model. Transfer the QA model to GPU.
	tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])
	qa_model = AutoModelForQuestionAnswering.from_pretrained(config['model_checkpoint']).to(DEVICE)

	eval_dataset = QADataset(
		data=datasets['validation'],
		tokenizer=tokenizer,
		config=config
	)
	eval_dataloader = DataLoader(
		eval_dataset,
		batch_size=config["batch_size"]
	)

	eval_data = eval_dataset.data
	eval_features = eval_dataset.tokenized_data

	start_logits, end_logits = qa_inference(qa_model, eval_dataloader)
	predicted_answers = post_processing(
		raw_dataset=eval_data,
		tokenized_dataset=eval_features,
		start_logits=start_logits,
		end_logits=end_logits
	)
	gold_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_data][:len(eval_data)]
	assert len(predicted_answers) == len(gold_answers)

	eval_metric = load("squad_v2")
	eval_results = eval_metric.compute(predictions=predicted_answers, references=gold_answers)
	import json

	with open('./results/squad_results.json', 'w') as f:
		json.dump(eval_results, f)

	test_dataset = load_dataset("csv", data_files="blind_test_set.csv", split='train')
	test_qa_dataset = QADataset(
		data=test_dataset,
		tokenizer=tokenizer,
		config=config
	)
	test_dataloader = DataLoader(
		test_qa_dataset,
		batch_size=config["batch_size"]
	)
	raw_test, tok_test = test_qa_dataset.data, test_qa_dataset.tokenized_data
	start_logits_test, end_logits_test = qa_inference(qa_model, test_dataloader)
	predicted_answers = post_processing(
		raw_dataset=raw_test,
		tokenized_dataset=tok_test,
		start_logits=start_logits_test,
		end_logits=end_logits_test
	)

	with open('blind_test_predictions.json', 'w') as fp:
		json.dump(predicted_answers, fp)

	sys.exit(0)
