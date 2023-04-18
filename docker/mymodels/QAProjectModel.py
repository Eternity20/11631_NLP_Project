import os
import random
import torch
import collections
import numpy as np
#from utils import set_random_seed
import transformers
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
from transformers import (
	RobertaTokenizerFast,
	RobertaTokenizer,
	RobertaForQuestionAnswering
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed: int = 0):
	"""
	Helper function to seed experiment for reproducibility.
	If -1 is provided as seed, experiment uses random seed from 0~9999
	Args:
		seed (int): integer to be used as seed, use -1 to randomly seed experiment
	"""
	#print("Seed: {}".format(seed))

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.deterministic = True

	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

class QAProjectModel:
	def __init__(self, model, device, max_length=386, truncation='only_second', padding='max_length',
	             return_overflowing_tokens=True, return_offsets_mapping=True, stride=128,
	             n_best_size=20, max_answer_length=30, batch_size=16):
		self.model_name = model
		self.device = device
		self.max_length = max_length
		self.truncation = truncation
		self.padding = padding
		self.return_overflowing_tokens = return_overflowing_tokens
		self.return_offsets_mapping = return_offsets_mapping
		self.stride = stridegit
		self.n_best_size = n_best_size
		self.max_answer_length = max_answer_length
		self.batch_size = batch_size
		set_random_seed()
		self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name)
		self.model = RobertaForQuestionAnswering.from_pretrained(self.model_name).to(self.device)

	@classmethod
	def from_config_dict(cls, model, device, config):
		return cls(model, device, **config)

	def qa_inference(self, data_loader):
		self.model.eval()
		start_logits = []
		end_logits = []
		for step, batch in enumerate(data_loader):
			with torch.no_grad():
				model_kwargs = {
					'input_ids': batch['input_ids'].to(self.device, dtype=torch.long),
					'attention_mask': batch['attention_mask'].to(self.device, dtype=torch.long)
				}
				# TODO 6: pass the model arguments to the model and store the output
				outputs = self.model(**model_kwargs)
				# TODO 7: Extract the start and end logits by extending `start_logits` and `end_logits`
				start_logits_cpu = outputs['start_logits'].cpu().detach().numpy()
				start_logits.extend(start_logits_cpu)
				end_logits_cpu = outputs['end_logits'].cpu().detach().numpy()
				end_logits.extend(end_logits_cpu)
		# TODO 8: Convert the start and end logits to a numpy array (by passing them to `np.array`)
		start_logits, end_logits = np.array(start_logits), np.array(end_logits)
		# TODO 9: return start and end logits
		return start_logits, end_logits

	def post_processing(self, questions, wiki_doc_str, tokenized_dataset, start_logits, end_logits):
		# Map each example to its features. This is done because an example can have multiple features
		# as we split the context into chunks if it exceeded the max length
		data2features = collections.defaultdict(list)
		for idx, feature_id in enumerate(tokenized_dataset['ID']):
			data2features[feature_id].append(idx)

		# Decode the answers for each datapoint
		predictions = []
		for k, question in enumerate(questions):
			answers = []
			data_id = k
			context = wiki_doc_str

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
				start_indexes = np.argsort(start_logits_for_feature_idx).tolist()[::-1][:self.n_best_size]
				end_indexes = np.argsort(end_logits_for_feature_idx).tolist()[::-1][:self.n_best_size]

				for start_index in start_indexes:
					for end_index in end_indexes:
						# TODO 12: Exclde answers that are not in the context
						_offset_mapping_start_idx, _offset_mapping_end_idx = \
							offset_mapping_for_feature_idx[start_index][0], offset_mapping_for_feature_idx[end_index][1]
						if _offset_mapping_start_idx == -1 or _offset_mapping_end_idx == -1:
							continue
						# TODO 13: Exclude answers if (answer length < 0) or (answer length > max_answer_length)
						if end_index - start_index < 0 or \
								end_index - start_index + 1 > self.max_answer_length:
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
