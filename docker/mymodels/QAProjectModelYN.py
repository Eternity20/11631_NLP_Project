import warnings
warnings.filterwarnings('ignore')

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
	RobertaForSequenceClassification
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

class QAProjectModelYN:
	def __init__(self, model, device, max_length=512, truncation='only_second', padding='max_length', batch_size=16):
		self.model_name = model
		self.device = device
		self.max_length = max_length
		self.truncation = truncation
		self.padding = padding
		self.batch_size = batch_size
		set_random_seed()
		self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
		self.model = RobertaForSequenceClassification.from_pretrained(self.model_name).to(self.device)
		self.model.eval()

	@classmethod
	def from_config_dict(cls, model, device, config):
		return cls(model, device, **config)

	def predict(self, question, passage):
		sequence = self.tokenizer.encode_plus(question, passage, truncation=True, max_length=512, return_tensors="pt")['input_ids'].to(self.device)
		logits = self.model(sequence)[0]
		probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
		_argmax = np.argmax(probabilities)
		result = {'prediction_text': 'Yes' if _argmax == 1 else 'No', 'score': probabilities[_argmax]}
		return result
		#proba_yes = round(probabilities[1], 2)
		#proba_no = round(probabilities[0], 2)

	def qa_inference(self, dataloader):
		predictions = []
		for step, batch in enumerate(dataloader):
			with torch.no_grad():
				model_kwargs = {
					'input_ids': batch['input_ids'].to(self.device, dtype=torch.long),
					'attention_mask': batch['attention_mask'].to(self.device, dtype=torch.long)
				}
				outputs = self.model(**model_kwargs)
				logits_cpu = outputs[0].cpu().detach().numpy()
				_argmax = np.argmax(logits_cpu, axis=1)
				predictions.extend([{'prediction_text': 'Yes' if _argm == 1 else 'No', 'score': logits_cpu[i][_argm]} for i, _argm in enumerate(_argmax)])
		return predictions
