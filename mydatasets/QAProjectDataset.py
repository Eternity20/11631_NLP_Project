import torch
from typing import List
from tqdm import tqdm
from torch.utils.data import (
	Dataset
)

class QAProjectDataset(Dataset):
	def preprocess_doc(self, wiki_doc:List[str]) -> str:
		# strip wiki doc of new lines
		return ' '.join([line.strip() for line in wiki_doc if line.strip()])

	def __init__(self, wiki_doc, questions, tokenizer, max_length, stride, truncation, padding,
	             return_overflowing_tokens, return_offsets_mapping):
		self.wiki_doc = wiki_doc
		self.wiki_doc_str = self.preprocess_doc(wiki_doc)
		self.questions = questions
		self.tokenizer = tokenizer
		#self.wiki_doc_tok = self.tokenizer(wiki_doc, max_length=max_length, stride=stride, truncation=truncation, padding=padding)
		self.ids = [x for x in range(len(questions))]
		self.tokenized_data = self.tokenizer(
			questions,
			[self.wiki_doc_str] * len(questions),
			max_length=max_length,
			stride=stride,
			truncation=truncation,
			padding=padding,
			return_overflowing_tokens=return_overflowing_tokens,
			return_offsets_mapping=return_offsets_mapping,
			return_attention_mask=True,
			add_special_tokens=True
		)

		example_ids = []
		for i, sample_mapping in enumerate(tqdm(self.tokenized_data["overflow_to_sample_mapping"])):
			example_ids.append(self.ids[sample_mapping])

			sequence_ids = self.tokenized_data.sequence_ids(i)
			offset_mapping = self.tokenized_data["offset_mapping"][i]

			# TODO 3: set the offset mapping of the tokenized data at index i to (-1, -1)
			# if the token is not in the context
			assert len(sequence_ids) == len(offset_mapping)
			# get rid of query tokens
			offset_mapping = [(-1, -1) if x is not None and x == 0 else offset_mapping[_c] for _c, x in
			                  enumerate(sequence_ids)]
			#if i == 0:
			#	print(offset_mapping)

			self.tokenized_data["offset_mapping"][i] = offset_mapping
		self.tokenized_data["ID"] = example_ids

	@classmethod
	def from_config_dict(cls, wiki_doc, questions, tokenizer, config):
		return cls(wiki_doc, questions, tokenizer, **config)

	def __len__(self):
		# TODO 4: define the length of the dataset equal to total number of unique features (not the total number of datapoints)
		return len(self.tokenized_data["input_ids"])

	def __getitem__(self, index: int):
		# TODO 5: Return the tokenized dataset at the given index. Convert the various inputs to tensor using torch.tensor
		return {
			'input_ids': torch.tensor(self.tokenized_data['input_ids'][index]),
			'attention_mask': torch.tensor(self.tokenized_data['attention_mask'][index]),
			'offset_mapping': torch.tensor(self.tokenized_data['offset_mapping'][index]),
			'example_id': self.tokenized_data["ID"][index],
		}

