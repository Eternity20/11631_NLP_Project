import torch
from tqdm import tqdm
from torch.utils.data import (
	DataLoader,
	Dataset
)

class QADataset(Dataset):
	def __init__(
			self,
			data,
			tokenizer,
			config
	):

		self.config = config
		self.data = data
		self.tokenizer = tokenizer
		self.tokenized_data = self.tokenizer(
			self.data["question"],
			self.data["context"],
			max_length=self.config["max_length"],
			stride=self.config["stride"],
			truncation=self.config["truncation"],
			padding=self.config["padding"],
			return_overflowing_tokens=self.config["return_overflowing_tokens"],
			return_offsets_mapping=self.config["return_offsets_mapping"],
			return_attention_mask=True,
			add_special_tokens=True
		)

		example_ids = []
		for i, sample_mapping in enumerate(tqdm(self.tokenized_data["overflow_to_sample_mapping"])):
			example_ids.append(self.data["id"][sample_mapping])

			sequence_ids = self.tokenized_data.sequence_ids(i)
			offset_mapping = self.tokenized_data["offset_mapping"][i]

			# TODO 3: set the offset mapping of the tokenized data at index i to (-1, -1)
			# if the token is not in the context
			assert len(sequence_ids) == len(offset_mapping)
			# get rid of query tokens
			offset_mapping = [(-1, -1) if x is not None and x == 0 else offset_mapping[_c] for _c, x in
			                  enumerate(sequence_ids)]
			if i == 0:
				print(offset_mapping)

			self.tokenized_data["offset_mapping"][i] = offset_mapping
		self.tokenized_data["ID"] = example_ids

	def __len__(
			self
	):
		# TODO 4: define the length of the dataset equal to total number of unique features (not the total number of datapoints)
		return len(self.tokenized_data["input_ids"])

	def __getitem__(
			self,
			index: int
	):
		# TODO 5: Return the tokenized dataset at the given index. Convert the various inputs to tensor using torch.tensor
		return {
			'input_ids': torch.tensor(self.tokenized_data['input_ids'][index]),
			'attention_mask': torch.tensor(self.tokenized_data['attention_mask'][index]),
			'offset_mapping': torch.tensor(self.tokenized_data['offset_mapping'][index]),
			'example_id': self.tokenized_data["ID"][index],
		}

	eval_data = eval_dataset.data
	eval_features = eval_dataset.tokenized_data

