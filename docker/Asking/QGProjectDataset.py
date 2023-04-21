import re
from typing import List
from torch.utils.data import (
	Dataset
)


class QGProjectDataset(Dataset):
	def split_wikidoc_into_paragraphs(self, text: str) -> List[str]:
		"""Splits a long text into segments short enough to be input into the transformer network.
		Segments are used as context for question generation.
		"""
		paragraphs = text.split("\n") # split into paragraphs
		paragraphs = [p for p in paragraphs if len(p) > 0] # remove empty paragraphs
		return paragraphs

	def split_paragraph_into_sentences(self, text: str) -> List[str]:
		#par_sents = [re.findall(r".*?[.!\?]", par) for par in paragraphs]  # split into list of sentences
		par_sents = re.split(r"[,;:)]", text) # split into smaller sentences
		# par_sents = [sent for sublist in par_sents for sent in sublist]  # flatten list of sentences
		# par_sents = [s.strip() for s in par_sents if len(s) > 0] # remove empty sentences
		par_sents = [s for s in par_sents if len(s.split(" ")) > 5]  # remove empty sentences
		return list(set([s.strip(' ') for s in par_sents]))  # Get only unique sentences

	def __init__(self, wiki_doc, tokenizer, ans_tok, context_tok, max_length, stride, truncation, padding,
	             return_overflowing_tokens, return_offsets_mapping, **kwargs):
		self.wiki_doc = wiki_doc
		self.wiki_doc_pars = self.split_wikidoc_into_paragraphs(wiki_doc)
		self.tokenizer = tokenizer
		self.answer_token = ans_tok
		self.context_token = context_tok
		self.max_length = max_length
		self.stride = stride
		self.truncation = truncation
		self.padding = padding
		#self.wiki_doc_tok = self.tokenizer(wiki_doc, max_length=max_length, stride=stride, truncation=truncation, padding=padding)


	def __len__(self):
		# TODO 4: define the length of the dataset equal to total number of unique features (not the total number of datapoints)
		#return len(self.tokenized_data["input_ids"])
		return len(self.wiki_doc_pars)

	def __getitem__(self, index: int):
		# Tokenize data on the fly to make the most out of multiple cpu processes
		paragraph = self.wiki_doc_pars[index]
		paragraph_sentences = self.split_paragraph_into_sentences(paragraph)
		if paragraph_sentences:
			candidate_inputs = [f"{self.answer_token} {sentence} {self.context_token} {paragraph}" for sentence in paragraph_sentences]

			#print(f"index: {index}")
			enconded_candidate_inputs = self.tokenizer.batch_encode_plus(
				candidate_inputs, padding=self.padding, max_length=self.max_length,
	            truncation=self.truncation, return_tensors="pt",)

			return enconded_candidate_inputs, paragraph_sentences
		else:
			#return [], []
			return None, None

	def filter_empty_paragraphs_collate_fn(self, batch):
		"""Collate function that filters out empty paragraphs"""
		#batch = [batch for (inputs, sentences) in batch if inputs]
		batch = list(filter(lambda x: x[0] is not None, batch))
		return batch
		#return default_collate(batch)
