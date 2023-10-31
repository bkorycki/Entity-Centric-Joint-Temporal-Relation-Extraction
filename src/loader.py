import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from typing import List, Dict

class TemporalRelationDataset(Dataset):
	def __init__(self, tokenizer, docs: List, label_mapping: Dict, max_length: int):
		
		self.data = []
		self.tokenizer = tokenizer
		self.label_mapping = label_mapping
		self.max_length = max_length

		for doc in docs:
			doc.tokenize(self.tokenizer, self.max_length)
			if len(doc.tokenized_labels)>0:
				self.data.append(doc)

	def label_repr(self, doc):
		"""
		Returns flattened list of label indicies (corresponding to label_mapping)
		"""
		labels = [self.label_mapping[l] for l in doc.tokenized_labels]
		return torch.tensor(labels)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		item = self.data[index]
		return {
			"doc_id": item.doc_id,
			"input_ids": torch.LongTensor(item.input_ids),
			"med_spans": item.tokenized_med_spans,
			"date_spans": item.tokenized_date_spans,
			"labels": self.label_repr(item)
		}

def collator(batch, pad_id):
	'''
	Collate batch (list of of docs) and pad according to max batch len.
	Returns dict. with keys:
		input_ids, masks, med_spans, data_spans, num_meds, num_dates, labels, doc_ids
	'''
	collate_data = {"input_ids": [], "masks": [], "med_spans": [], "date_spans": [], "num_meds": [], "num_dates": [], "labels": [], "doc_ids": [doc["doc_id"] for doc in batch]}

	for key in ["med_spans", "date_spans", "labels"]:
		collate_data[key] =[doc[key] for doc in batch]

	batch_input_ids = [doc["input_ids"] for doc in batch]
	padded_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_id)
	collate_data["input_ids"] = padded_input_ids
	collate_data["masks"]  = (padded_input_ids != pad_id).float()

	max_label_length = max([len(label) for label in collate_data["labels"]])
	collate_data["labels"] = torch.stack([F.pad(label, pad=(0, max_label_length-len(label)), value=-100) for label in collate_data["labels"]])
	collate_data["max_label_length"] = max_label_length

	# for key in ["labels"]:#["med_spans", "date_spans", "labels"]:
	# 	batch_data = [doc[key] for doc in batch]
	# 	collate_data[key] = pad_sequence(batch_data, batch_first=True, padding_value=-pad_id)

	for doc in batch:
		collate_data["num_meds"].append(len(doc["med_spans"]))
		collate_data["num_dates"].append(len(doc["date_spans"]))

	return collate_data


def get_dataloader(tokenizer, data: list, label_mapping: dict, max_length=128, batch_size=8, shuffle=True):
	dataset = TemporalRelationDataset(tokenizer, data, label_mapping, max_length)
	wrapped_collator = lambda batch: collator(batch, tokenizer.pad_token_id)

	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=wrapped_collator)
