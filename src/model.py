import torch.nn as nn
from transformers import  RobertaModel
import torch
from util import to_cuda, pad_and_stack

class Model(nn.Module):
	def __init__(self, vocab_size, out_dim=7, model_name="roberta-base", embed_dim=768, aggr="mean"):
		nn.Module.__init__(self)
		self.encoder = EntityEncoder(vocab_size, model_name=model_name, aggr=aggr)
		self.scorer = PairScorer(embed_dim=embed_dim, out_dim=out_dim)

	def forward(self, inputs):
		output = self.encoder(inputs)
		output = self.scorer(*output)
		return output

class EntityEncoder(nn.Module):
	def __init__(self, vocab_size, model_name="roberta-base", aggr="mean"):
		nn.Module.__init__(self)
		self.model = RobertaModel.from_pretrained(model_name)
		self.model.resize_token_embeddings(vocab_size)
		# self.model = nn.DataParallel(self.model)
		self.aggr = aggr
	
	def forward(self, inputs):
		"""
		Returns: med_embeddings, time_embeddings
		"""
		# Compute doc. embedding
		input_ids = inputs["input_ids"]
		attention_mask = inputs["masks"]
		output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state
		med_embeds = self.aggregate_entity_representations(output, inputs["med_spans"])
		time_embeds = self.aggregate_entity_representations(output, inputs["date_spans"])

		return med_embeds, time_embeds
	
	def aggregate_entity_representations(self, doc_embeds, entity_spans):
		"""
		Hierarchical entity-embedding aggregation
		"""
		entity_embeds = []
		for i, doc_embed in enumerate(doc_embeds):
			doc_entity_embeds = []
			for entity_mentions in entity_spans[i]:
				if self.aggr == "mean":
					mention_embeds = [torch.mean(doc_embed[start_idx:end_idx], dim=0) for start_idx, end_idx in entity_mentions]
					entity_embedding = torch.mean(torch.stack(mention_embeds), dim=0)
				elif self.aggr == "max":
					mention_embeds = [doc_embed[start_idx:end_idx].max(0)[0] for start_idx, end_idx in entity_mentions]
					entity_embedding = torch.stack(mention_embeds).max(0)[0]
				else:
					raise NotImplementedError
				doc_entity_embeds.append(entity_embedding)
			entity_embeds.append(torch.stack(doc_entity_embeds))
		return entity_embeds

class Score(nn.Module):
	""" Generic scoring module
	"""
	def __init__(self, embeds_dim, out_dim=6, hidden_dim=150):
		super().__init__()

		self.score = nn.Sequential(
			nn.Linear(embeds_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.20),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.20),
			nn.Linear(hidden_dim, out_dim)
		)

	def forward(self, x):
		""" Output a scalar score for an input x """
		return self.score(x)

class PairScorer(nn.Module):
	def __init__(self, embed_dim=768, out_dim=6):
		nn.Module.__init__(self)
		self.out_dim = out_dim
		self.embed_dim = embed_dim
		self.score = Score(embed_dim * 2, out_dim=out_dim)
	
	def forward(self, med_embeds, time_embeds):
		all_scores = []
		for doc_ix in range(len(med_embeds)):
			doc_med_embeds = med_embeds[doc_ix]
			doc_time_embeds = time_embeds[doc_ix]
			med_id, time_id = zip(*[(m, t) for m in range(len(doc_med_embeds)) for t in range(len(doc_time_embeds))])
			med_id, time_id = to_cuda(torch.tensor(med_id)), to_cuda(torch.tensor(time_id))
			med_embed = torch.index_select(doc_med_embeds, 0, med_id)
			time_embed = torch.index_select(doc_time_embeds, 0, time_id)
			pairs = torch.cat((med_embed, time_embed), dim=1)
			scores = self.score(pairs)
			all_scores.append(scores)
	
		all_scores, sizes = pad_and_stack(all_scores, value=-1000)
		return all_scores

