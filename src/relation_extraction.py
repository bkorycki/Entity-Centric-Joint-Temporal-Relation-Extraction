I was diagnosed with bipolar 2 years ago and started taking Lamictal. Last April, my psych switched me to Abilify because of issues with my blood tests. I didn't have any side effects with the lamictal but now I feel just super antsy and restless on the abilify-- has anyone else had this issue?
import os
import sys
import warnings
import argparse
import random

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import Adam

from util import load_config, setup_log, to_cuda, save_training_history
from reader import read_docs
from loader import get_dataloader
from model import Model

warnings.filterwarnings("ignore")

def set_seed(seed=0):
	np.random.seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = False

def evaluate(model, dataloader, class_names, label_ids):
	"""
	Returns: loss, micro_f1
	"""
	losses = []
	pred_list = []
	label_list = []

	Loss = nn.CrossEntropyLoss(ignore_index=-100)
	model.eval()
	with torch.no_grad():
		for data in dataloader:
			for k in data:
				if isinstance(data[k], torch.Tensor):
					data[k] = to_cuda(data[k])
			scores = model(data)
			labels = data["labels"]
			scores = scores.view(-1, scores.size(-1))
			labels = labels.view(-1)

			loss = Loss(scores, labels)
			losses.append(loss.item())

			pred = torch.argmax(scores, dim=-1)
			pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
			label_list.extend(labels[labels>=0].cpu().numpy().tolist())

	loss = np.mean(losses)
	result = classification_report(label_list, pred_list, output_dict=True, target_names=class_names, labels=label_ids)

	if "micro avg" not in result:
		score = result["accuracy"]
	else:
		score = result["micro avg"]["f1-score"]
	
	return loss, score

def predict(model, dataloader):
	all_preds = []
	model.eval()
	with torch.no_grad():
		for data in tqdm(dataloader, desc="Predict"):
			for k in data:
				if isinstance(data[k], torch.Tensor):
					data[k] = to_cuda(data[k])
			scores = model(data)
			labels = data["labels"]
			scores = scores.view(-1, scores.size(-1))
			labels = labels.view(-1)
			pred = torch.argmax(scores, dim=-1)
			max_label_length = data["max_label_length"]
			n_doc = len(labels) // max_label_length
			assert len(labels) % max_label_length == 0
			for i in range(n_doc):
				selected_index = labels[i*max_label_length:(i+1)*max_label_length] >= 0
				all_preds.append({
					"doc_id": data["doc_id"][i],
					"preds": pred[i*max_label_length:(i+1)*max_label_length][selected_index].cpu().numpy().tolist(),
				})
	return all_preds


def train(params, train_data, val_data):
	output_dir = setup_log(params, "train")
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {device}")

	print('Loading data........')
	print(f"\t{len(train_data)} train docs, {len(val_data)} val docs")

	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	train_dataloader = get_dataloader(tokenizer, train_data, params['label_map'], params['max_length'], params['batch_size'], shuffle=True)
	val_dataloader = get_dataloader(tokenizer, val_data, params['label_map'], params['max_length'], params['batch_size'], shuffle=False)
	
	print(f"After processing:\n\t{len(train_dataloader.dataset)} train docs, {len(val_dataloader.dataset)} val docs")

	print("Loading model........")
	classes, label_ids = list(params["label_map"].keys()), list(params["label_map"].values())
	model = Model(len(tokenizer), out_dim=len(label_ids))
	model = to_cuda(model)
	print(model)

	bert_optimizer = AdamW([p for p in model.encoder.model.parameters() if p.requires_grad], lr=params['bert_lr'])
	optimizer = Adam([p for p in model.scorer.parameters() if p.requires_grad], lr=params['lr'])
	scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=200, num_training_steps=len(train_dataloader) * params['epochs'])

	Loss = nn.CrossEntropyLoss(ignore_index=-100)
	glb_step = 0
	print("******************** Training *********************")
	train_losses = []# [[step, loss]]
	val_losses = []
	val_f1s = []
	best_score = 0.0
	for epoch in range(params['epochs']):
		with tqdm(train_dataloader, unit="batch",  desc=f"Epoch {epoch}") as tepoch:
			for data in tepoch:
			# for data in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
				losses = []
				model.train()
				for k in data:
					if isinstance(data[k], torch.Tensor):
						data[k] = to_cuda(data[k])
				scores = model(data)
				labels = data["labels"]
				scores = scores.view(-1, scores.size(-1))
				labels = labels.view(-1)
				loss = Loss(scores, labels)

				losses.append(loss.item())
				loss.backward()
				optimizer.step()
				bert_optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				bert_optimizer.zero_grad()

				glb_step += 1

				if glb_step % params['log_steps'] == 0:
					loss = np.mean(losses)
					train_losses.append([glb_step, loss] )
					losses = []
					tepoch.set_postfix(loss=loss)
				if glb_step % params['eval_steps'] == 0:
					val_loss, val_f1 = evaluate(model, val_dataloader, classes, label_ids)
					val_losses.append([glb_step, val_loss])
					val_f1s.append([glb_step, val_f1])
					tepoch.set_postfix(val_loss=val_loss, val_f1=val_f1)

		if glb_step % params['eval_steps'] > 0: # Eval at end of epoch (if it wasn't already at the last step)
			val_loss, val_f1 = evaluate(model, val_dataloader, classes, label_ids)
			val_losses.append([glb_step, val_loss])
			val_f1s.append([glb_step, val_f1])
		# Update saved model if it's improved
		if val_f1s[-1][1] > best_score:
			print("Improved model performance")
			best_score = val_f1s[-1][1]
			state = {"model":model.state_dict(), "optimizer":optimizer.state_dict(), "scheduler": scheduler.state_dict()}
			torch.save(state, os.path.join(output_dir / "models", "best"))
	train_log = {"Train Loss": train_losses, "Val. Loss": val_losses, "Val. F1": val_f1s}
	save_training_history(params['epochs'], glb_step, output_dir, **train_log)
	sys.stdout.close()


def test(params, data):
	setup_log(params, "test")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Device: {device}")

	print(f'Loading data {len(data)} test docs...')

	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
	dataloader = get_dataloader(tokenizer, data, params.label_mapping, params.max_length, params.batch, shuffle=False, batch_size=params.batch_size, sample_rate=params.sample_rate)
	
	# print("*" * 30 + "Predict"+ "*" * 30)
	# if os.path.exists(os.path.join(output_dir, "best")):
	# 	print("loading from", os.path.join(output_dir, "best"))
	# 	state = torch.load(os.path.join(output_dir, "best"))
	# 	model.load_state_dict(state["model"])
	# all_preds = predict(model, test_dataloader)
	# dump_result("../data/MAVEN_ERE/test.jsonl", all_preds, output_dir, ignore_nonetype=args.ignore_nonetype)
	sys.stdout.close()

def main(config_name, eval_only):
	params = load_config(config_name)
	set_seed(params['seed'])

	print("Reading data....")
	docs = read_docs(params['data_path'])

	train_data, test_data = train_test_split(docs, train_size=int(params["train_split"]*len(docs)), random_state=params["seed"])
	test_data, val_data = train_test_split(test_data, train_size=int(params["test_split"]*len(docs)), random_state=params["seed"])

	if not eval_only:
		train(params, train_data, val_data)

	# test(params, test_data)
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', help='Config file name',  type=str, required=True)
	parser.add_argument("--eval_only", action="store_true")

	args = parser.parse_args()

	main(args.config, args.eval_only)

	args = parser.parse_args()

	