import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
import yaml
import yamlordereddictloader
import torch
import torch.nn.functional as F
from pathlib import Path 
from pprint import pprint


def load_config(fname):
	with open(f"configs/{fname}.yaml", 'r') as f:
		parameters = yaml.load(f, Loader=yamlordereddictloader.Loader)
	parameters = dict(parameters)
	return parameters

def setup_log(config, mode):
	"""
	Setup .log file to record training process and results.

	Args:
		config (dict): model parameters
		mode (str): 'train' | 'test'
	"""
	output_dir = Path(f"output/{config['seed']}_ignorenone_{config['ignore_none_rel']}_batch_{config['batch_size']}_epochs_{config['epochs']}")
	output_dir.mkdir(exist_ok=True)
	model_dir = output_dir / "models"
	model_dir.mkdir(exist_ok=True)
	sys.stdout = open(os.path.join(output_dir, f"{mode}_log.txt"), 'w')

	print("CONFIG:")
	pprint(config)
	return output_dir

def to_cuda(x):
	""" GPU-enable a tensor """
	if torch.cuda.is_available():
		x = x.cuda()
	return x

def pad_and_stack(tensors, pad_size=None, value=0):
	""" 
	Pad and stack an uneven tensor of token lookup ids.
	Assumes batch_first=True
	"""
	# Get their original sizes (measured in number of tokens)
	sizes = [s.shape[0] for s in tensors]
	if not pad_size:
		pad_size = max(sizes)

	# Pad all sentences to the max observed size
	# TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
	padded = torch.stack([F.pad(input=sent[:pad_size],
								pad=(0, 0, 0, max(0, pad_size-size)),
								value=value)
						  for sent, size in zip(tensors, sizes)], dim=0)

	return padded, sizes

def save_training_history(num_epochs, num_steps, dir, **metric_logs):
	with open(dir / 'train_history.pkl', 'wb') as file:
		pickle.dump(metric_logs, file)

	for metric, log in metric_logs.items():
		steps, vals = zip(*log)
		plt.plot(steps, vals, label=metric)

	tick_locations = np.linspace(0, num_steps, num_epochs)
	tick_labels = list(range(1, num_epochs + 1))
	plt.xticks(tick_locations, tick_labels)
	plt.xlabel('Epochs')
	plt.ylabel('Value')
	plt.legend()
	plt.title('Training History')
	plt.grid(True)
	# Save the figure to a file
	plt.savefig(dir / "training_history.png", bbox_inches='tight', dpi=300)
