import torch
import torch.utils
import torch.utils.data
import numpy as np


def batch2RNNinput(x_batch, device):
	x_data = x_batch[:, :-1]
	x_lengths = x_batch[:, -1:]
	x_lengths = x_lengths.sort(axis=0)[0]
	x_lengths = np.flip(np.array(x_lengths.cpu()), axis=0).copy()
	x_lengths = torch.tensor([float(x) for x in x_lengths])

	x_data = x_data.sort(axis=0)[0]
	x_data = np.flip(np.array(x_data.cpu()), axis=0).copy()
	x_data = x_data.T
	x_data = torch.from_numpy(x_data)

	return x_data.to(device), x_lengths.to(device)


def accuracy(model, data, device, mode):
	"""Return model accuracy"""
	model.eval()
	with torch.no_grad():
		val_acc = 0
		total = 0
		for x, y in data:
			x = x.long().to(device)
			y = y.long().to(device)

			if mode == 'rnn':
				x_data, x_len = batch2RNNinput(x, device)
				out = model(x_data, x_len).squeeze(1)
			if mode == 'cnn':
				out = model(x)

			total += x.size(0)
			pred = torch.max(out, dim=1)[1]
			val_acc += torch.sum(pred == y).item()

		return val_acc / total


def train(model, train_set, optimizer, criterion, device, mode='cnn'):
	epoch_loss = 0
	epoch_acc = 0

	for x, y in train_set:
		model.train()
		optimizer.zero_grad()

		x = x.long().to(device)
		y = y.long().to(device)

		if mode == 'rnn':
			x_data, x_len = batch2RNNinput(x, device)
			output = model(x_data, x_len).squeeze(1)
		if mode == 'cnn':
			output = model(x)

		loss = criterion(output, y)

		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += accuracy(model, train_set, device, mode)

	return epoch_loss / len(train_set), epoch_acc / len(train_set)


def evaluate(model, eval_set, criterion, device, mode):
	epoch_loss = 0
	epoch_acc = 0

	model.eval()

	with torch.no_grad():
		for x, y in eval_set:

			x = x.long().to(device)
			y = y.long().to(device)

			if mode == 'rnn':
				x_data, x_len = batch2RNNinput(x, device)
				output = model(x_data, x_len).squeeze(1)
			if mode == 'cnn':
				output = model(x)

			loss = criterion(output, y)

			epoch_loss += loss.item()
			epoch_acc += accuracy(model, eval_set, device, mode)

		return epoch_loss / len(eval_set), epoch_acc / len(eval_set)
