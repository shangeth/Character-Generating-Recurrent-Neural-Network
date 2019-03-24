
import numpy as np
from data_process import *
import argparse


def encode_text(text):
	char = list(set(text))
	int2char = dict(enumerate(char))
	char2int = {ch: i for i, ch in int2char.items()}
	encoded = np.array([char2int[ch] for ch in text])
	return char, int2char, char2int, encoded


def onehot(arr, nlabels):
	one_hot = np.zeros((np.multiply(*arr.shape), nlabels), dtype=np.float32)
	one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
	one_hot = one_hot.reshape((*arr.shape, nlabels))
	return one_hot


def batcher(arr, batch_size, seq_len):
	n_batches = len(arr)//(batch_size * seq_len)
	arr = arr[:(n_batches * batch_size * seq_len)]
	arr = arr.reshape((batch_size, -1))
	for n in range(0, arr.shape[1], seq_len):
		x = arr[:, n:n+seq_len]
		y = np.zeros_like(x)
		try:
		    y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_len]
		except IndexError:
		    y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
		yield x, y

def main():
	DATA_FILE = '/content/data.txt'
	with open(DATA_FILE, 'r') as f:
	  text = f.read()
	char, int2char, char2int, encoded = encode_text(text)


if __name__ == '__main__':
	main()