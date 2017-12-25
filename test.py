import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from ctextgen.dataset import SST_Dataset
from ctextgen.model import RNN_VAE

import argparse
import random


parser = argparse.ArgumentParser(
    description='Conditional Text Generation'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')

args = parser.parse_args()


mb_size = 32
z_dim = 100
h_dim = 128
lr = 1e-3
lr_decay_every = 5000
n_iter = 20000
log_interval = 200
z_dim = h_dim

dataset = SST_Dataset()

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)

model.load_state_dict(torch.load('models/model.bin'))

# Reconstruct first sentence in current batch
idx = random.randint(0, 32)

orig_sentence = dataset.next_batch()[0][:, idx]
orig_idxs = orig_sentence.data.numpy().astype(int)

print('Original sentence: {}'.format(dataset.idxs2sentence(orig_idxs)))

# Use the mean
z, _ = model.forward_encoder(orig_sentence.unsqueeze(1))

sample_idxs = model.sample_sentence(z, stochastic=False)
print('Reconstruction (MAP): {}'.format(dataset.idxs2sentence(sample_idxs)))

sample_idxs = model.sample_sentence(z, stochastic=True)
print('Reconstruction (Sampling): {}'.format(dataset.idxs2sentence(sample_idxs)))

print()

# Interpolation
z1 = Variable(torch.zeros(z_dim) - 0.5).view(1, 1, z_dim)
z2 = Variable(torch.zeros(z_dim) + 0.5).view(1, 1, z_dim)

# Interpolation coefficients
alphas = np.linspace(0, 1, 5)

for alpha in alphas:
    z = float(1-alpha)*z1 + float(alpha)*z2

    sample_idxs = model.sample_sentence(z, stochastic=False)
    sample_sent = dataset.idxs2sentence(sample_idxs)

    print("{}".format(sample_sent))

print()
