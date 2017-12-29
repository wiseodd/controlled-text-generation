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
import time


parser = argparse.ArgumentParser(
    description='Conditional Text Generation'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--sentiment', default='positive', metavar='',
                    help='choose the generated sentiment: {`positive`, `negative`}, (default: `positive`)')

args = parser.parse_args()


mb_size = 32
z_dim = 100
h_dim = 128
lr = 1e-3
lr_decay_every = 5000
n_iter = 20000
log_interval = 200
z_dim = h_dim
c_dim = 2

dataset = SST_Dataset()

torch.manual_seed(int(time.time()))

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)

if args.gpu:
    model.load_state_dict(torch.load('models/ctextgen.bin'))
else:
    model.load_state_dict(torch.load('models/ctextgen.bin', map_location=lambda storage, loc: storage))

# Samples latent and conditional codes randomly from prior
z = model.sample_z_prior(1)
c = model.sample_c(1)

if args.sentiment == 'positive':
    c[0, 0], c[0, 1] = 1, 0
else:
    c[0, 0], c[0, 1] = 0, 1

_, c_idx = torch.max(c, dim=1)
print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))

sample_idxs = model.sample_sentence(z, c, stochastic=False)
print('Reconstruction (MAP): {}'.format(dataset.idxs2sentence(sample_idxs)))

sample_idxs = model.sample_sentence(z, c, stochastic=True)
print('Reconstruction (Sampling): {}'.format(dataset.idxs2sentence(sample_idxs)))

print()

# Interpolation
z1 = Variable(torch.zeros(z_dim) - 3).view(1, 1, z_dim)
z2 = Variable(torch.zeros(z_dim) + 3).view(1, 1, z_dim)
c = model.sample_c(1)

# Interpolation coefficients
alphas = np.linspace(0, 1, 5)

for alpha in alphas:
    z = float(1-alpha)*z1 + float(alpha)*z2

    sample_idxs = model.sample_sentence(z, c, stochastic=True)
    sample_sent = dataset.idxs2sentence(sample_idxs)

    print("{}".format(sample_sent))

print()
