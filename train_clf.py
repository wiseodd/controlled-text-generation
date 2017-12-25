
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from ctextgen.dataset import *
from ctextgen.model import RNN_VAE

import argparse


parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')

args = parser.parse_args()


mb_size = 32
z_dim = 100
h_dim = 128
lr = 1e-3
lr_decay_every = 1000000
n_iter = 50000
log_interval = 200
z_dim = h_dim

dataset = SST_Dataset()
# dataset = WikiText_Dataset()
# dataset = IMDB_Dataset()


class Clf(nn.Module):

    def __init__(self):
        super(Clf, self).__init__()

        emb_dim = dataset.get_vocab_vectors().size(1)
        self.word_emb = nn.Embedding(dataset.n_vocab, emb_dim)
        # Set pretrained embeddings
        self.word_emb.weight.data.copy_(dataset.get_vocab_vectors())
        self.word_emb.weight.requires_grad = False

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d(1)
        )

        self.discriminator = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward(self, inputs):
        inputs = self.word_emb(inputs)
        inputs = inputs.unsqueeze(1)

        features = self.cnn(inputs)
        features = features.view(features.size(0), -1)

        y = self.discriminator(features)

        return y


model = Clf()
trainer = optim.Adam(model.trainable_parameters(), lr=lr, weight_decay=1e-4)

if args.gpu:
    model.cuda()


for it in range(n_iter):
    inputs, labels = dataset.next_batch(args.gpu)

    inputs = inputs.transpose(0, 1)  # mbsize x seq_len
    y = model.forward(inputs)

    loss = F.cross_entropy(y, labels)

    loss.backward()
    trainer.step()
    trainer.zero_grad()

    if it % log_interval == 0:
        accs = []

        # Test on validation
        for _ in range(20):
            inputs, labels = dataset.next_validation_batch(args.gpu)
            inputs = inputs.transpose(0, 1)

            _, y = model.forward(inputs).max(dim=1)

            acc = float((y == labels).sum()) / y.size(0)
            accs.append(acc)

        print('Iter-{}; loss: {:.4f}; val_acc: {:.4f}'.format(it, float(loss), np.mean(accs)))
