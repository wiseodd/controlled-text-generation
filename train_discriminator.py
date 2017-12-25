
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
lr = 1e-4
lr_decay_every = 1000000
n_iter = 50000
log_interval = 200
z_dim = h_dim

dataset = SST_Dataset()
# dataset = WikiText_Dataset()
# dataset = IMDB_Dataset()

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=False,
    gpu=args.gpu
)

# Load pretrained base VAE with c ~ p(c)
model.load_state_dict(torch.load('models/vae.bin'))


def main():
    trainer_D = optim.Adam(model.discriminator_params, lr=lr)
    trainer_VAE = optim.Adam(model.vae_params, lr=lr)

    for it in range(n_iter):
        inputs, labels = dataset.next_batch(args.gpu)

        # Update discriminator
        y_disc = model.forward_discriminator(inputs.transpose(0, 1))
        x_gen = model

        loss_s = F.cross_entropy(y_disc, labels)
        loss_u = F

        loss = recon_loss + kl_weight(it) * kl_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.discriminator_params, 5)
        trainer.step()
        trainer.zero_grad()

        recon_loss, kl_loss = model.forward(inputs)
        vae_loss = recon_loss + kl_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.discriminator_params, 5)
        trainer.step()
        trainer.zero_grad()

        if it % log_interval == 0:
            # Reconstruct first sentence in current batch
            orig_sentence = inputs[:, 0].unsqueeze(1)

            z_mean, z_logvar = model.forward_encoder(orig_sentence)
            # z = model.sample_z(z_mean, z_logvar)

            sample_idxs = model.sample_sentence(z_mean)
            sample_sent = dataset.idxs2sentence(sample_idxs)

            print('Iter-{}; Loss: {:.4f}; Recon: {:.4f}; KL: {:.4f}; Grad_norm: {:.4f};'
                  .format(it, loss.data[0], recon_loss.data[0], kl_loss.data[0], grad_norm))

            orig_idxs = orig_sentence.squeeze().data
            orig_idxs = orig_idxs.cpu() if args.gpu else orig_idxs

            print('Original: "{}"'.format(dataset.idxs2sentence(orig_idxs.numpy())))
            print('Reconstruction: "{}"'.format(sample_sent))
            print()

        # Anneal learning rate
        new_lr = lr * (0.5 ** (it // lr_decay_every))
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr


def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/ctextgen.bin')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        save_model()

    save_model()
