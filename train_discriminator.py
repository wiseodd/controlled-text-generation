
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
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train Discriminator'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=False, action='store_true',
                    help='whether to save model or not')

args = parser.parse_args()


mbsize = 20
z_dim = 20
h_dim = 64
lr = 1e-3
lr_decay_every = 1000000
n_iter = 5000
log_interval = 100
z_dim = h_dim
c_dim = 2
kl_weight_max = 0.4

# Specific hyperparams
beta = 0.1
lambda_c = 0.1
lambda_z = 0.1
lambda_u = 0.1

dataset = SST_Dataset(mbsize=mbsize)

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)

# Load pretrained base VAE with c ~ p(c)
model.load_state_dict(torch.load('models/vae.bin'))


def kl_weight(it):
    """
    Credit to: https://github.com/kefirski/pytorch_RVAE/
    0 -> 1
    """
    return (math.tanh((it - 3500)/1000) + 1)/2


def temp(it):
    """
    Softmax temperature annealing
    1 -> 0
    """
    return 1-kl_weight(it) + 1e-5  # To avoid overflow


def main():
    trainer_D = optim.Adam(model.discriminator_params, lr=lr)
    trainer_G = optim.Adam(model.decoder_params, lr=lr)
    trainer_E = optim.Adam(model.encoder_params, lr=lr)

    for it in tqdm(range(n_iter)):
        inputs, labels = dataset.next_batch(args.gpu)

        """ Update discriminator, eq. 11 """
        batch_size = inputs.size(1)
        # get sentences and corresponding z
        x_gen, c_gen  = model.generate_sentences(batch_size)  # mbsize x 16

        y_disc_real = model.forward_discriminator(inputs.transpose(0, 1))
        y_disc_fake = model.forward_discriminator(x_gen)

        log_y_disc_fake = F.log_softmax(y_disc_fake, dim=1)
        entropy = -log_y_disc_fake.mean()

        loss_s = F.cross_entropy(y_disc_real, labels)
        # eq. 10 
        target_c = torch.argmax(c_gen, dim=1)
        loss_u = F.cross_entropy(y_disc_fake, target_c) + beta*entropy

        loss_D = loss_s + lambda_u*loss_u

        loss_D.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.discriminator_params, 5)
        trainer_D.step()
        trainer_D.zero_grad()

        """ Update generator, eq. 8 """
        # Forward VAE with c ~ q(c|x) instead of from prior
        recon_loss, kl_loss = model.forward(inputs, use_c_prior=False)
        # x_gen: mbsize x seq_len x emb_dim
        x_gen_attr, target_z, target_c = model.generate_soft_embed(batch_size, temp=temp(it))

        # y_z: mbsize x z_dim
        y_z, _ = model.forward_encoder_embed(x_gen_attr.transpose(0, 1))
        y_c = model.forward_discriminator_embed(x_gen_attr)

        loss_vae = recon_loss + kl_weight_max * kl_loss
        loss_attr_c = F.cross_entropy(y_c, target_c)
        loss_attr_z = F.mse_loss(y_z, target_z)

        loss_G = loss_vae + lambda_c*loss_attr_c + lambda_z*loss_attr_z

        loss_G.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.decoder_params, 5)
        trainer_G.step()
        trainer_G.zero_grad()

        """ Update encoder, eq. 4 """
        recon_loss, kl_loss = model.forward(inputs, use_c_prior=False)

        loss_E = recon_loss + kl_weight_max * kl_loss

        loss_E.backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.encoder_params, 5)
        trainer_E.step()
        trainer_E.zero_grad()

        if it % log_interval == 0:
            z = model.sample_z_prior(1)
            c = model.sample_c_prior(1)

            sample_idxs = model.sample_sentence(z, c)
            sample_sent = dataset.idxs2sentence(sample_idxs)

            print('Iter-{}; loss_D: {:.4f}; loss_G: {:.4f}'
                  .format(it, float(loss_D), float(loss_G)))

            _, c_idx = torch.max(c, dim=1)

            print('c = {}'.format(dataset.idx2label(int(c_idx))))
            print('Sample: "{}"'.format(sample_sent))
            print()


def save_model():
    if not os.path.exists('models/'):
        os.makedirs('models/')

    torch.save(model.state_dict(), 'models/ctextgen.bin')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if args.save:
            save_model()

        exit(0)

    if args.save:
        save_model()
