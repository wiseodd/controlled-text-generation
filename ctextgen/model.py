import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNN_VAE(nn.Module):
    """
    Hu, Zhiting, et al. "Toward controlled generation of text." ICML. 2017.
    """

    def __init__(self, n_vocab, h_dim, z_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3, max_sent_len=15, pretrained_embeddings=None, freeze_embeddings=False, gpu=False):
        super(RNN_VAE, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx
        self.MAX_SENT_LEN = max_sent_len

        self.n_vocab = n_vocab
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.p_word_dropout = p_word_dropout

        self.gpu = gpu

        # Embeddings layer
        if pretrained_embeddings is None:
            self.emb_dim = h_dim
            self.word_emb = nn.Embedding(n_vocab, h_dim)
        else:
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

        # Encoder
        self.encoder = nn.GRU(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)

        # Decoder
        self.decoder = nn.GRU(self.emb_dim+self.z_dim, h_dim)
        self.decoder_fc = nn.Linear(h_dim, n_vocab)

        # Discriminator
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.AvgPool2d(1)
        )

        self.discriminator = nn.Linear(128, 2)

        if self.gpu:
            self.cuda()

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward_encoder(self, inputs):
        inputs = self.word_emb(inputs)

        _, h = self.encoder(inputs, None)

        # Forward to latent
        h = h.view(-1, self.h_dim)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        return mu + torch.exp(logvar/2) * eps

    def forward_decoder(self, inputs, init_h):
        dec_inputs = self.word_dropout(inputs)

        # Forward
        seq_len = dec_inputs.size(0)

        init_h = init_h.unsqueeze(0)  # 1 x mbsize x h_dim

        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(-1, self.h_dim)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y

    def forward_discriminator(self, inputs):
        features = self.cnn(inputs)
        features = features.view(features.size(1), -1)
        y = self.discriminator(features)

        return y

    def forward(self, sentence):
        """
        Params:
        -------
        sentence: sequence of word indices.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """
        self.train()

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: 'I want to fly'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)

        enc_inputs = sentence[1:-1, :]
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar = self.forward_encoder(enc_inputs)
        z = self.sample_z(mu, logvar)

        # Decoder: sentence -> y
        y = self.forward_decoder(sentence, z)

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))

        return recon_loss, kl_loss

    def sample_sentence(self, z, stochastic=True):
        self.eval()

        word = torch.LongTensor([self.START_IDX])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        z = z.view(1, 1, -1)
        h = z.clone()

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []

        for i in range(self.MAX_SENT_LEN):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z], 2)

            output, h = self.decoder(emb, h)
            y = self.decoder_fc(output)

            # Sample next word
            y = F.softmax(y.view(-1), dim=0)
            y = y.cpu() if self.gpu else y
            y = y.data.numpy()

            if stochastic:
                idx = np.random.multinomial(1, y).argmax()
            else:
                idx = np.argmax(y)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            if idx == self.EOS_IDX:
                break

            outputs.append(idx)

        return outputs

    def word_dropout(self, inputs):
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                     .astype('uint8')
        )

        if self.gpu:
            mask = mask.cuda()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)
