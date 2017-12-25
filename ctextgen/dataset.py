from torchtext import data, datasets
from torchtext.vocab import GloVe


class SST_Dataset:

    def __init__(self, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=16)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15

        train, val, test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False,
            filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=100))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)

        self.train_iter, _, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, device=-1
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])


class IMDB_Dataset:

    def __init__(self, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=None)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        f = lambda ex: len(ex.text) <= 15

        train, test = datasets.IMDB.splits(
            self.TEXT, self.LABEL, filter_pred=f
        )

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=100))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)

        self.train_iter, _ = data.BucketIterator.splits(
            (train, test), batch_size=mbsize, device=-1
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])


class WikiText_Dataset:

    def __init__(self, mbsize=32):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy', fix_length=None)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        train, val, test = datasets.WikiText2.splits(self.TEXT)

        self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=100))
        self.LABEL.build_vocab(train)

        self.n_vocab = len(self.TEXT.vocab.itos)

        self.train_iter, _, _ = data.BPTTIterator.splits(
            (train, val, test), batch_size=10, bptt_len=15, device=-1
        )

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(iter(self.train_iter))
        return batch.text.cuda() if gpu else batch.text

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])
