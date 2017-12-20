import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from lm import repackage_hidden, LM_LSTM
import time
import numpy as np
'''
Checkout tutorial LSTM: https://github.com/deeplearningathome/pytorch-language-model/blob/master/lm.py
Plus, check out library `torchtext` buat loading dataset (kita pake SST as in paper): https://github.com/pytorch/text
'''
# Approach 1:
# set up fields
# TEXT = data.Field()
# LABEL = data.Field(sequential=False)

# make splits for data
# train, val, test = datasets.SST.splits(
#     TEXT, LABEL, fine_grained=True, train_subtrees=True,
#     filter_pred=lambda ex: ex.label != 'neutral')

# print information about the data
# print('train.fields', train.fields)
# print('len(train)', len(train))
# print('vars(train[0])', vars(train[0]))

# build the vocabulary
# url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
# TEXT.build_vocab(train, vectors=Vectors('wiki.simple.vec', url=url))
# LABEL.build_vocab(train)

# print vocab information
# print('len(TEXT.vocab)', len(TEXT.vocab))
# print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
# for text in train:
	# for word in example.text:
	# print(text.text)
# 		print('TEXT.vocab.stoi', TEXT.vocab.stoi[word])
# 		print('TEXT.vocab.itos', TEXT.vocab.itos[TEXT.vocab.stoi[word]])

# make iterator for splits
# train_iter, val_iter, test_iter = datasets.SST.iters(vectors=Vectors('wiki.simple.vec', url=url), batch_size=5)

# print(len(TEXT.vocab.stoi))
# print(len(TEXT.vocab.itos))

# for i in range(len(train)*2):
# batch = next(iter(train_iter))
# print(batch.text)
# print(batch.label)
	# print(batch.text)
	# for j in batch.text:
		# print(i, end=" ")
		# for k in j:
			# print(TEXT.vocab.itos[k.data[0]], end=" ")
		# print()

# time_steps = 10
seq_size = 6
batch_size = 5
num_steps = 16
# in_size = 5
# classes_no = 7

# ===============================================================================================

url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'

# set up fields
TEXT = data.Field()
LABEL = data.Field(sequential=False)
train, val, test = datasets.SST.splits(TEXT, LABEL)
# train_iter, val_iter, test_iter = data.Iterator.splits(datasets=(train, val, test), batch_size=5)
# train_iter, val_iter, test_iter = datasets.SST.iters(batch_size=20, device=0, root='.data', vectors=Vectors('wiki.simple.vec', url=url))
TEXT.build_vocab(train, vectors=Vectors('wiki.simple.vec', url=url))
LABEL.build_vocab(train)
train_seq = torch.cuda.LongTensor(0, seq_size)
target_seq = torch.cuda.LongTensor(0, 1)
i = 0
for training in train:
	# for idx in training.text:
	# 	print(idx)
	if len(training.text) < seq_size:
		continue
	sent_idx = [TEXT.vocab.stoi[word] for word in training.text]
	for idx in range(0, len(sent_idx)+1-seq_size):
		new_train_seq = torch.cuda.LongTensor([sent_idx[idx:idx+seq_size-1]]).cuda()
		train_seq = torch.cat((train_seq, new_train_seq), 0).cuda()

		new_target_seq = torch.cuda.LongTensor([sent_idx[idx:idx+seq_size-1]]).cuda()
		target_seq = torch.cat((target_seq, new_target_seq), 0).cuda()
	i+=1
	if i > 20:
		break

# train_iter = data.BucketIterator(dataset=train_seq, batch_size=batch_size)
train_dataset = data_utils.TensorDataset(train_seq, target_seq)
# print(train_dataset)
train_iter = data_utils.DataLoader(train_dataset, batch_size=num_steps, shuffle=True)
# for idx, batch in enumerate(train_iter):
# 	print(idx, batch)
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 30
HIDDEN_DIM = 20

criterion = nn.CrossEntropyLoss()
def run_epoch(model, data, is_train=False, lr=1.0, prt_out=False):
	"""Runs the model on the given data."""
	model.train()

	epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
	start_time = time.time()
	hidden = model.init_hidden()
	costs = 0.0
	iters = 0

	for step, (x, y) in enumerate(train_iter):
		if len(x) < num_steps:
			continue
		inputs = Variable(x.contiguous()).cuda()
		# print("inputs", inputs)

		targets = Variable(y.contiguous()).cuda()
		# print("targets", targets.size())
		model.zero_grad()
		hidden = repackage_hidden(hidden)
		outputs, hidden = model(inputs, hidden)
		tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))
		if prt_out:
			for o in outputs:
				for w in o:
					val, idx = torch.max(w, 0)
					# print(m)
					print(TEXT.vocab.itos[idx.data[0]], end=" ")
				print()
	loss = criterion(outputs.view(-1, model.vocab_size), tt)
	costs += loss.data[0] * model.num_steps
	iters += model.num_steps

	loss.backward()
	torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
	
	for p in model.parameters():
		p.data.add_(-lr, p.grad.data)
	if step % 30 == 0:
		print("{} perplexity: {:8.2f} speed: {} wps".format(step * 1.0 / epoch_size, np.exp(costs / iters),iters * model.batch_size / (time.time() - start_time)))
	
	return np.exp(costs / iters)

model = LM_LSTM(embedding_dim=HIDDEN_DIM, num_steps=num_steps, batch_size=batch_size, vocab_size=18280, 
	num_layers=EMBEDDING_DIM, dp_keep_prob=0.9)
model.cuda()
lr = 20
# decay factor for learning rate
lr_decay_base = 1 / 1.15
# we will not touch lr for the first m_flat_lr epochs
m_flat_lr = 14.0

for epoch in range(300):
	prt = False
	if epoch % 30 == 0:
		prt = True
	else:
		prt = False
	lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
	lr = lr * lr_decay # decay lr if it is time
	train_p = run_epoch(model, train_seq, True, lr, prt)
	print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
	# print('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, run_epoch(model, valid_data)))

# ====================================================================================================
# class LSTMTagger(nn.Module):

# 	def __init__(self, embedding_dim, hidden_dim, vocab_size, vector_size):
# 		super(LSTMTagger, self).__init__()
# 		self.hidden_dim = hidden_dim

# 		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

# 		# The LSTM takes word embeddings as inputs, and outputs hidden states
# 		# with dimensionality hidden_dim.
# 		self.lstm = nn.LSTM(embedding_dim, hidden_dim)

# 		# The linear layer that maps from hidden state space to tag space
# 		self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
# 		self.hidden = self.init_hidden()

# 	def init_hidden(self):
# 		# Before we've done anything, we dont have any hidden state.
# 		# Refer to the Pytorch documentation to see exactly
# 		# why they have this dimensionality.
# 		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
# 		return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
# 				autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))

# 	def forward(self, sentence):
# 		embeds = self.word_embeddings(sentence)
# 		print("embeds", embeds)
# 		lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
# 		tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
# 		tag_scores = F.log_softmax(tag_space)
# 		return tag_scores

# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, VOCAB_SIZE)
# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# model.cuda()

# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
# 	# for sentence, tags in training_data:
# 	for idx, batch in enumerate(train_iter):
# 		# Step 1. Remember that Pytorch accumulates gradients.
# 		# We need to clear them out before each instance
# 		model.zero_grad()

# 		# Also, we need to clear out the hidden state of the LSTM,
# 		# detaching it from its history on the last instance.
# 		model.hidden = model.init_hidden()

# 		# Step 2. Get our inputs ready for the network, that is, turn them into
# 		# Variables of word indices.
# 		# sentence_in = prepare_sequence(sentence, word_to_ix)
# 		sentence_in = batch[0][0]
# 		targets = batch[1][0]
# 		print(sentence_in)
# 		print(targets)
# 		# targets = prepare_sequence(tags, tag_to_ix)

# 		# Step 3. Run our forward pass.
# 		tag_scores = model(sentence_in)
# 		print(tag_scores.size())
# 		loss = loss_function(tag_scores, targets)
# 		itr += 1
# 		# Step 4. Compute the loss, gradients, and update the parameters by
# 		#  calling optimizer.step()
# 		loss.backward()
# 		optimizer.step()
# 		if i%100 == 0:
# 			print(loss.data[0])
# 	# if epoch % 10 == 0:
# 	# 	print(str(epoch)+" ", end="")
# 	# 	for score in tag_scores:
# 	# 		values, indices = torch.max(score, 0)
# 	# 		# print(indices)
# 	# 		print(TEXT.vocab.itos[indices.data[0]], end=" ")
# 	# 	print()