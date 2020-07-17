#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import pickle
import numpy as np
import os
import math
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import homogeneity_score as Hom
from sklearn.metrics import v_measure_score as VM
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import completeness_score as Com

import data

from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from detm import DETM
from utils import nearest_neighbors, get_topic_coherence

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='un', help='name of corpus')
parser.add_argument('--data_path', type=str, default='un/', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='skipgram/embeddings.txt', help='directory containing embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='number of documents in a batch for training')
parser.add_argument('--min_df', type=int, default=100, help='to get the right data..minimum document frequency')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')
parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')
parser.add_argument('--delta', type=float, default=0.005, help='prior variance')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')

args = parser.parse_args()

pca = PCA(n_components=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## set seed
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)


print('Getting vocabulary ...')
data_file = os.path.join(args.data_path, 'min_df_{}'.format(args.min_df))
vocab, cluster_valid = data.get_all_data(data_file, temporal=True)
vocab_size = len(vocab)

# get data
print('Getting full data ...')
tokens = cluster_valid['tokens']
counts = cluster_valid['counts']
times = cluster_valid['times']
num_times = len(np.unique(times))
num_docs = len(tokens)

## get embeddings
print('Getting embeddings ...')
emb_path = args.emb_path
vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')
vectors = {}
with open(emb_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if word in vocab:
            vect = np.array(line[1:]).astype(np.float)
            vectors[word] = vect
embeddings = np.zeros((vocab_size, args.emb_size))
words_found = 0
for i, word in enumerate(vocab):
    try:
        embeddings[i] = vectors[word]
        words_found += 1
    except KeyError:
        embeddings[i] = np.random.normal(scale=0.6, size=(args.emb_size, ))
embeddings = torch.from_numpy(embeddings).to(device)
args.embeddings_dim = embeddings.size()

print('\n')
print('=*'*100)
print('Training a Dynamic Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*'*100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path,
        'detm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_L_{}_minDF_{}_trainEmbeddings_{}'.format(
        args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act,
            args.lr, args.batch_size, args.rho_size, args.eta_nlayers, args.min_df, args.train_embeddings))

## define model and optimizer
if args.load_from != '':
    print('Loading checkpoint from {}'.format(args.load_from))
    with open(args.load_from, 'rb') as f:
        model = torch.load(f)
else:
    model = DETM(args, embeddings)
print('\nDETM architecture: {}'.format(model))
model.to(device)

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

def visualize(times):
    """Visualizes topics and embeddings and word usage evolution.
    """
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha)
        print('beta: ', beta.size())
        print('\n')
        print('#'*100)
        print('Visualize topics...')
        topics_words = []
        for k in range(args.num_topics):
            for t in times:
                gamma = beta[k, t, :]
                top_words = list(gamma.detach().numpy().argsort()[-args.num_words+1:][::-1])
                topic_words = [vocab[a] for a in top_words]
                topics_words.append(' '.join(topic_words))
                print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words))

        print('\n')
        # print('Visualize word embeddings ...')
        # queries = ['economic', 'assembly', 'security', 'management', 'debt', 'rights',  'africa']
        # try:
        #     embeddings = model.rho.weight  # Vocab_size x E
        # except:
        #     embeddings = model.rho         # Vocab_size x E
        # neighbors = []
        # for word in queries:
        #     print('word: {} .. neighbors: {}'.format(
        #         word, nearest_neighbors(word, embeddings, vocab, args.num_words)))
        print('#'*100)

        # print('\n')
        # print('Visualize word evolution ...')
        # topic_0 = None ### k
        # queries_0 = ['woman', 'gender', 'man', 'mankind', 'humankind'] ### v

        # topic_1 = None
        # queries_1 = ['africa', 'colonial', 'racist', 'democratic']

        # topic_2 = None
        # queries_2 = ['poverty', 'sustainable', 'trade']

        # topic_3 = None
        # queries_3 = ['soviet', 'convention', 'iran']

        # topic_4 = None # climate
        # queries_4 = ['environment', 'impact', 'threats', 'small', 'global', 'climate']

def _eta_helper(rnn_inp):
    inp = model.q_eta_map(rnn_inp).unsqueeze(1)
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    output = output.squeeze()
    etas = torch.zeros(model.num_times, model.num_topics).to(device)
    inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
    etas[0] = model.mu_q_eta(inp_0)
    for t in range(1, model.num_times):
        inp_t = torch.cat([output[t], etas[t-1]], dim=0)
        etas[t] = model.mu_q_eta(inp_t)
    return etas

def get_eta(rnn_inp):
    model.eval()
    with torch.no_grad():
        return _eta_helper(rnn_inp)

def get_theta(eta, bows):
    model.eval()
    with torch.no_grad():
        inp = torch.cat([bows, eta], dim=1)
        q_theta = model.q_theta(inp)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta

def _diversity_helper(beta, num_tops):
    list_w = np.zeros((args.num_topics, num_tops))
    for k in range(args.num_topics):
        gamma = beta[k, :]
        top_words = gamma.detach().numpy().argsort()[-num_tops:][::-1]
        list_w[k, :] = top_words
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (args.num_topics * num_tops)
    return diversity

def get_cluster_quality():
    """Returns cluster quality.
    """
    topics_distributions = []
    #df = pd.read_csv("/home/beck/Repositories/Data/trec2011_microblog/trec2011_2012_final.tsv", sep="\t")
    cluster_ids = []
    with open(os.path.join(data_file, 'topic_ids.txt'), 'r') as fp:
        for line in fp:
            cluster_ids.append(line.strip())

    # same number of docs as cluster ids
    assert len(cluster_ids) == len(tokens)

    rnn_inp = data.get_rnn_input(tokens, counts, times, num_times, vocab_size, num_docs)
    model.eval()
    with torch.no_grad():
        indices = torch.split(torch.tensor(range(num_docs)), args.eval_batch_size)

        eta = get_eta(rnn_inp)

        for idx, ind in enumerate(indices):
            data_batch, times_batch = data.get_batch(
                tokens, counts, ind, vocab_size, args.emb_size, temporal=True, times=times)
            sums = data_batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch

            eta_td = eta[times_batch.type('torch.LongTensor')]
            theta = get_theta(eta_td, normalized_data_batch)
            topics_distributions += torch.argmax(theta, 1).tolist()

    for (name,fun) in {'NMI': NMI,'Hom': Hom, 'Com': Com,'VM': VM,'Acc': Acc}.items():
        print(name, fun(cluster_ids, topics_distributions))



with open(ckpt, 'rb') as f:
    model = torch.load(f)
model = model.to(device)

print('saving alpha...')
with torch.no_grad():
    alpha = model.mu_q_alpha.detach().numpy()
    scipy.io.savemat(ckpt+'_alpha.mat', {'values': alpha}, do_compression=True)

print('compute clustering metrics')
get_cluster_quality()
# print('computing validation perplexity...')
# val_ppl = get_completion_ppl('val')
# print('computing test perplexity...')
# test_ppl = get_completion_ppl('test')
# print('computing topic coherence and topic diversity...')
# get_topic_quality()
print('visualizing topics and embeddings...')
visualize([0, 5, 10, 15])

