import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob, os
from datetime import date, datetime
from contextlib import contextmanager
from timeit import default_timer
import numpy as np
import getpass
import argparse

from graph_network import build_graph
from sentence_stream import get_idx, get_QandA_idx
from word_selection import word_selector, test_word_selector

torch.manual_seed(1)

'''
Based on graph built in graph_network.py and holding all edges and nodes within it constant, train a network to share
within the graph network and an LSTM to transform the words to a sentence vector (last hidden state). The top classifier
can be negative sampling or negative log likelihood or entropy...
'''

def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg': edges.src['h']}

# TODO is there a better way of consolidating the messages, part of training

def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

class GCNLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all edges
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        # g.ndata['h'] = self.linear(h)
        return self.linear(h)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        # can toggle this to use one or two message passing
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, out_feats)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.tanh(h)
        h = self.gcn2(g, h)
        g.ndata['h'] = h
        return g

class LSTM_sentences(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTM_sentences, self).__init__()
        # convert word embeddings to a sentence embedding
        # self.hidden_dim = hidden_dim  # dimension of eventual sentence embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, embeds):
        outputs, (hn, cn) = self.lstm(embeds.view(embeds.shape[0], 1, -1))  # see what embeds actually are, to get how many there are
        return hn[-1]

class sentence_classifier(nn.Module):
    def __init__(self, sentence_dim, hidden_dim, answer_dim):
        # classify the sentences into a score, higher the better, simple MLP
        super(sentence_classifier, self).__init__()
        self.linear1 = nn.Linear(sentence_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, answer_dim)

    def forward(self, sentence):
        h = self.linear1(sentence)
        h = torch.relu(h)
        h = self.linear2(h)
        return h

class GCN_LSTM(nn.Module):
    def __init__(self, embedding_dim, gn_hidden_dim1, gn_hidden_dim2, lstm_hidden_dim, answer_hidden_dim, answer_dim):
        # combining all parts
        super(GCN_LSTM, self).__init__()

        self.GCN = GCN(in_feats=embedding_dim, hidden_size=gn_hidden_dim1, out_feats=gn_hidden_dim2)
        self.LSTM = LSTM_sentences(embedding_dim=gn_hidden_dim2, hidden_dim=lstm_hidden_dim)
        self.scorer = sentence_classifier(sentence_dim=lstm_hidden_dim, hidden_dim=answer_hidden_dim, answer_dim=answer_dim)

    def forward(self, g, inputs, sentence_idx_list):

        # message passing
        g = self.GCN(g, inputs)

        scores = torch.empty((len(sentence_idx_list), 4))
        for ii, sentence_grouping in enumerate(sentence_idx_list):

            assert len(sentence_grouping) == 4

            # for each selection of four options...
            for jj in range(4):
                # extract word embeddings, create sentence embedding and score
                word_embeds = g.nodes[sentence_grouping[jj]].data['h']

                sentence_embed = self.LSTM(word_embeds)

                sentence_embed = torch.relu(sentence_embed)

                scores[ii, jj] = self.scorer(sentence_embed)

        return scores

# used to time runs
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main file of training for question answering JSM 2019')

    parser.add_argument('-corpus_fn', type=str, help='The filename where the corpus is located', default='ARC/visualization/moon_dataset.txt')
    parser.add_argument('-threshold', type=float,
                        help='The co-occurance threshold, higher means less words, between -1 and 1, currently using 0.5 most of the time',
                        default=0.5)
    parser.add_argument('-QA_difficulty', type=str, help='the difficulty of the questions to be extracted, either EASY or CHALLENGE or all',
                        default='MOON')
    parser.add_argument('-QA_subset', type=str, help='the subset of the questions to be extracted, either TRAIN or VALIDATION or TEST or all',
                        default='MOON')
    parser.add_argument('-gn_hidden_dim1', type=int, help='dimension of first hidden layer in graph network',
                        default=300)
    parser.add_argument('-gn_hidden_dim2', type=int, help='dimension of the second hidden layer in graph network',
                        default=200)
    parser.add_argument('-lstm_hidden_dim', type=int, help='dimension of hidden layer in LSTM', default=200)
    parser.add_argument('-answer_hidden_dim', type=int, help='dimension of hidden layer in answer classifier',
                        default=100)
    parser.add_argument('-num_epochs', type=int, help='number of epochs to run', default=501)
    parser.add_argument('-learning_rate', type=float, help='learning rate of the parameters', default=0.01)
    parser.add_argument('-QA_proportion', type=float,
                        help='the proportion of question answers to be used in training with sentences from the corpus',
                        default=0.3)
    parser.add_argument('-num_sentences', type=int, help='number of sentences to be used in training for each epoch',
                        default=100)
    parser.add_argument('-evaluate_every', type=int, help='print out guesses to questions every this many epochs',
                        default=50)

    # dict of arguments
    args = vars(parser.parse_args())

    # to run on my laptop and university server
    if getpass.getuser() == 'Mitch':
        os.chdir('C:/Users/Mitch/PycharmProjects')
    else:
        os.chdir('/home/kinne174/private/PythonProjects/')

    # set seed
    np.random.seed(0)

    # use glob to see how many files in directory and then create a new one with this length to keep a log
    all_log_filenames = glob.glob('JSM_2019/logs/{}_*'.format(date.today()))
    num_files = len(all_log_filenames)

    log_filename = 'JSM_2019/logs/{}_attempt{}.txt'.format(date.today(), num_files)
    with open(log_filename, 'w') as f:
        f.write('START: {}\n\n'.format(datetime.now()))
        f.write('Creating sentence ids and graph network...')

    # extract sentence information from corpus and from question and answers
    # build the graph with appropriate edges and node embeddings
    with elapsed_timer() as elapsed:
        unique_word_idx, word_vectors, sentence_idx, co_occ_list, word_idx = \
            get_idx(sentences_filename=args['corpus_fn'], spacy_language='en_core_web_md', threshold=args['threshold'])

        q_and_a_idx = get_QandA_idx(word_idx=word_idx, difficulty=args['QA_difficulty'], subset=args['QA_subset'])

        G = build_graph(edge_tuples=co_occ_list, unique_ids=unique_word_idx, embedding_dict=word_vectors)
        G_ndata = G.ndata.pop('embeddings')

        duration = '%.2f' % elapsed()

    with open(log_filename, 'a') as f:
        f.write('Done! Time elapsed: {}\n\n'.format(duration))

    network = GCN_LSTM(embedding_dim=300, gn_hidden_dim1=args['gn_hidden_dim1'], gn_hidden_dim2=args['gn_hidden_dim2'],
                       lstm_hidden_dim=args['lstm_hidden_dim'], answer_hidden_dim=args['answer_hidden_dim'], answer_dim=1)
    loss_function = nn.NLLLoss()

    params = network.parameters()
    optimizer = torch.optim.Adam(params, lr=args['learning_rate'])

    # number of sentences to use in training
    num_sentences_per_epoch = args['num_sentences']
    prop_QA = args['QA_proportion']

    # when evaluating on true questions and answers, how many of the training set to include and which ones
    # should be left out for testing
    test_indices = list(range(6))
    indices_to_select_from = np.array([ind for ind in range(len(q_and_a_idx)) if ind not in test_indices])
    training_indices = list(np.random.permutation(indices_to_select_from)[:15])

    for epoch in range(args['num_epochs']):
        # network.zero_grad()

        # extract which sentences to train on, part from corpus part from question answers
        # insert word selection here, labels will be the same for each 4-tuple (correct, wrong, wrong, wrong)
        training_sentences = word_selector(corpus_sentences=sentence_idx, QA_sentences=q_and_a_idx,
                                           num_examples=num_sentences_per_epoch, prop_QA=prop_QA,
                                           leave_out_QA_indices=test_indices)
        # training_labels = [[1, 0, 0, 0] for _ in range(num_sentences_per_epoch)]
        # training_labels = torch.tensor([l for sublist in training_labels for l in sublist]).view(size=(num_sentences_per_epoch, 4))
        training_labels = torch.tensor([0]*num_sentences_per_epoch)

        with elapsed_timer() as elapsed:
            # extract logits and softmax them, then back propagate
            logits = network(g=G, inputs=G_ndata, sentence_idx_list=training_sentences)

            logp = F.log_softmax(input=logits, dim=1)
            loss = loss_function(logp, training_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            duration = '%.2f' % elapsed()

        with open(log_filename, 'a') as f:
            f.write('Epoch %d | Loss: %.4f\n' % (epoch, loss.item()))
            f.write('Elapsed time: {}\n\n'.format(duration))

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        print('Elapsed time: {}'.format(duration))

        # intermediate validation testing to just get an idea of how the model is doing
        if epoch is not 0 and epoch % args['evaluate_every'] == 0:
            all_indices = test_indices + training_indices
            test_questions = test_word_selector(QA_sentences=q_and_a_idx, which_examples=all_indices)
            with torch.no_grad():
                test_logits = network(g=G, inputs=G_ndata, sentence_idx_list=test_questions)
                test_logp = F.log_softmax(input=test_logits, dim=1)
                test_logp = test_logp.numpy()

            true_question_answer_words = [q_and_a_idx[ti] for ti in all_indices]
            with open(log_filename, 'a') as f:
                for qa_ind, qa_pair in enumerate(true_question_answer_words):
                    f.write('{}. {}\n'.format(qa_ind+1, ' '.join([word_idx[ind] for ind in qa_pair.question])))
                    for ans_ind, ans in enumerate(qa_pair.answers):
                        correct_answer = '*' if qa_pair.labels[ans_ind] == 1 else ' '
                        selected_answer = '#' if test_logp[qa_ind, ans_ind] == max(test_logp[qa_ind, :]) else ' '
                        f.write('{}{} {}\n'.format(correct_answer, selected_answer, ' '.join([word_idx[ind] for ind in ans])))
                f.write('\n')

    # TODO at the end evaluate on VALIDATION

# TODO add a breaking character in between questions and answers, breaking character/embedding for LSTM only not clear why it is needed
# TODO figure out a way to put this on the UMN computers, update all files again before running






