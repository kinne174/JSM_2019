import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob, os
from datetime import date, datetime
from contextlib import contextmanager
from timeit import default_timer

torch.manual_seed(1)

'''
Based on graph built in graph_network.py and holding all edges and nodes within it constant, train a network to share
within the graph network and an LSTM to transform the words to a sentence vector (last hidden state). The top classifier
can be negative sampling or negative log likelihood or entropy...
'''

def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
    return {'msg': edges.src['embeddings']}

# TODO is there a better way of consolidating the messages

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
        g.ndata['embeddings'] = inputs
        # trigger message passing on all edges
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('embeddings')
        # perform linear transformation
        # return self.linear(h)
        g.ndata['embeddings'] = self.linear(h)

        return g, h

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, out_feats)
        # self.gcn2 = GCNLayer(hidden_size, out_feats)
        # TODO figure out why can't do multiple levels for graph network, should be in 1_first.py

    def forward(self, g, inputs):
        g, h = self.gcn1(g, inputs)
        # h = torch.relu(h)
        # g, _ = self.gcn2(g, h)
        return g

class LSTM_sentences(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTM_sentences, self).__init__()
        # self.hidden_dim = hidden_dim  # dimension of eventual sentence embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, embeds):
        outputs, (hn, cn) = self.lstm(embeds.view(embeds.shape[0], 1, -1))  # see what embeds actually are, to get how many there are
        return hn[-1]

class sentence_classifier(nn.Module):
    def __init__(self, sentence_dim, hidden_dim, answer_dim):
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
        super(GCN_LSTM, self).__init__()

        self.GCN = GCN(in_feats=embedding_dim, hidden_size=gn_hidden_dim1, out_feats=gn_hidden_dim2)
        self.LSTM = LSTM_sentences(embedding_dim=gn_hidden_dim2, hidden_dim=lstm_hidden_dim)
        self.scorer = sentence_classifier(sentence_dim=lstm_hidden_dim, hidden_dim=answer_hidden_dim, answer_dim=answer_dim)

    def forward(self, g, inputs, sentence_idx_list):

        g = self.GCN(g, inputs)

        scores = torch.empty((len(sentence_idx_list), 4))
        for ii, sentence_grouping in enumerate(sentence_idx_list):

            assert len(sentence_grouping) == 4

            # ammend this to handle more than just four sentences, should be able to handle groups of four

            for jj in range(4):
                word_embeds = g.nodes[sentence_grouping[jj]].data['embeddings']

                sentence_embed = self.LSTM(word_embeds)

                sentence_embed = torch.relu(sentence_embed)

                scores[ii, jj] = self.scorer(sentence_embed)

        return scores

# TODO create evaluation tool on all questions and answers

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


if __name__ == '__main__':
    from graph_network import build_graph
    from sentence_stream import get_idx, get_QandA_idx
    from word_selection import word_selector

    os.chdir('C:/Users/Mitch/PycharmProjects')

    # use glob to see how many files in directory and then create a new one with this length to keep a log
    all_log_filenames = glob.glob('JSM_2019/logs/{}_*'.format(date.today()))
    num_files = len(all_log_filenames)

    log_filename = 'JSM_2019/logs/{}_attempt{}.txt'.format(date.today(), num_files)
    with open(log_filename, 'w') as f:
        f.write('START: {}\n\n'.format(datetime.now()))
        f.write('Creating sentence ids and graph network...')

    with elapsed_timer() as elapsed:
        unique_word_idx, word_vectors, sentence_idx, co_occ_list, word_idx = get_idx(sentences_filename='ARC/visualization/moon_dataset.txt', spacy_language='en_core_web_md')

        q_and_a_idx = get_QandA_idx(word_idx=word_idx, difficulty='MOON', subset='MOON')

        G = build_graph(edge_tuples=co_occ_list, unique_ids=unique_word_idx, embedding_dict=word_vectors)
        G_ndata = G.ndata.pop('embeddings')

        duration = '%.2f' % elapsed()

    with open(log_filename, 'a') as f:
        f.write('Done! Time elapsed: {}\n\n'.format(duration))

    network = GCN_LSTM(embedding_dim=300, gn_hidden_dim1=500, gn_hidden_dim2=400, lstm_hidden_dim=300, answer_hidden_dim=200, answer_dim=1)
    loss_function = nn.NLLLoss()

    params = network.parameters()
    optimizer = torch.optim.Adam(params, lr=0.01)

    num_sentences_per_epoch = 100
    prop_QA = 0.3

    all_logits = []
    for epoch in range(1000):
        # network.zero_grad()

        # insert word selection here, labels will be the same for each 4-tuple (correct, wrong, wrong, wrong)
        training_sentences = word_selector(corpus_sentences=sentence_idx, QA_sentences=q_and_a_idx,
                                           num_examples=num_sentences_per_epoch, prop_QA=prop_QA)
        # training_labels = [[1, 0, 0, 0] for _ in range(num_sentences_per_epoch)]
        # training_labels = torch.tensor([l for sublist in training_labels for l in sublist]).view(size=(num_sentences_per_epoch, 4))
        training_labels = torch.tensor([0]*num_sentences_per_epoch)

        with elapsed_timer() as elapsed:
            logits = network(g=G, inputs=G_ndata, sentence_idx_list=training_sentences)
            duration = '%.2f' % elapsed()

        all_logits.append(logits.detach())
        logp = F.log_softmax(input=logits, dim=1)
        loss = loss_function(logp, training_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with open(log_filename, 'a') as f:
            f.write('Epoch %d | Loss: %.4f\n' % (epoch, loss.item()))
            f.write('Elapsed time: {}\n\n'.format(duration))

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
        print('Elapsed time: {}'.format(duration))

        if epoch % 10 == 0 and epoch is not 0:
            continue
            # TODO figure out a way to connect this with q_and_a_idx to get the question/answers/labels
            with torch.no_grad():
                test_questions = word_selector(corpus_sentences=sentence_idx, QA_sentences=q_and_a_idx,
                                           num_examples=5, prop_QA=1.)
                tag_scores = network(g=G, inputs=G_ndata, sentence_idx_list=test_questions)
                with open(log_filename, 'a') as f:
                    for questions in test_questions:
                        for question in questions:
                            question_text = ' '.join([word_idx[ind] for ind in question])







# TODO add a breaking character in between questions and answers
# TODO figure out a way to put this on the UMN computers






