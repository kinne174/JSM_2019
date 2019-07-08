import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import numpy as np

from graph_network import build_graph
from sentence_stream import get_idx

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

def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
    return {'embeddings': torch.sum(nodes.mailbox['msg'], dim=1)}

class GCNLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        _ = g.ndata.pop('embeddings')
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
    def __init__(self, sentence_dim, answer_dim):
        super(sentence_classifier, self).__init__()
        self.linear = nn.Linear(sentence_dim, answer_dim)

    def forward(self, sentence):
        # either need to do preprocessing here or before to put spacing between the sentences, later
        h = self.linear(sentence)
        return h

class GCN_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim1, hidden_dim2, answer_dim):
        super(GCN_LSTM, self).__init__()

        self.GCN = GCNLayer(embedding_dim=embedding_dim, hidden_dim=hidden_dim1)
        self.LSTM = LSTM_sentences(embedding_dim=hidden_dim1, hidden_dim=hidden_dim2)
        self.scorer = sentence_classifier(sentence_dim=hidden_dim2, answer_dim=answer_dim)

    def forward(self, g, inputs, word_idx_list):

        g = self.GCN(g, inputs)

        assert len(word_idx_list) == 4
        #
        scores = torch.empty((4, 2))
        for i in range(len(word_idx_list)):
            word_embeds = g.nodes[word_idx_list[i]].data['embeddings']

            sentence_embed = self.LSTM(word_embeds)

            sentence_embed = torch.relu(sentence_embed)

            scores[i, :] = self.scorer(sentence_embed)

        return scores


if __name__ == '__main__':
    unique_word_idx, word_vectors, sentence_idx, co_occ_list = get_idx(spacy_language='en_core_web_md')

    G = build_graph(edge_tuples=co_occ_list, unique_ids=unique_word_idx, embedding_dict=word_vectors)
    G_ndata = G.ndata['embeddings']

    network = GCN_LSTM(embedding_dim=300, hidden_dim1=100, hidden_dim2=15, answer_dim=2)
    loss_function = nn.NLLLoss()

    params = network.parameters()
    optimizer = torch.optim.Adam(params, lr=0.01)

    all_logits = []
    for epoch in range(30):
        # network.zero_grad()

        word_idx_list = [[2*i + j for j in range(2)] for i in range(4)]
        labels = torch.tensor([0, 0, 0, 1])

        logits = network(g=G, inputs=G_ndata, word_idx_list=word_idx_list)
        # we save the logits for visualization later
        # all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = loss_function(logp, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))












