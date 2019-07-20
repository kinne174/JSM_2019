import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob

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

    def forward(self, g, inputs, sentence_idx_list):

        g = self.GCN(g, inputs)

        scores = torch.empty((len(sentence_idx_list), 2))
        for sentence_grouping in sentence_idx_list:

            assert len(sentence_grouping) == 4

            # ammend this to handle more than just four sentences, should be able to handle groups of four

            for i in range(4):
                word_embeds = g.nodes[sentence_grouping[i]].data['embeddings']

                sentence_embed = self.LSTM(word_embeds)

                sentence_embed = torch.relu(sentence_embed)

                scores[i, :] = self.scorer(sentence_embed)

        return scores


if __name__ == '__main__':
    from graph_network import build_graph
    from sentence_stream import get_idx, get_QandA_idx
    from word_selection import word_selector

    unique_word_idx, word_vectors, sentence_idx, co_occ_list, word_idx = get_idx(sentences_filename='ARC/visualization/moon_dataset.txt', spacy_language='en_core_web_md')

    q_and_a_idx = get_QandA_idx(word_idx=word_idx, difficulty='MOON', subset='MOON')

    G = build_graph(edge_tuples=co_occ_list, unique_ids=unique_word_idx, embedding_dict=word_vectors)
    G_ndata = G.ndata['embeddings']

    network = GCN_LSTM(embedding_dim=300, hidden_dim1=100, hidden_dim2=15, answer_dim=2)
    loss_function = nn.NLLLoss()

    params = network.parameters()
    optimizer = torch.optim.Adam(params, lr=0.01)

    num_sentences_per_epoch = 10
    prop_QA = .3

    # use glob to see how many files in directory and then create a new one with this length to keep a log

    all_logits = []
    for epoch in range(30):
        # network.zero_grad()

        # insert word selection here, labels will be the same for each 4-tuple (correct, wrong, wrong, wrong)
        training_sentences = word_selector(corpus_sentences=sentence_idx, QA_sentences=q_and_a_idx,
                                           num_examples=num_sentences_per_epoch, prop_QA=prop_QA)
        training_labels = [[1, 0, 0, 0] for _ in range(num_sentences_per_epoch)]
        training_labels = [l for sublist in training_labels for l in sublist]

        # word_idx_list = [[2*i + j for j in range(2)] for i in range(4)]
        # labels = torch.tensor([0, 0, 0, 1])

        logits = network(g=G, inputs=G_ndata, sentence_idx_list=training_sentences)
        # we save the logits for visualization later
        # all_logits.append(logits.detach())
        logp = F.log_softmax(logits, 1)
        # we only compute loss for labeled nodes
        loss = loss_function(logp, training_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))












