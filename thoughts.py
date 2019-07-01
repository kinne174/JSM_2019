'''
A place to think about what I need to do to prepare for JSM 2019, countdown: 44 days

Preproccessing stage:
-Be able to take in a sentence cleanly
    -different endings/beginnings should be discarded to only focus on the core word
    -punctuation should be discarded
    -make all words lowercase
    -keep sentence order in tact
    -maybe create a dictionary with index: word pairs and another dictionary with sentence index: word index pairs
        so I can build sentences without keeping track of all the words all the time
    -possibly build a nonsense catcher so that if there are mis spelled words or a lot of characters I can filter them
    -only focus on the sentences in the corpus for right now, question and answers can come later
    -filter out stop words

First stage:
-The first stage is to get the graph neural network and populate it with word embeddings
    -The word embeddings can be random or pre trained w2v or glove
    -Only use the top K words depending on the number of unique words
-Also want to base edges off of co occurance of words in a sentence, still not sure what these edges should represent
-Do DGL-pytorch tutorials to get understanding of how to update them

Second Stage:
-Build simple LSTM model to act like a sentence completer attached to a classifier that says whether the completion is
    valid or not, can look up papers on techniques of how people train these
    -Input will be ordered words of the first part of a sentence, break token, then the second part of the sentence
-Also with this make sure that the

Looks like in the message passing I can train a linear function of a transformation using something like in 1_first.py,
then the second part is to train the LSTM using the correctly popped nodes, see if the backwards propagation works
all the way through to training the weight matrices, can also do like that one paper (graphSage) with multiple
aggregation functions and weight matrices

create a file that streams through the words and cleans the sentences removing endings and punctuation and stop words,
save a dictionary object with index and word, also sentences with sentence indicators and word indexes as a lst in order,
on the way can create objects that store word counts and how often words appear together (this can also be a post processing
thing in order to get all the memory correct on the first try) can start small again with the moon dataset

next thing to do is to use the vectors and the edge tuples to create a graph network using dgl library
'''