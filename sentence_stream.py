# nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])
# doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)
# for token in doc:
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov)

# for token1 in doc:
#     for token2 in doc:
#         print(token1.text, token2.text, token1.similarity(token2))

# nlp = spacy.load("en_core_web_sm")
# doc = nlp(u"Hello, world. Here are two sentences.")
# print([t.text for t in doc])
#
# from spacy.tokens import Span
#
# doc = nlp(u"FB is hiring a new VP of global policy")
# doc.ents = [Span(doc, 0, 1, label=doc.vocab.strings[u"ORG"])]
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)

# nlp = spacy.load("en_core_web_md")
# doc = nlp(u"Apple and banana are similar. Pasta and hippo aren't.")
#
# apple = doc[0]
# banana = doc[2]
# pasta = doc[6]
# hippo = doc[8]
#
# print("apple <-> banana", apple.similarity(banana))
# print("pasta <-> hippo", pasta.similarity(hippo))
# print(apple.has_vector, banana.has_vector, pasta.has_vector, hippo.has_vector)

# start = time.time()
# for _ in range(1000):
#     doc = nlp(u"Peach emoji is where it has always been. Peach is the superior "
#               u"emoji. It's outranking eggplant 🍑 ")
# print(time.time() - start)
# # sentences = list(doc.sents)
# # first_sentence = sentences[0]
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.is_stop)
# print(doc[0].text)          # 'Peach'
# print(doc[1].text)          # 'emoji'
# print(doc[-1].text)         # '🍑'
# print(doc[17:19].text)      # 'outranking eggplant'
#
# noun_chunks = list(doc.noun_chunks)
# print(noun_chunks[0].text)  # 'Peach emoji'
#
# sentences = list(doc.sents)
# assert len(sentences) == 3
# print(sentences[1].text)

import spacy
import numpy as np
import json
import os
import codecs
from collections import Counter
from itertools import combinations

'''
based on a .txt file of sentences separated by '\n' run through them and save them to a list of spacy Docs. Then can get
a count of lemmitizations in each sentence and how those counts relate to interword counts to get how to define edges
in the graph network, also need a way to define the vector-nodes in the graph network
'''

nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])

def doc_to_spans(list_of_texts, join_string=' ||| '):
    # https://towardsdatascience.com/a-couple-tricks-for-using-spacy-at-scale-54affd8326cf
    stripped_string = join_string.strip()
    all_docs = nlp(join_string.join(list_of_texts))
    split_inds = [i for i, token in enumerate(all_docs) if token.text == stripped_string] + [len(all_docs)]
    new_docs = [all_docs[(i + 1 if i > 0 else i):j] for i, j in zip([0] + split_inds[:-1], split_inds)]
    return new_docs

# load in corpus
os.chdir('C:/Users/Mitch/PycharmProjects')
moon_dataset_filename = 'ARC/visualization/test_dataset.txt'

with codecs.open(moon_dataset_filename, 'r', encoding='utf-8', errors='ignore') as corpus:
    text_list = corpus.read().splitlines()
    # for line in corpus:
    #     text_list.append(line)

docs = doc_to_spans(text_list)

def filter_docs(original_docs):
    new_docs = []
    # keep tokens that are not stop words, punctuation, docs that contain less than some threshold on punctuation/ numbers
    for doc in original_docs:
        if len(doc) < 5:  # less than 5 tokens in the doc skip it
            continue
        values = [(token.is_punct, token.is_digit, token.pos_ == 'SYM', not token.has_vector, token.like_url) for token in doc]
        sums = np.sum(values, axis=0)
        if sums[0] + sums[1] + sums[2] >= .5*len(doc):  # more than half of doc is punctuation, digits and symbols
            continue
        if sums[3] + sums[4] > 0:  # at least one of the tokens does not have a vector or is a url
            continue
        new_docs.append([token.lemma_ for token in doc if not (token.is_stop or token.is_punct)])

    return new_docs

filtered_docs = filter_docs(docs)

def sentence_and_word_idx(all_docs):
    # create a dictionary with index: lemma pairs and sentences: [indices] pairs
    flattened_docs = [t for doc in all_docs for t in set(doc)]
    word_idx = {}
    for i, lemma in enumerate(flattened_docs):
        word_idx[i] = lemma
        word_idx[lemma] = i

    sentence_idx = {}
    for j, sent in enumerate(all_docs):
        sentence_idx[j] = [word_idx[l] for l in sent]

    return sentence_idx, word_idx

sentence_idx, word_idx = sentence_and_word_idx(filtered_docs)

def co_occurance(all_docs):
    # based on a list of lists of tokens get the co occurance values for tokens
    flattened_docs = [t for doc in all_docs for t in set(doc)]
    all_lemma_counter = Counter(flattened_docs)
    joined_lemmas = [x for doc in all_docs for x in combinations(set(doc), 2)]
    joined_lemma_counter = Counter(joined_lemmas)
    co_occ = {k: np.log((joined_lemma_counter[k] * len(all_docs))/(all_lemma_counter[k[0]] * all_lemma_counter[k[1]]))/
                 (-1*np.log(joined_lemma_counter[k]/len(all_docs)))
                    for k in joined_lemma_counter.keys()}
    return co_occ

co_occ_dict = co_occurance(list(sentence_idx.values()))

threshold = 0.5
co_occ_list = [key for key, val in co_occ_dict.values() if val >= threshold]

unique_word_idx = set([idx for t in co_occ_list for idx in t])
unique_word_vectors = {idx: nlp.vocab.get_vector(word_idx[idx]) for idx in unique_word_idx}












