import spacy
import numpy as np
import json
import os, getpass
import codecs
from collections import Counter, namedtuple
from itertools import combinations, product

from q_and_a_generator import get_qa_info

'''
based on a .txt file of sentences separated by '\n' run through them and save them to a list of spacy Docs. Then can get
a count of lemmitizations in each sentence and how those counts relate to interword counts to get how to define edges
in the graph network
'''

# this function takes in a corpus of sentences and converts each of them to a list of numbers that represent the words
# then based on the co occurance returns which words are paired, this is how edges are defined in the graph network
def get_idx(sentences_filename='ARC/visualization/test_dataset.txt', spacy_language='en_core_web_sm', threshold=0.5):
    # load interpreter
    nlp = spacy.load(spacy_language, disable=['ner', 'parser'])

    # load in corpus
    if getpass.getuser() == 'Mitch':
        os.chdir('C:/Users/Mitch/PycharmProjects')
    else:
        os.chdir('/home/kinne174/private/PythonProjects')

    assert 0. <= threshold <= 1.

    # total lines used to exit at the right time
    num_lines = sum([1 for _ in codecs.open(sentences_filename, 'r', encoding='utf-8', errors='ignore')])

    def doc_to_spans(list_of_texts, join_string=' ||| '):
        # convert list of text to lemmas using work around
        # https://towardsdatascience.com/a-couple-tricks-for-using-spacy-at-scale-54affd8326cf
        num_iterations = int(np.ceil(len(list_of_texts) / 1000))
        new_docs = []
        for ii in range(num_iterations):
            temp_list_of_texts = list_of_texts[ii * 1000:(ii + 1) * 1000]
            stripped_string = join_string.strip()
            all_docs = nlp(join_string.join(temp_list_of_texts))
            split_inds = [i for i, token in enumerate(all_docs) if token.text == stripped_string] + [len(all_docs)]
            new_docs.extend([all_docs[(i + 1 if i > 0 else i):j] for i, j in zip([0] + split_inds[:-1], split_inds)])
        return new_docs

    def filter_docs(original_docs):
        new_docs = []
        # keep tokens that are not stop words, punctuation, docs that contain less than some threshold on punctuation/ numbers
        for doc in original_docs:
            if len(doc) < 5:  # less than 5 tokens in the doc skip it
                continue
            values = [(token.is_punct, token.is_digit, token.pos_ == 'SYM', not token.has_vector, token.like_url) for
                      token in doc]
            sums = np.sum(values, axis=0)
            if sums[0] + sums[1] + sums[2] >= .5 * len(doc):  # more than half of doc is punctuation, digits and symbols
                continue
            if sums[3] + sums[4] > 0:  # at least one of the tokens does not have a vector or is a url
                continue
            new_docs.append([token.lemma_ for token in doc if not (token.is_stop or token.is_punct)])

        return new_docs

    def sentence_and_word_idx(all_docs, word_idx, sentence_idx):
        # create a dictionary with index: lemma / lemma: index and sentences: [indices]
        # this is called multiple times until all sentences are accounted for
        flattened_docs = [t for doc in all_docs for t in set(doc)]

        current_word_ind = len(word_idx)

        for lemma in flattened_docs:
            if lemma not in word_idx:
                word_idx[current_word_ind] = lemma
                word_idx[lemma] = current_word_ind
                current_word_ind += 1

        current_sentence_ind = len(sentence_idx)
        for sent in all_docs:
            sentence_idx[current_sentence_ind] = [word_idx[l] for l in sent]
            current_sentence_ind += 1

        return word_idx, sentence_idx

    def co_occurance(all_docs):
        # based on a list of lists of tokens get the co occurance values for tokens
        flattened_docs = [t for doc in all_docs for t in set(doc)]
        all_lemma_counter = Counter(flattened_docs)
        joined_lemmas = [x for doc in all_docs for x in combinations(set(doc), 2)]
        joined_lemma_counter = Counter(joined_lemmas)
        co_occ = {
            k: np.log((joined_lemma_counter[k] * len(all_docs)) / (all_lemma_counter[k[0]] * all_lemma_counter[k[1]])) /
               (-1 * np.log(joined_lemma_counter[k] / len(all_docs)))
            for k in joined_lemma_counter.keys()}
        return co_occ

    with codecs.open(sentences_filename, 'r', encoding='utf-8', errors='ignore') as corpus:
        text_list = []
        word_idx = {}
        sentence_idx = {}

        for i, line in enumerate(corpus):
            text_list.append(line.strip())

            # spacy can only handle so many characters so only take the text 1000 lines at a time to be safe
            if (i % 1000 == 0 and i is not 0) or (len(text_list) == num_lines - i//1000):

                docs = doc_to_spans(text_list)

                filtered_docs = filter_docs(docs)

                word_idx, sentence_idx = sentence_and_word_idx(filtered_docs, word_idx, sentence_idx)

                text_list = []

    co_occ_dict = co_occurance(list(sentence_idx.values()))

    co_occ_list = [key for key, val in co_occ_dict.items() if val >= threshold]

    # based on words selected by co occurance retrieve only the ones to keep
    unique_word_idx = list(set([idx for t in co_occ_list for idx in t]))
    unique_word_vectors = {idx: nlp.vocab.get_vector(word_idx[idx]) for idx in unique_word_idx}

    # update sentence and word dicts removing words that did not make the cut
    keep_word_idx = [False] * len(word_idx)
    updated_word_idx = {}
    for ii, uwi in enumerate(unique_word_idx):
        updated_word_idx[ii] = word_idx[uwi]
        updated_word_idx[word_idx[uwi]] = ii
        keep_word_idx[uwi] = True

    updated_sentence_idx = {}
    for id, sentence_inds in sentence_idx.items():
        updated_sentence_idx[id] = [unique_word_idx.index(ind) for ind in sentence_inds if keep_word_idx[ind]]

    return unique_word_idx, unique_word_vectors, updated_sentence_idx, co_occ_list, updated_word_idx


# this is run after the corpus get_idx so that no new words are added, this function uses the word to index dict that
# was created above
def get_QandA_idx(word_idx, difficulty, subset, spacy_language='en_core_web_sm'):
    # load intepreter
    nlp = spacy.load(spacy_language, disable=['ner', 'parser'])

    if not isinstance(difficulty, list) and difficulty is not 'all':
        difficulty = [difficulty]
    if not isinstance(subset, list) and subset is not 'all':
        subset = [subset]

    # depending on combination extract appropriate question and answer sentences
    if difficulty is 'all' and subset is 'all':
        subsets = ['TRAIN', 'TEST', 'VALIDATION']
        difficulties = ['EASY', 'CHALLENGE']

        combinations = list(product(subsets, difficulties))

        qa_named_tuples = []

        for c_tup in combinations:
            qa_named_tuples.extend(get_qa_info(c_tup[1], c_tup[0]))

    elif subset is 'all':
        subsets = ['TRAIN', 'TEST', 'VALIDATION']
        if difficulty in ['EASY', 'CHALLENGE']:
            combinations = list(product(subsets, difficulty))
            qa_named_tuples = []

            for c_tup in combinations:
                qa_named_tuples.extend(get_qa_info(c_tup[1], c_tup[0]))
        else:
            raise Exception('difficulty must be EASY or HARD')

    elif difficulty is 'all':
        difficulties = ['EASY', 'CHALLENGE']
        if subset in ['TRAIN', 'TEST', 'VALIDATION']:
            combinations = list(product(subset, difficulties))
            qa_named_tuples = []

            for c_tup in combinations:
                qa_named_tuples.extend(get_qa_info(c_tup[1], c_tup[0]))
        else:
            raise Exception('subset must be TRAIN TEST or VALIDATION')

    elif all([s in ['TRAIN', 'TEST', 'VALIDATION'] for s in subset]) and all([d in ['EASY', 'CHALLENGE'] for d in difficulty]):
        combinations = list(product(subset, difficulty))
        qa_named_tuples = []

        for c_tup in combinations:
            qa_named_tuples.extend(get_qa_info(c_tup[1], c_tup[0]))

    elif all([s in ['MOON'] for s in subset]) and all([d in ['MOON'] for d in difficulty]):
        combinations = list(product(subset, difficulty))
        qa_named_tuples = []

        for c_tup in combinations:
            qa_named_tuples.extend(get_qa_info(c_tup[1], c_tup[0], special='MOON'))

    else:
        raise Exception('difficulty and or subset is wrong!')

    def doc_to_spans(list_of_texts, join_string=' ||| '):
        # https://towardsdatascience.com/a-couple-tricks-for-using-spacy-at-scale-54affd8326cf
        num_iterations = int(np.ceil(len(list_of_texts) / 1000))
        new_docs = []
        for ii in range(num_iterations):
            temp_list_of_texts = list_of_texts[ii * 1000:(ii + 1) * 1000]
            stripped_string = join_string.strip()
            all_docs = nlp(join_string.join(temp_list_of_texts))
            split_inds = [i for i, token in enumerate(all_docs) if token.text == stripped_string] + [len(all_docs)]
            new_docs.extend([all_docs[(i + 1 if i > 0 else i):j] for i, j in zip([0] + split_inds[:-1], split_inds)])
        return new_docs

    # similar to above, stream through sentences converting them to tokens and save them using corpus unique words
    # retrieved
    all_text = []
    labels = []
    for nt in qa_named_tuples:
        all_text.extend([nt.question] + nt.choices_text)
        correct_choice = [0]*len(nt.choices_text)
        correct_choice[nt.choices_labels.index(nt.answer)] = 1
        labels.append(correct_choice)

    if len(all_text) <= 1000:
        docs = doc_to_spans(all_text)
    else:
        num_iterations = (len(all_text)//1000) + 1
        docs = []
        for ii in range(num_iterations):
            if ii == num_iterations-1:
                docs.extend(doc_to_spans(all_text[(ii*1000):]))
            else:
                docs.extend(doc_to_spans(all_text[(ii*1000):((ii+1)*1000)]))

    Q_and_A_Document = namedtuple('Q_and_A_Document', 'question answers labels')
    Q_and_A_docs = []

    # separate the questions and answers since above they were all smashed together
    question_now = True
    answers = []
    ii = 0
    jj = 0
    for doc in docs:
        if question_now:
            question = [word_idx[token.lemma_] for token in doc if token.lemma_ in word_idx]
            question_now = False
        else:
            answers.append([word_idx[token.lemma_] for token in doc if token.lemma_ in word_idx])
            jj += 1

            if jj == len(labels[ii]):
                jj = 0
                question_now = True

                Q_and_A_docs.append(Q_and_A_Document(question=question, answers=answers, labels=labels[ii]))
                answers = []
                ii += 1

    return Q_and_A_docs

if __name__ == '__main__':
   uwi, uwv, si, col, wids = get_idx(sentences_filename='ARC/visualization/moon_dataset.txt', spacy_language='en_core_web_sm', threshold=0.9)

   q_and_a_idx = get_QandA_idx(word_idx=wids, difficulty='all', subset='all')












