import numpy as np

def word_selector(corpus_sentences, QA_sentences, num_examples, prop_QA):
    # returns ndarray with dimensions num_examples x 4 x len of longest sentence
    # question words followed by answer words without a separator
    # or first half of the sentence followed by second half of the sentence

    # assuming that corpus_sentences is a dictionary with sentence:words pairs and QA_sentences is a list
    # of namedtuples with keys question, answers, labels

    assert 0. <= prop_QA <= 1.

    num_QA_examples = int(num_examples * prop_QA)
    num_corpus_examples = num_examples - num_QA_examples

    QA_indices = np.random.permutation(np.arange(len(QA_sentences)))[:num_QA_examples]

    QA_combined = []
    for QA_i in QA_indices:
        current_question = QA_sentences[QA_i].question
        current_QA_words = []
        for a in QA_sentences[QA_i].answers:
            current_QA_words.append(current_question + a)

        current_QA_words = [qa for _, qa in sorted(zip(QA_sentences[QA_i].labels, current_QA_words), key= lambda x: x[0], reverse=True)]
        QA_combined.append(current_QA_words)

    corpus_combined = []
    corpus_iter = iter(np.random.permutation(np.arange(len(corpus_sentences))))
    while len(corpus_combined) < num_corpus_examples:
    # for corpus_i in corpus_indices:
        corpus_i = next(corpus_iter)
        correct_sentence = corpus_sentences[corpus_i]
        if len(correct_sentence) <= 4:
            continue
        first_half = correct_sentence[:len(correct_sentence)//2]
        current_corpus_words = []
        for _ in range(3):
            iterating = True
            while iterating:
                random_corpus_i = np.random.randint(len(corpus_sentences), size=None)
                iterating = random_corpus_i is corpus_i

            current_corpus_words.append(first_half + corpus_sentences[random_corpus_i][-(len(corpus_sentences[random_corpus_i])//2):])
        corpus_combined.append([correct_sentence] + current_corpus_words)

    all_words = QA_combined + corpus_combined

    return all_words

if __name__ == '__main__':
    from sentence_stream import get_idx, get_QandA_idx

    unique_word_idx, word_vectors, sentence_idx, co_occ_list, word_idx = \
        get_idx(sentences_filename='ARC/visualization/moon_dataset.txt', spacy_language='en_core_web_sm')

    q_and_a_idx = get_QandA_idx(word_idx=word_idx, difficulty='MOON', subset='MOON')

    all_words = word_selector(corpus_sentences=sentence_idx, QA_sentences=q_and_a_idx, num_examples=10, prop_QA=.5)


