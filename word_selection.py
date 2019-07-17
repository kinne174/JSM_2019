import numpy as np

def word_selector(corpus_sentences, QA_sentences, num_examples, prop_QA):
    # returns ndarray with dimensions num_examples x 4 x len of longest sentence
    # question words followed by answer words without a separator
    # or first half of the sentence followed by second half of the sentence

    # assuming that corpus_sentences is a dictionary with sentence:words pairs and QA_sentences is a list
    # of namedtuples with keys question, answers, labels

    num_QA_examples = int(num_examples * prop_QA)
    num_corpus_examples = num_examples - num_QA_examples

    QA_indices = np.random.permutation(np.arange(len(QA_sentences)))[:num_QA_examples]
    corpus_indices = np.random.permutation(np.arange(len(corpus_sentences)))[:num_corpus_examples]

    QA_combined = []
    for QA_i in QA_indices:
        current_question = QA_sentences[QA_i].question
        current_QA_words = []
        for a in QA_sentences[QA_i].answers:
            current_QA_words.append(current_question.extend(a))

        current_QA_words = [qa for _, qa in sorted(zip(QA_sentences[QA_i].labels, current_QA_words), key= lambda x: x[0], reverse=True)]
        QA_combined.append(current_QA_words)

    corpus_combined = []
    for corpus_i in corpus_indices:
        correct_sentence = [corpus_sentences[corpus_i]]
        if len(correct_sentence) <= 4:
            continue
        first_half = correct_sentence[:len(correct_sentence)//2]
        current_corpus_words = []
        for _ in range(3):
            iterating = True
            while iterating:
                random_corpus_i = np.random.randint(len(corpus_sentences), size=None)
                iterating = random_corpus_i is corpus_i

            current_corpus_words.append(first_half.extend(corpus_sentences[corpus_i][-len(corpus_sentences)//2:]))
        corpus_combined.append(correct_sentence.extend(current_corpus_words))

    all_words = QA_combined.extend(corpus_combined)

    return all_words


