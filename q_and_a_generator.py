import json_lines
from collections import namedtuple
import numpy as np
import os

def get_qa_info(difficulty, subset, limit=0):
    limit_bool = True if np.bool(limit) else False
    header = r'C:\Users\Mitch\PycharmProjects\ARC'

    if difficulty == 'EASY':
        if subset == 'TRAIN':
            EASY_TRAIN_filename = r'ARC-V1-Feb2018-2\ARC-Easy\ARC-Easy-Train.jsonl'
            EASY_TRAIN_allinfo = []
            EASY_TRAIN_document = namedtuple('EASY_TRAIN_document', 'id question choices_text choices_labels answer')

            with open(os.path.join(header, EASY_TRAIN_filename), 'rb') as f:
                for item_no, item in enumerate(json_lines.reader(f)):
                    id = item['id']
                    question = item['question']['stem']
                    choices_text = [choice['text'] for choice in item['question']['choices']]
                    choices_labels = [choice['label'] for choice in item['question']['choices']]
                    answer = item['answerKey']

                    EASY_TRAIN_allinfo.append(EASY_TRAIN_document(id, question, choices_text, choices_labels, answer))

                    if limit_bool and item_no > limit:
                        break
            return EASY_TRAIN_allinfo
        elif subset == 'DEV':
            EASY_DEV_filename = r'ARC-V1-Feb2018-2\ARC-Easy\ARC-Easy-Dev.jsonl'
            EASY_DEV_allinfo = []
            EASY_DEV_document = namedtuple('EASY_DEV_docuemnt', 'id question choices_text choices_labels answer')

            with open(os.path.join(header, EASY_DEV_filename), 'rb') as f:
                for item_no, item in enumerate(json_lines.reader(f)):
                    id = item['id']
                    question = item['question']['stem']
                    choices_text = [choice['text'] for choice in item['question']['choices']]
                    choices_labels = [choice['label'] for choice in item['question']['choices']]
                    answer = item['answerKey']

                    EASY_DEV_allinfo.append(EASY_DEV_document(id, question, choices_text, choices_labels, answer))

                    if limit_bool and item_no > limit:
                        break
            return EASY_DEV_allinfo
        else:
            EASY_TEST_filename = r'ARC-V1-Feb2018-2\ARC-Easy\ARC-Easy-Test.jsonl'
            EASY_TEST_allinfo = []
            EASY_TEST_document = namedtuple('EASY_TEST_allinfo', 'id question choices_text choices_labels answer')

            with open(os.path.join(header, EASY_TEST_filename), 'rb') as f:
                for item_no, item in enumerate(json_lines.reader(f)):
                    id = item['id']
                    question = item['question']['stem']
                    choices_text = [choice['text'] for choice in item['question']['choices']]
                    choices_labels = [choice['label'] for choice in item['question']['choices']]
                    answer = item['answerKey']

                    EASY_TEST_allinfo.append(EASY_TEST_document(id, question, choices_text, choices_labels, answer))

                    if limit_bool and item_no > limit:
                        break
            return EASY_TEST_allinfo
    else:
        if subset == 'TRAIN':
            CHALLENGE_TRAIN_filename = r'ARC-V1-Feb2018-2\ARC-Challenge\ARC-Challenge-Train.jsonl'
            CHALLENGE_TRAIN_allinfo = []
            CHALLENGE_TRAIN_document = namedtuple('CHALLENGE_TRAIN_allinfo', 'id question choices_text choices_labels answer')

            with open(os.path.join(header, CHALLENGE_TRAIN_filename), 'rb') as f:
                for item_no, item in enumerate(json_lines.reader(f)):
                    id = item['id']
                    question = item['question']['stem']
                    choices_text = [choice['text'] for choice in item['question']['choices']]
                    choices_labels = [choice['label'] for choice in item['question']['choices']]
                    answer = item['answerKey']

                    CHALLENGE_TRAIN_allinfo.append(CHALLENGE_TRAIN_document(id, question, choices_text, choices_labels, answer))

                    if limit_bool and item_no > limit:
                        break
            return CHALLENGE_TRAIN_allinfo
        elif subset == 'DEV':
            CHALLENGE_DEV_filename = r'ARC-V1-Feb2018-2\ARC-Challenge\ARC-Challenge-Dev.jsonl'
            CHALLENGE_DEV_allinfo = []
            CHALLENGE_DEV_document = namedtuple('CHALLENGE_DEV_allinfo', 'id question choices_text choices_labels answer')

            with open(os.path.join(header, CHALLENGE_DEV_filename), 'rb') as f:
                for item_no, item in enumerate(json_lines.reader(f)):
                    id = item['id']
                    question = item['question']['stem']
                    choices_text = [choice['text'] for choice in item['question']['choices']]
                    choices_labels = [choice['label'] for choice in item['question']['choices']]
                    answer = item['answerKey']

                    CHALLENGE_DEV_allinfo.append(CHALLENGE_DEV_document(id, question, choices_text, choices_labels, answer))

                    if limit_bool and item_no > limit:
                        break
            return CHALLENGE_DEV_allinfo
        else:
            CHALLENGE_TEST_filename = r'ARC-V1-Feb2018-2\ARC-Challenge\ARC-Challenge-Test.jsonl'
            CHALLENGE_TEST_allinfo = []
            CHALLENGE_TEST_document = namedtuple('CHALLENGE_TEST_allinfo', 'id question choices_text choices_labels answer')

            with open(os.path.join(header, CHALLENGE_TEST_filename), 'rb') as f:
                for item_no, item in enumerate(json_lines.reader(f)):
                    id = item['id']
                    question = item['question']['stem']
                    choices_text = [choice['text'] for choice in item['question']['choices']]
                    choices_labels = [choice['label'] for choice in item['question']['choices']]
                    answer = item['answerKey']

                    CHALLENGE_TEST_allinfo.append(CHALLENGE_TEST_document(id, question, choices_text, choices_labels, answer))

                    if limit_bool and item_no > limit:
                        break
            return CHALLENGE_TEST_allinfo
