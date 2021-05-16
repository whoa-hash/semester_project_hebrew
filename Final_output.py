import codecs
import numpy as np
import nltk
import pycrfsuite
import sys
# from bs4 import BeautifulSoup as bs
# from bs4.element import Tag
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from typing import List, Tuple
from pprint import pprint
import os
from nltk.classify.maxent import MaxentClassifier
from trial import print_word_pos

def extract_features(pesukim: List[List[Tuple]]): # [[('subs', 'בְּרֵאשִׁית'), ('verb', 'בָּרָא'), ('subs',
    # 'אֱלֹהִים'), ('prep', 'אֵת'), ('subs', 'הַשָּׁמַיִם'), ('prep', 'וְאֵת'), ('subs', 'הָאָרֶץ')], [('subs',
    # 'וְהָאָרֶץ')
    d = {}
    pesukim_d = []
    current_pasuk = []
    count = 0
    # get the features for each word
    for pasuk in pesukim:
        count += 1
        for i in range(len(pasuk)):
            word = pasuk[i][1]
            # if count == 2:
            #     print(word)
            d['last_two_letters'] = word[-2:]
            d['beginVav'] = word[0] == 'ו'
            d['endsAlef'] = word[-1] == 'א'
            d['endsOs'] = word[-2:] == 'ות'
            d['beginHeh'] = word[0] == 'ה'
            d['beginShin'] = word[0] == 'ש'
            d['endsIm'] = word[-2:] == 'ים'

            new_d = d.copy()
            current_pasuk.append(new_d)
        # if count == 2:
        #     print('current_pasuk', current_pasuk, len(current_pasuk))
        # if not current_pasuk:
        #     current_pasuk = [word]

        pesukim_d.append(current_pasuk)
        current_pasuk = []
        # if count == 2:
        #     print("list_d is", pesukim_d)
    return pesukim_d

def extract_labels(pesukim: List[List[Tuple]]): # [[('subs', 'בְּרֵאשִׁית'), ('verb', 'בָּרָא'), ('subs',
    # 'אֱלֹהִים'), ('prep', 'אֵת'), ('subs', 'הַשָּׁמַיִם'), ('prep', 'וְאֵת'), ('subs', 'הָאָרֶץ')], [('subs',
    # 'וְהָאָרֶץ')
    d = {}
    pesukim_d = []
    current_pasuk = []
    count = 0
    # get the labels for each word
    for pasuk in pesukim:
        count += 1
        for i in range(len(pasuk)):
            label = pasuk[i][0]
            current_pasuk.append(label)
        # if count == 2:
        #     print('current_pasuk', current_pasuk, len(current_pasuk))
        pesukim_d.append(current_pasuk)
        current_pasuk = []
        # if count == 2:
        #     print("list_d is", pesukim_d)
    return pesukim_d



text = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃"
print("--------------------------------------")
print('pesukim', len(extract_features(print_word_pos())[:2]), extract_features(print_word_pos())[:2])
training_features = extract_features(print_word_pos())
labels = extract_labels(print_word_pos())
print('labels', len(extract_labels(print_word_pos())[:2]), extract_features(print_word_pos())[:2])

train_docs, test_docs, train_labels, test_labels = train_test_split(training_features, labels)
print('train docs len', len(train_docs), 'test docs len', len(test_docs))
print('test docs', test_docs)
print('train docs', train_docs)
trainer = pycrfsuite.Trainer(verbose=False)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 200,

    'feature.possible_transitions': True
})
for xseq, yseq in zip(train_docs, train_labels):
    trainer.append(xseq, yseq)

trainer.train(r'C:\Users\shira\PycharmProjects\semester_project_hebrew\model.crfsuite')
print(trainer.logparser.last_iteration)

crf_tagger = pycrfsuite.Tagger()
crf_tagger.open(r'C:\Users\shira\PycharmProjects\semester_project_hebrew\model.crfsuite')
predicted_tags = crf_tagger.tag(test_docs[1])
print("Predicted: ",predicted_tags)
print("Correct  : ",test_labels[1])

from sklearn.metrics import classification_report

all_true, all_pred = [], []

for i in range(len(test_docs)):
    all_true.extend(test_labels[i])
    all_pred.extend(crf_tagger.tag(test_docs[i]))

print(classification_report(all_true, all_pred))

