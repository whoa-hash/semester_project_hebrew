import codecs
import numpy as np
import nltk
import pycrfsuite
import sys
# from bs4 import BeautifulSoup as bs
# from bs4.element import Tag
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from typing import List
from pprint import pprint
import os
from nltk.classify.maxent import MaxentClassifier


def extract_features(sentence: List[str]):
    d = {}
    list_d = []
    for i in range(len(sentence)):
        word = sentence[i]
        d['last_two_letters'] = word[-2:]
        d['beginVav'] = word[0] == 'ו'
        d['endsAlef'] = word[-1] == 'א'
        d['endsOs'] = word[-2:] == 'ות'
        d['beginHeh'] = word[0] == 'ה'
        d['beginShin'] = word[0] == 'ש'
        d['endsIm'] = word[-2:] == 'ים'
        print(d)
        new_d = d.copy()
        list_d.append(new_d)
    print("list_d is", list_d)
    return list_d


text = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃"

training_features = extract_features(text.split())
labels = []
for i in range(7):
    labels2 = ["NOUN", "VERB", "NOUN", "PREP", "NOUN", "PREP", "NOUN"]
    labels.append(labels2)

print('labels', labels, len(labels))

train_docs, test_docs, train_labels, test_labels = train_test_split(training_features, labels)

print(len(train_docs), len(test_docs))

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






# # training = [[extract_features(s, position) for position in range(len(s))] for s in data]
# # A function fo generating the list of labels for each document
# def get_labels(sentence):
#     word_label = []
#     sentence = sentence.split()
#     count = 0
#     for i in sentence:
#         if count == 0 or count == 2 or count == 4 or count == 6:
#             word_label.append((i, "NOUN"))
#         elif count == 1:
#             word_label.append((i, "VERB"))
#         else:
#             word_label.append((i, "PREP"))
#         count += 1
#     # print("list of words and labels", word_label)
#     return word_label
#
#
# training_labels = get_labels(text)
#
#
# def labels_features(sentence, li):
#     new_list = []
#     list_features = extract_features(sentence.split())
#     for feature_list in range(len(list_features)):
#         label = get_labels(sentence)[feature_list][1]
#         dictionary = list_features[feature_list]
#         print('this is the current dict of features', dictionary)
#         dict_label = (dictionary, label)
#         print(dict_label)
#         new_list.append(dict_label)
#     print("this is the final  list of features and label tuples")
#
#     return new_list
#
#
# training_features_labels = labels_features(text, training_labels)
# print(training_features_labels)

# X_train = extract_features(text.split()[0])
# y_train = get_labels(text)[0][1]
# X_test = extract_features(text.split())[2]
# y_test = get_labels(text)[2][1]
# print('y train', y_train)
# bTesting = True
# if bTesting:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# else:
#     X_train, y_train = X, y
# print("x train:", X_train,"y train:", y_train)
# trainer = pycrfsuite.Trainer(verbose=True)
# train_data = get_labels(text)
# for sent in train_data:
#             print('sent', sent)
#             tokens,labels = zip(*sent)
#             print('tokens', tokens, 'labels', labels)
            # features = [trainer._feature_func(tokens,i) for i in range(len(tokens))]
            # trainer.append(features,labels)
# # # training data to the trainer
# for xseq, yseq in zip(X_train, y_train):
#     print(xseq, yseq)
#     trainer.append(xseq, yseq)
# # #
# # # set the parameters of the model
# trainer.set_params({
#     # coefficient for L1 penalty
#     'c1': .1,
#     #coefficient for L2 penalty
#     'c2': .01,
#     'max_iterations': 30,
#     'feature.possible_transitions': True
# })
# #
# trainer.train('dikduk.crfsuite')
# print(trainer.logparser.last_iteration)
# #
# # # generate predictions
# tagger = pycrfsuite.Tagger()
# tagger.open('crf.model')
#
# if bTesting:
#     y_pred = [tagger.tag(xseq) for xseq in X_test]
#     # Let's take a look at a random sample in the testing set
#     i = 0
#     for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
#         print("%s (%s)" % (y, x))
#
