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
        d['first_letter'] = word[0]
        d['last_letter'] = word[-1]
        d['last_two_letters'] = word[-2:]
        d['endsAlef'] = word[-1] == 'א'
        d['endsOs'] = word[-2:] == 'ות'
        d['endsIn'] = word[-2:] == 'ין'
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


# training = [[extract_features(s, position) for position in range(len(s))] for s in data]
# A function fo generating the list of labels for each document
def get_labels(sentence):
    word_label = []
    sentence = sentence.split()
    count = 0
    for i in sentence:
        if count == 0 or count == 2 or count == 4 or count == 6:
            word_label.append((i, "NOUN"))
        elif count == 1:
            word_label.append((i, "VERB"))
        else:
            word_label.append((i, "PREP"))
        count += 1
    # print("list of words and labels", word_label)
    return word_label


training_labels = get_labels(text)


def labels_features(sentence, li):
    new_list = []
    list_features = extract_features(sentence.split())
    for feature_list in range(len(list_features)):
        label = get_labels(sentence)[feature_list][1]
        dictionary = list_features[feature_list]
        print('this is the current dict of features', dictionary)
        dict_label = (dictionary, label)
        print(dict_label)
        new_list.append(dict_label)
    print("this is the final  list of features and label tuples")

    return new_list


training_features_labels = labels_features(text, training_labels)
print(training_features_labels)

X = extract_features(text.split())
y = get_labels(text)
bTesting = True
if bTesting:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
else:
    X_train, y_train = X, y

trainer = pycrfsuite.Trainer(verbose=True)
#
# # training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    print(xseq, yseq)
    trainer.append(xseq, yseq)
#
# # set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': .1,

    #coefficient for L2 penalty
    'c2': .01,

    'max_iterations': 200,

    'feature.possible_transitions': True
})

trainer.train('crf.model')
#
# # generate predictions
tagger = pycrfsuite.Tagger()
tagger.open('crf.model')

if bTesting:
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    # Let's take a look at a random sample in the testing set
    i = 0
    for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
        print("%s (%s)" % (y, x))

