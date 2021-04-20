import codecs
import numpy as np
import nltk
import pycrfsuite
import sys
# from bs4 import BeautifulSoup as bs
# from bs4.element import Tag
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from typing import List
from pprint import pprint
import os
from nltk.classify.maxent import MaxentClassifier


def extract_features(sentence: List[str]):
    d = {}
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
    return d


text = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃"
extract_features(text.split())
