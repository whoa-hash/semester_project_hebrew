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


def extract_features(sentence: List[str], position: int):
    d = {}
    word = sentence[position][0]
    d['first_letter'] = word[0]
    d['last_letter'] = word[-1]
    d['last_two_letters'] = word[-2:]
    d['endsAlef'] = word[-1] == 'א'
    d['endsOs'] = word[-2:] == 'ות'
    d['endsIn'] = word[-2:] == 'ין'
    d['beginHeh'] = word[0] == 'ה'
    d['beginShin'] = word[0] == 'ש'

    return d


print(extract_features(["hello my name is", "my name is you", "name is her", "is"], 2))
