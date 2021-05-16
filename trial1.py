import nltk
from nltk import *
from tf.app import use
from tf.core.api import Api
from tf.fabric import Fabric

import collections
import sys

TF = Fabric(locations=r'C:\Users\shira\text-fabric-data\etcbc\bhsa\tf\c')
TF.explore(show=True)
# TF.loadAll(silent=None)

features = 'sp root g_word g_word_utf8 trailer_utf8 ls typ rela function qere_utf8 qere'
TF.load(features, add=False)
api = TF.load('sp root g_word g_word_utf8 trailer_utf8 ls typ rela function qere_utf8 qere')
api = TF.load('sp root g_word g_word_utf8 trailer_utf8 ls typ rela function qere_utf8 qere')
print(api)
# TF.ensureLoaded(features)
api.makeAvailableIn(globals())
# print(api)
# A = use('corpus', hoist=globals())
F = api.F
T = api.T
C = api.C
L = api.L


def print_original_words():
    for i in range(1, 12):
        print(api.T.text([i], 'text-orig-full'))


print_original_words()


def print_word_pos():
    count = 0
    for verse in F.otype.s('verse'):
        book = T.sectionFromNode(verse)[0]
        print(T.sectionFromNode(verse)[0], T.sectionFromNode(verse)[1], str(T.sectionFromNode(verse)[2]) + ':', end=' ')
        pasuk = []
        s = ''  # type: str
        aggregate_pos = []
        aggregate_pos_subtype = []
        aggregate_lex = []
        parts = L.d(verse, 'half_verse')
        for hv, half_verse in enumerate(parts):
            words = L.d(half_verse, 'word')
            for w in words:
                if count == 30:
                    break
                count += 1
                part_of_speech, lex, trailer = F.sp.v(w), F.g_word_utf8.v(w), F.trailer_utf8.v(w)
                print(part_of_speech, lex, trailer)


print_word_pos()
# A = use('corpus', hoist=globals())
# print(A)
text = word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))
text = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃"
import subprocess
import unicodedata
def remover_nkd(text):
    nikud = text
    normalized = unicodedata.normalize('NFKD', nikud)  # Reduce hebrew vowel ניקוד marks
    removed_nikud = ""
    # since hebrew reads the other way compared to english
    for char in range(len(normalized), 0, -1):
        character = normalized[char - 1]
        if not unicodedata.combining(character):
            removed_nikud = character + removed_nikud
    return removed_nikud
nikud = text
normalized = unicodedata.normalize('NFKD', nikud)  # Reduce hebrew vowel ניקוד marks
removed_nikud = ""
# since hebrew reads the other way compared to english
for char in range(len(normalized), 0, -1):
    character = normalized[char - 1]
    if not unicodedata.combining(character):
        removed_nikud = character + removed_nikud

print(removed_nikud)

# features list
features = []
# past binyan kal
# there are different options for the nekudos could include a sheva or not
I = text.startswith("ָ")
features.append(I)
you_single_male = "תִִִִֵָתָ"
you_single_female = "ת"
features.append(you_single_female)
them_male = "תֶם"
features.append(them_male)
them_female = "תֶן"
features.append(them_female)
text = word_tokenize("בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ")
text_eng = word_tokenize("hello my name is joe")
text2 = "אָכָֽלְתָּ"
this = nltk.pos_tag(text)
trial = nltk.pos_tag(text_eng)
print("nltk tagger: ", this)
print("nltk eng tagger", trial)
print("features list: ", features)
