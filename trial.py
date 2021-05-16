import nltk
from nltk import *
from tf.app import use
from tf.core.api import Api
from tf.fabric import Fabric
from trial1 import remover_nkd
import collections
import sys

TF = Fabric(locations=r'C:\Users\shira\text-fabric-data\etcbc\bhsa\tf\c')
TF.explore(show=True)

features = 'sp root g_word g_word_utf8 trailer_utf8 ls typ rela function qere_utf8 qere'
TF.load(features, add=False)
api = TF.load('sp root g_word g_word_utf8 trailer_utf8 ls typ rela function qere_utf8 qere')
api = TF.load('sp root g_word g_word_utf8 trailer_utf8 ls typ rela function qere_utf8 qere')
print(api)

api.makeAvailableIn(globals())
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
    # open a file to append the tag and the hebrew word
   # with open("output.txt", "w", encoding="utf-8") as f:
    # list of pesukim
    pesukim = []
    for verse in F.otype.s('verse'):
        # add words to the currentpasuk
        # dictionary {}, word
        # currentpasuk= [(prep, word), (art,word2)]
        currentpasuk = []
        # we have to just add on the prep, conj (vavs, hehs..)
        currentword = ""
        book = T.sectionFromNode(verse)[0]
        parts = L.d(verse, 'half_verse')

        for hv, half_verse in enumerate(parts):
            words = L.d(half_verse, 'word')
            for w in words:
                if count == 1000:  # only do 1000 words?
                    break
                count += 1
                part_of_speech, lex = F.sp.v(w), F.g_word_utf8.v(w)

                try:

                    for symbol in lex:
                        # ordinals of trup symbols to remove
                        # print(ord(symbol), symbol)  # 1445, 1448, 1443, 1428, 1430, 1425, 1469, 1447, 1435
                        # (1468 is a dagesh in the first letter)
                        if ord(symbol) == 1445 or ord(symbol) == 1448 or ord(symbol) == 1443 or ord(
                                symbol) == 1428 or ord(symbol) == 1430 or ord(symbol) == 1425 or ord(
                            symbol) == 1469 or ord(symbol) == 1447 or (ord(symbol)) == 1435 or ord(
                            symbol) == 1433 or ord(symbol) == 1431 or ord(symbol) == 1444:
                            lex = lex.replace(symbol, "")

                    #print(part_of_speech, lex)
                    currentword += lex
                    #f.write(part_of_speech + " " + lex)
                    # don't count the ha, li, prep, conj that split up the word
                    if part_of_speech == 'art' or part_of_speech == 'conj' or part_of_speech == 'prep' \
                            and len(remover_nkd(lex)) == 1:
                        # print('remove nikud', remover_nkd(lex))
                        continue
                    currentpasuk.append((part_of_speech, currentword))
                    # reset the current word to blank so that we can set it to the new lex
                    currentword = ""
                    #f.write("\n")
                    # print("wrote to the file")
                except Exception as error:
                    print("could not write to a file")
        pesukim.append(currentpasuk)
    #f.close()
    return pesukim


print_word_pos()

text = word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))
text = "בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים אֵ֥ת הַשָּׁמַ֖יִם וְאֵ֥ת הָאָֽרֶץ׃"
import subprocess
import unicodedata

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
# past binyan kal   # this has to be updated, this is just for practice
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
