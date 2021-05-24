import pycrfsuite
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from trial import print_word_pos


def extract_features(pesukim: List[List[Tuple]]):  # [[('subs', 'בְּרֵאשִׁית'), ('verb', 'בָּרָא'), ('subs',
    # 'אֱלֹהִים'), ('prep', 'אֵת'), ('subs', 'הַשָּׁמַיִם'), ('prep', 'וְאֵת'), ('subs', 'הָאָרֶץ')], [('subs',
    # 'וְהָאָרֶץ')
    d = {}
    pesukim_d = []
    current_pasuk = []
    count = 0
    # get the features for each word
    for pasuk in pesukim:

        for i in range(len(pasuk)):
            d = {}
            word = pasuk[i][1]
            d['last_two_letters'] = word[-2:]
            d['last_three_letters'] = word[-3:]

            try:
                if len(word) > 0:
                    d['endsOs'] = word[-2:] == 'ות'
                    d['endsKametzHeh'] = word[-2:] == 'ָה'
                    d['endsKametzHeh'] = word[-2:] == 'ֳה'

                    d['endsIm'] = word[-2:] == 'ים'


            except Exception as e:
                # do nothing
                pass

            d['endsAlef'] = word[-1] == 'א'
            d['endsOs'] = word[-2:] == 'ות'
            d['them_female'] = word[-3:] == "תֶן"
            d['them_male'] = word[-3:] == "תֶם"

            d['endsIm'] = word[-2:] == 'ים'
            new_d = d.copy()
            current_pasuk.append(new_d)

        pesukim_d.append(current_pasuk)
        current_pasuk = []

    return pesukim_d


def extract_labels(pesukim: List[List[Tuple]]):  # [[('subs', 'בְּרֵאשִׁית'), ('verb', 'בָּרָא'), ('subs',
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

        pesukim_d.append(current_pasuk)
        current_pasuk = []

    return pesukim_d


print("--------------------------------------")

training_features = extract_features(print_word_pos())
labels = extract_labels(print_word_pos())

train_docs, test_docs, train_labels, test_labels = train_test_split(training_features, labels)
print('train docs len', len(train_docs), 'test docs len', len(test_docs))
print('test docs', test_docs)
print('train docs', train_docs)
trainer = pycrfsuite.Trainer(verbose=False)

trainer.set_params({
    'c1': 1.0,  # coefficient for L1 penalty
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
print("Predicted: ", predicted_tags)
print("Correct  : ", test_labels[1])

from sklearn.metrics import classification_report

all_true, all_pred = [], []

for i in range(len(test_docs)):
    all_true.extend(test_labels[i])
    all_pred.extend(crf_tagger.tag(test_docs[i]))

print(classification_report(all_true, all_pred))
