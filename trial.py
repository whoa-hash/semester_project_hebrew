from tf.fabric import Fabric
from trial1 import remover_nkd

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
                # if count == 1000:  # only do 1000 words?
                #     break
                count += 1
                part_of_speech, lex, krei = F.sp.v(w), F.g_word_utf8.v(w), F.qere_utf8.v(w)
                if krei is not None:
                    lex = krei

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

                    # print(part_of_speech, lex)
                    currentword += lex
                    # f.write(part_of_speech + " " + lex)
                    # don't count the ha, li, prep, conj that split up the word
                    if part_of_speech == 'art' or part_of_speech == 'conj' or part_of_speech == 'prep' \
                            and len(remover_nkd(lex)) == 1:
                        continue
                    currentpasuk.append((part_of_speech, currentword))
                    # reset the current word to blank so that we can set it to the new lex
                    currentword = ""

                except Exception as error:
                    print("could not write to a file")
        pesukim.append(currentpasuk)
    # f.close()
    return pesukim
