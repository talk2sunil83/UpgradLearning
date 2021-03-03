# %%
# import urllib
# import pyodbc
# from sqlalchemy import create_engine

# # params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 10.0};SERVER=tmapdwsvr1.database.windows.net;DATABASE=TmapTWoDSql;UID=tmapdwadmin;PWD=TMAPazureDWsvr$1")
# params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};SERVER=tmapdwsvr1.database.windows.net;DATABASE=TmapTWoDSql;UID=tmapdwadmin;PWD=TMAPazureDWsvr$1")
# # %%
# params

# # %%
# engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

# # %%
# engine.connect()

# # %%
# engine.raw_connection


# %%


from nltk.parse import pchart
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Callable, Tuple
from math import log2
from IPython.display import display
from nltk.metrics.distance import edit_distance
from functools import reduce
from nltk.tokenize import sent_tokenize
from math import log10
from nltk.parse import RecursiveDescentParser
import nltk
from nltk import tree
from nltk import parse, CFG
# %%
# https://explosion.ai/demos/displacy?text=man%20saw%20dogs&model=en_core_web_sm&cpu=1&cph=1


def gini(a: int, b: int) -> float:
    t = float(a+b)
    pa, pb = a/t, b/t
    return pa*(1-pa) + pb*(1-pb)  # 1 - ((a/t)**2 + (b/t)**2)


def entropy(a: int, b: int) -> float:
    t = float(a+b)
    pa, pb = a/t, b/t

    return (pa*log2(pa) if pa > 0. else 0. + pb*log2(pb) if pb > 0. else 0.)*(-1.)


def calc_gain(original: Tuple[int, int], first: Tuple[int, int], second: Tuple[int, int], fun: Callable[[int, int], float], n=4):
    o1, o2 = original
    c1f, c2f = first
    c1s, c2s = second
    if (o1+o2) != (c1f + c2f + c1s + c2s):
        print("Something wrong in split")
        return
    t = float(o1+o2)
    pa, pb = o1/t, o2/t
    imp_o, imf, ims = fun(o1, o2), fun(c1f, c2f), fun(c1s, c2s)
    imp_n = (pa*imf + pb*ims)
    tg = imp_o - imp_n
    print(f"pa:{round(pa,n)},\npb:{round(pb,n)},\nimp_o:{round(imp_o,n)},\n----\nimp_f:{round(imf,n)},\nimp_s:{round(ims,n)},\nimp_n:{round(imp_n,n)},\ntg:{round(tg,n)}")


def get_m(b1, b2):
    return np.linalg.inv(b2)@b1  # np.matmul(np.linalg.pinv(b2), b1)


def get_v2(b1, b2, v1):
    m = get_m(b1, b2)
    v2 = m@v1  # np.matmul(m, v1)
    print(f"m:\n{m}\n----\nv2:\n{v2}")


def preprocess(document):
    document = document.lower()
    words = word_tokenize(document)
    words = [word for word in words if word not in stopwords.words("english")]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    document = " ".join(words)
    return document


def calc_tfidf(documents, word):
    documents = [preprocess(document) for document in documents]
    vectorizer = TfidfVectorizer()
    tf_idf_vals = vectorizer.fit_transform(documents)
    score = tf_idf_vals[1].toarray()[0][vectorizer.get_feature_names().index(word)]
    print(score)


def get_soundex(token):
    """Get the soundex code for the string"""
    token = token.upper()

    soundex = ""

    # first letter of input is always the first letter of soundex
    soundex += token[0]

    # create a dictionary which maps letters to respective soundex codes. Vowels and 'H', 'W' and 'Y' will be represented by '.'
    dictionary = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6", "AEIOUHWY": "."}

    for char in token[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != '.':
                    if code != soundex[-1]:
                        soundex += code

    # trim or pad to make soundex a 4-character code
    soundex = soundex[:4].ljust(4, "0")

    print(soundex)


def lev_distance(source='', target=''):
    import pandas as pd
    n1, n2 = len(source), len(target)
    matrix = [[0 for i1 in range(n1 + 1)] for i2 in range(n2 + 1)]
    for i1 in range(1, n1 + 1):
        matrix[0][i1] = i1
    for i2 in range(1, n2 + 1):
        matrix[i2][0] = i2
    for i2 in range(1, n2 + 1):
        for i1 in range(1, n1 + 1):
            if (source[i1-1] == target[i2-1]):
                value = matrix[i2-1][i1-1]               # top-left cell value
            else:
                value = min(matrix[i2-1][i1] + 1,      # left cell value     + 1
                            matrix[i2][i1-1] + 1,      # top cell  value     + 1
                            matrix[i2-1][i1-1] + 1)      # top-left cell value + 1

            matrix[i2][i1] = value
    frame = pd.DataFrame(matrix, columns=[c for c in "."+source], index=[c for c in "."+target])
    print(matrix[-1][-1])
    display(frame)


def get_edit_distance(source='', target=''):
    return lev_distance(source, target)


def get_word_prob(word, docs):
    return len([sent for sent in docs if word.lower() in sent]) / len(docs)


def get_pmi(term: str, text: str, round_till: int = 3) -> float:
    docs = sent_tokenize(text.lower())
    return round(log10(get_word_prob(term, docs) / reduce(
        (lambda p1, p2: p1*p2), [get_word_prob(w, docs) for w in term.split()])), round_till)


# %%
# Top Down parser
# nltk.app.rdparser()
groucho_grammar = nltk.CFG.fromstring(
    '''
        S -> NP VP
        NP -> Det N| Det Adj N
        VP -> V NP |V
        
        Det -> 'the'
        V -> 'booked'
        N -> 'man' | 'flight'
        Adj -> 'old'
    ''')
sent = list('The old man booked the flight'.lower().split())
parser = nltk.ChartParser(groucho_grammar)
for tree in parser.parse(sent):
    print(tree)
# %%
# Bottom up parser
# nltk.app.srparser()
grammar = CFG.fromstring("""
S -> NP VP
NP -> DT N
VP -> V | V PP
PP -> P NP
DT -> 'the'
N -> 'child' | 'kitchen'
V -> 'ran'
P -> 'to'
    """)

str_val = "the child ran to the kitchen"
srp = parse.ShiftReduceParser(grammar, trace=2)

for res in srp.parse(list(str_val.lower().split())):
    if res is None:
        print("No Res")
    print(res)

# %%
pcfg_grammar = nltk.PCFG.fromstring("""
S -> NP VP [1.0]
PP -> P NP [1.0]
VP -> V NP [0.3]| VP PP [0.4] | V [0.3]
NP -> DT N [0.4] | N [0.35] | NP PP [0.25]
P -> 'on' [0.4] | 'with' [0.6]
V -> 'sleeps' [1.0]
N -> 'child' [0.5]| 'bed' [0.3] | 'cushion' [0.2]
DT -> 'the' [1.0]
    """)
str = "the child sleeps on the bed with cushion"
parser = pchart.InsideChartParser(pcfg_grammar)
for t in parser.parse(str.split()):
    print(t)

# %%


calc_gain((6, 4), (5, 2), (1, 2), gini)
print("\n")
calc_gain((6, 4), (5, 1), (1, 3), gini)
# # %%
# calc_gain((6, 4), (5, 2), (1, 2), entropy)
# print("\n")
# calc_gain((6, 4), (5, 1), (1, 3), entropy)

# %%
calc_gain((11, 7), (3, 5), (8, 2), gini)
print("\n")
calc_gain((11, 7), (9, 3), (2, 4), gini)

# %%

# %%
calc_gain((11, 7), (3, 5), (8, 2), gini)
print("\n")
calc_gain((11, 7), (9, 3), (2, 4), gini)
# %%
gini(2, 3)
# %%
calc_gain((20, 30), (10, 20), (10, 10), gini, 8)

# %%
# %%
calc_gain((40, 10), (10, 30), (10, 0), gini, 8)

# %%
calc_gain((4, 8), (2, 4), (2, 4), gini, 8)

# %%
4/9
# %%
calc_gain((4, 2), (2, 2), (2, 0), gini, 8)
# %%
1/3
# %%
calc_gain((2, 2), (1, 1), (1, 1), gini, 8)

# %%
calc_gain((2, 2), (1, 1), (1, 1), gini, 8)

# %%
calc_gain((2, 2), (2, 1), (0, 1), gini, 8)

# %%


# %%
v1 = np.array([[3], [2]])
B1 = np.array([[1, 0], [0, 1]])
B2 = np.array([[3, -3], [4, -5]])
get_v2(B1, B2, v1)

# %%


documents = ["The coach lumbered on again, with heavier wreaths of mist closing round it as it began the descent.",
             "The guard soon replaced his blunderbuss in his arm-chest, and, having looked to the rest of its contents, and having looked to the supplementary pistols that he wore in his belt, looked to a smaller chest beneath his seat, in which there were a few smith's tools, a couple of torches, and a tinder-box.",
             "For he was furnished with that completeness that if the coach-lamps had been blown and stormed out, which did occasionally happen, he had only to shut himself up inside, keep the flint and steel sparks well off the straw, and get a light with tolerable safety and ease (if he were lucky) in five minutes.",
             "Jerry, left alone in the mist and darkness, dismounted meanwhile, not only to ease his spent horse, but to wipe the mud from his face, and shake the wet out of his hat-brim, which might be capable of holding about half a gallon.",
             "After standing with the bridle over his heavily-splashed arm, until the wheels of the mail were no longer within hearing and the night was quite still again, he turned to walk down the hill."]
calc_tfidf(documents, "belt")


# %%
get_soundex("Upgrad")

# %%
get_soundex("Mumbai")
get_soundex("Bombay")

# %%
get_edit_distance('Mumbai', 'Bombay')

# %%
