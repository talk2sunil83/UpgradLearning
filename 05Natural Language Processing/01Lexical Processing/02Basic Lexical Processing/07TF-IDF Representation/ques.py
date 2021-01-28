# %%
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# consider the following set of documents
documents = ["The coach lumbered on again, with heavier wreaths of mist closing round it as it began the descent.",
             "The guard soon replaced his blunderbuss in his arm-chest, and, having looked to the rest of its contents, and having looked to the supplementary pistols that he wore in his belt, looked to a smaller chest beneath his seat, in which there were a few smith's tools, a couple of torches, and a tinder-box.",
             "For he was furnished with that completeness that if the coach-lamps had been blown and stormed out, which did occasionally happen, he had only to shut himself up inside, keep the flint and steel sparks well off the straw, and get a light with tolerable safety and ease (if he were lucky) in five minutes.",
             "Jerry, left alone in the mist and darkness, dismounted meanwhile, not only to ease his spent horse, but to wipe the mud from his face, and shake the wet out of his hat-brim, which might be capable of holding about half a gallon.",
             "After standing with the bridle over his heavily-splashed arm, until the wheels of the mail were no longer within hearing and the night was quite still again, he turned to walk down the hill."]


# preprocess document
def preprocess(document):
    'changes document to lower case, removes stopwords and stems words'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    # stem
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # join words to make sentence
    document = " ".join(words)

    return document


# preprocess documents using the preprocess function and store the documents again in a list
documents = [preprocess(document) for document in documents]  # write code here

# create tf-idf matrix
## write code here ##

vectorizer = TfidfVectorizer()
tf_idf_vals = vectorizer.fit_transform(documents)


# extract score
# replace -1 with the score of 'belt' in document two. You can manually write the value by looking at the tf_idf model
score = tf_idf_vals[1].toarray()[0][vectorizer.get_feature_names().index("belt")]

# print the score -- don't change the following piece od code, it's used to evaluate your code
print(round(score, 4))
# %%
tf_idf_vals
