# %%
# load all necessary libraries
import math
from nltk.stem.porter import PorterStemmer
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('max_colwidth', 100)

# %%
d1 = "Vapour, Bangalore has a really great terrace seating and an awesome view of the Bangalore skyline"
d2 = "The beer at Vapour, Bangalore was amazing. My favourites are the wheat beer and the ale beer."
d3 = "Vapour, Bangalore has the best view in Bangalore."

docs = [d1, d2, d3]
# %%

stemmer = PorterStemmer()

# add stemming and lemmatisation in the preprocess function


def preprocess(document):
    'changes document to lower case and removes stopwords'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    # stem
    #words = [stemmer.stem(word) for word in words]

    # join words to make sentence
    document = " ".join(words)

    return document


# %%
documents = [preprocess(document) for document in docs]
print(documents)
# %%
vectorizer = TfidfVectorizer()
tfidf_model = vectorizer.fit_transform(documents)
print(tfidf_model)
# %%
tfidf_model.toarray()
# %%
vectorizer.get_feature_names()
# %%
# %%
(3/9)*math.log10(3/1)
# %%
# Vapour
(3/9)*math.log10(3/3)
