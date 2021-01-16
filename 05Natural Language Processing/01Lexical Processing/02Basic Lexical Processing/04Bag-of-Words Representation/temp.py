# %%
import re
from numpy.core.numeric import count_nonzero
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('max_colwidth', 100)

# %%
d1 = "there was a place on my ankle that was itching"
d2 = "but I did not scratch it"
d3 = "and then my ear began to itch"
d4 = "and next my back"
docs = [d1, d2, d3, d4]
# %%
len(set(list(d1.split()) + list(d2.split()) +
        list(d3.split()) + list(d4.split())))
# %%


def preprocess(document):
    'changes document to lower case and removes stopwords'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    # join words to make sentence
    document = " ".join(words)

    return document


documents = [preprocess(document) for document in docs]
print(documents)
# %%
count_vec = CountVectorizer()
bow = count_vec.fit_transform(docs)

print(bow)
# %%
bow.toarray()
# %%
data = pd.read_csv("SMSSpamCollection.txt", sep='\t', names=['label', 'message'])
data.head()
# %%
messages = data['message'][:100].apply(preprocess)
len(messages)
# %%
cv_msg = CountVectorizer()
bow_msg = cv_msg.fit_transform(messages)

# %%
bow_msg_df = pd.DataFrame(bow_msg.toarray(), columns=cv_msg.get_feature_names())
bow_msg_df.shape
# %%

word = "playing"
word = "played"

# create function to chop off the suffixes 'ing' and 'ed'


def stemmer(word):
    # write your code here
    word = re.sub("[ing$|ed$]", "", word, flags=re.I)
    return word


# stem word -- don't change the following code, it is used to evaluate your code
print(stemmer(word))
# %%
