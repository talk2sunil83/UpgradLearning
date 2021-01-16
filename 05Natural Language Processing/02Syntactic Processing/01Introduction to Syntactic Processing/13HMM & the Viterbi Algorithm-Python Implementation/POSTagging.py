# %% [markdown]
# ## POS Tagging, HMMs, Viterbi
#
# Let's learn how to do POS tagging by Viterbi Heuristic using tagged Treebank corpus. Before going through the code, let's first understand the pseudo-code for the same.
#
# 1. Tagged Treebank corpus is available (Sample data to training and test data set)
#    - Basic text and structure exploration
# 2. Creating HMM model on the tagged data set.
#    - Calculating Emission Probabaility: P(observation|state)
#    - Calculating Transition Probability: P(state2|state1)
# 3. Developing algorithm for Viterbi Heuristic
# 4. Checking accuracy on the test data set
#
#
# ## 1. Exploring Treebank Tagged Corpus

# %%
# Importing libraries
from typing import Sequence, Tuple
import nltk
import re
import pprint
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


# %%
# reading the Treebank tagged sentences
wsj = list(nltk.corpus.treebank.tagged_sents())


# %%
# first few tagged sentences
print(wsj[:40])


# %%
# Splitting into train and test
random.seed(1234)
train_set, test_set = train_test_split(wsj, test_size=0.3)

print(len(train_set))
print(len(test_set))
print(train_set[:40])


# %%
# Getting list of tagged words
train_tagged_words = [tup for sent in train_set for tup in sent]
len(train_tagged_words)


# %%
# tokens
tokens = [pair[0] for pair in train_tagged_words]
tokens[:10]


# %%
# vocabulary
V = set(tokens)
print(len(V))


# %%
# number of tags
T = set([pair[1] for pair in train_tagged_words])
len(T)


# %%
print(T)

# %% [markdown]
# ## 2. POS Tagging Algorithm - HMM
#
# We'll use the HMM algorithm to tag the words. Given a sequence of words to be tagged, the task is to assign the most probable tag to the word.
#
# In other words, to every word w, assign the tag t that maximises the likelihood P(t/w). Since P(t/w) = P(w/t). P(t) / P(w), after ignoring P(w), we have to compute P(w/t) and P(t).
#
#
# P(w/t) is basically the probability that given a tag (say NN), what is the probability of it being w (say 'building'). This can be computed by computing the fraction of all NNs which are equal to w, i.e.
#
# P(w/t) = count(w, t) / count(t).
#
#
# The term P(t) is the probability of tag t, and in a tagging task, we assume that a tag will depend only on the previous tag. In other words, the probability of a tag being NN will depend only on the previous tag t(n-1). So for e.g. if t(n-1) is a JJ, then t(n) is likely to be an NN since adjectives often precede a noun (blue coat, tall building etc.).
#
#
# Given the penn treebank tagged dataset, we can compute the two terms P(w/t) and P(t) and store them in two large matrices. The matrix of P(w/t) will be sparse, since each word will not be seen with most tags ever, and those terms will thus be zero.
#
# %% [markdown]
# ### Emission Probabilities

# %%
# computing P(w/t) and storing in T x V matrix
t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))


# %%
# compute word given tag: Emission Probability
def word_given_tag(word, tag, train_bag=train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    count_w_given_tag = len(w_given_tag_list)

    return (count_w_given_tag, count_tag)


# %%
# examples

# large
print("\n", "large")
print(word_given_tag('large', 'JJ'))
print(word_given_tag('large', 'VB'))
print(word_given_tag('large', 'NN'), "\n")

# will
print("\n", "will")
print(word_given_tag('will', 'MD'))
print(word_given_tag('will', 'NN'))
print(word_given_tag('will', 'VB'))

# book
print("\n", "book")
print(word_given_tag('book', 'NN'))
print(word_given_tag('book', 'VB'))


# %%
def emission_prob(word: str, tag: str, train_bag=train_tagged_words) -> float:
    wc, tc = word_given_tag(word, tag, train_bag)
    return wc/tc if tc > 0. else 0.


emission_prob("Android", "NN")

# %% [markdown]
# ### Transition Probabilities

# %%
# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability


def t2_given_t1(t2, t1, train_bag=train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index] == t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# %%
# examples
print(t2_given_t1(t2='NNP', t1='JJ'))
print(t2_given_t1('NN', 'JJ'))
print(t2_given_t1('NN', 'DT'))
print(t2_given_t1('NNP', 'VB'))
print(t2_given_t1(',', 'NNP'))
print(t2_given_t1('PRP', 'PRP'))
print(t2_given_t1('VBG', 'NNP'))


# %%
# Please note P(tag|start) is same as P(tag|'.')
print(t2_given_t1('DT', '.'))
print(t2_given_t1('VBG', '.'))
print(t2_given_t1('NN', '.'))
print(t2_given_t1('NNP', '.'))

# %%


def transition_prob(t2: str, t1: str, train_bag: Sequence[Tuple[str, str]] = train_tagged_words, print_count: bool = False):
    count_t2_t1, count_t1 = t2_given_t1(t2, t1, train_bag)
    if print_count:
        print(f"count_t2_t1: {count_t2_t1}, count_t1: {count_t1}")
    return count_t2_t1/count_t1 if count_t1 > 0. else 0.


# %%
transition_prob("VB", "MD", print_count=True)


# %%
q5 =[("Donald","NN"),("Trump","NN"), ("is","VB"), ("the","DT"), ("current","JJ"), ("President","NN"), ("of","IN"), ("US","NN"), ("Before","IN"), ("entering","VB"), ("into","IN"), ("dirty","JJ"), ("politics","NN"), ("he","PRP"),("was","VB"), ("a","DT"), ("domineering","JJ"), ("businessman","NN"), ("and","CC"), ("television","NN"), ("personality","NN"),("Trump","NN"), ("entered","VB"), ("the","DT"), ("2016","CD"), ("presidential","JJ"), ("race","NN"), ("as","IN"), ("a","DT"), ("Republican","NN"), ("and","CC"), ("defeated","VBD"), ("16","CD"), ("opponents","NN")]
# %%
# %%
t2_given_t1("NN", "JJ", q5)

# %%
# creating t x t transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)


tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)):

        tags_matrix[i, j] = transition_prob(t2, t1)  # t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]

# %%
tags_matrix

# %%
# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns=list(T), index=list(T))
# %%
tags_df
# %%
tags_df.loc['.', :]


# %%
# heatmap of tags matrix
# T(i, j) means P(tag j given tag i)
plt.figure(figsize=(18, 12))
sns.heatmap(tags_df)
plt.show()


# %%
# frequent tags
# filter the df to get P(t2, t1) > 0.5
tags_frequent = tags_df[tags_df > 0.5]
plt.figure(figsize=(18, 12))
sns.heatmap(tags_frequent)
plt.show()

# %% [markdown]
# ## 3. Viterbi Algorithm
#
# Let's now use the computed probabilities P(w, tag) and P(t2, t1) to assign tags to each word in the document. We'll run through each word w and compute P(tag/w)=P(w/tag).P(tag) for each tag in the tag set, and then assign the tag having the max P(tag/w).
#
# We'll store the assigned tags in a list of tuples, similar to the list 'train_tagged_words'. Each tuple will be a (token, assigned_tag). As we progress further in the list, each tag to be assigned will use the tag of the previous token.
#
# Note: P(tag|start) = P(tag|'.')

# %%
len(train_tagged_words)


# %%
# Viterbi Heuristic
def Viterbi(words, train_bag=train_tagged_words):
    state = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_parts = word_given_tag(words[key], tag)
            emission_p = emission_parts[0]/emission_parts[1]
            # emission_p = emission_prob(words[key], tag)
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)
    return list(zip(words, state))

# %% [markdown]
# ## 4. Evaluating on Test Set

# %%
# Running on entire test dataset would take more than 3-4hrs.
# Let's test our Viterbi algorithm on a few sample sentences of test dataset


random.seed(1234)

# choose random 5 sents
rndom = [random.randint(1, len(test_set)) for x in range(5)]

# list of sents
test_run = [test_set[i] for i in rndom]

# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]

# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]
test_run


# %%
# tagging the test sentences
start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end-start


# %%
print("Time taken in seconds: ", difference)
print(tagged_seq)
# print(test_run_base)


# %%
# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]


# %%
accuracy = len(check)/len(tagged_seq)


# %%
accuracy


# %%
incorrect_tagged_cases = [[test_run_base[i-1], j] for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0] != j[1]]


# %%
incorrect_tagged_cases


# %%
# Testing
sentence_test = 'Twitter is the best networking social site. Man is a social animal. Data science is an emerging field. Data science jobs are high in demand.'
words = word_tokenize(sentence_test)

start = time.time()
tagged_seq = Viterbi(words)
end = time.time()
difference = end-start


# %%
print(tagged_seq)
print(difference)


# %%
tagged_seq = Viterbi(words)
end = time.time()
difference = end-start


# %%
print(tagged_seq)
# %%

q5 =[("Donald","NN"),("Trump","NN"), ("is","VB"), ("the","DT"), ("current","JJ"), ("President","NN"), ("of","IN"), ("US","NN"), ("Before","IN"), ("entering","VB"), ("into","IN"), ("dirty","JJ"), ("politics","NN"), ("he","PRP"),("was","VB"), ("a","DT"), ("domineering","JJ"), ("businessman","NN"), ("and","CC"), ("television","NN"), ("personality","NN"),("Trump","NN"), ("entered","VB"), ("the","DT"), ("2016","CD"), ("presidential","JJ"), ("race","NN"), ("as","IN"), ("a","DT"), ("Republican","NN"), ("and","CC"), ("defeated","VBD"), ("16","CD"), ("opponents","NN")]
# %%
transition_prob("JJ", "NN", q5, print_count=True)


# %%
t2_given_t1("JJ", "NN", q5)
# %%
t2_given_t1("NN", "JJ", q5)
# %%
