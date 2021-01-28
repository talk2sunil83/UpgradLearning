# %% [markdown]
# ## POS tagging using modified Viterbi
# %% [markdown]
# ### Data Preparation

# %%
# Importing libraries
from typing import Callable, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import random
import pprint

import time

# %%
# reading the Treebank tagged sentences
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
# %%
nltk_data[:2]
# %%
# Splitting into train and test
random.seed(1234)
train_set, test_set = train_test_split(nltk_data, test_size=0.05)

# %%
print(len(train_set))
print(len(test_set))
print(train_set[:2])
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
# number of tags
T = set([pair[1] for pair in train_tagged_words])
print(len(T))

# %%
# compute word given tag: Emission Probability


def word_given_tag(word, tag, train_bag=train_tagged_words) -> Tuple[int, int]:
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    count_w_given_tag = len(w_given_tag_list)

    return (count_w_given_tag, count_tag)


# %%
# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability


def t2_given_t1(t2, t1, train_bag=train_tagged_words) -> Tuple[int, int]:
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index] == t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

# %%


def emission_prob(word: str, tag: str, train_bag=train_tagged_words, apply_smoothing: bool = False, epsilon: float = 1e-4) -> float:
    wc, tc = word_given_tag(word, tag, train_bag)
    return wc/tc if tc > 0. else epsilon if apply_smoothing else 0.


def transition_prob(t2: str, t1: str, train_bag: Sequence[Tuple[str, str]] = train_tagged_words, apply_smoothing: bool = False, epsilon: float = 1e-4, print_count: bool = False) -> float:
    count_t2_t1, count_t1 = t2_given_t1(t2, t1, train_bag)
    if print_count:
        print(f"count_t2_t1: {count_t2_t1}, count_t1: {count_t1}")
    return count_t2_t1/count_t1 if count_t1 > 0. else epsilon if apply_smoothing else 0

# %%
transition_prob("VB", "MD", print_count=True)
# %%
# creating t x t transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)


tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)):

        tags_matrix[i, j] = transition_prob(t2, t1)  # t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns=list(T), index=list(T))
# %% [markdown]
# ### Build the vanilla Viterbi based POS tagger

# %%
# Viterbi Heuristic


def Viterbi(
        words:Sequence[str],
        tags_df:pd.DataFrame,
        emission_prob_func:Callable[[str,str, Sequence[Tuple[str, str]],bool, float],float ],
        train_bag:Sequence[Tuple[str, str]]=train_tagged_words,
        apply_smoothing: bool = False, epsilon: float = 1e-4)->Sequence[Tuple[str, str]]:
    """"Apply Viterbi Heuristics

    Args:
        words (Sequence[str]): words
        tags_df (pd.DataFrame): tags dataframe
        emission_prob_func (Callable[[str,str, Sequence[Tuple[str, str]],bool, float],float ]):function pointer to calculate emission probabilities
        train_bag (Sequence[Tuple[str, str]], optional): training set words with tags. Defaults to train_tagged_words.
        apply_smoothing (bool, optional): Defaults to False.
        epsilon (float, optional): smoothing value. Defaults to 1e-4.

    Returns:
        Sequence[Tuple(str, str)]: words with next states
    """
    state = []
    all_tags = list(set([pair[1] for pair in train_bag]))

    for idx, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in all_tags:
            if idx == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            emission_p = emission_prob_func(word, tag,train_bag, apply_smoothing, epsilon)
            state_probability = emission_p * transition_p
            p.append(state_probability)

        max_probability = max(p)
        # getting state for which probability is maximum
        state_max = all_tags[p.index(max_probability)]
        state.append(state_max)
    return list(zip(words, state))

def prepare_metrics(tagged_seq, test_run_base):
    
    # accuracy
    check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
    accuracy = len(check)/len(tagged_seq)
    incorrect_tagged_cases = [[test_run_base[i-1], j] for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0] != j[1]]
    return accuracy, incorrect_tagged_cases


# %%
# Running on entire test dataset would take more than 3-4hrs.
# Let's test our Viterbi algorithm on a few sample sentences of test dataset


random.seed(1234)

# choose random 5 sentences
random_ints = [random.randint(1, len(test_set)) for _ in range(5)]

# list of sentences
test_run = [test_set[i] for i in random_ints]

# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]

# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]
test_run

# %%
# tagging the test sentences
start = time.time()
tagged_seq = Viterbi(test_tagged_words, tags_df, emission_prob)
end = time.time()
difference = end-start


# %%
print("Time taken in seconds: ", difference)
print(tagged_seq)

# %% [markdown]
# ### Solve the problem of unknown words

# %%


# %%


# %% [markdown]
# #### Evaluating tagging accuracy

# %%


# %%


# %% [markdown]
# ### Compare the tagging accuracies of the modifications with the vanilla Viterbi algorithm

# %%


# %%


# %%


# %% [markdown]
# ### List down cases which were incorrectly tagged by original POS tagger and got corrected by your modifications

# %%


# %%


# %%
