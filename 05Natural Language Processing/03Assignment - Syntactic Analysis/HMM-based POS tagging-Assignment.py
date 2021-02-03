# %% [markdown]
# ## POS tagging using modified Viterbi
# %% [markdown]
# ### Data Preparation

# %%
# Importing libraries
from IPython.display import display
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
test_tagged_words = [tup for sent in test_set for tup in sent]
len(train_tagged_words)


# %% [markdown]
# ### Helper utilities
# %%

# Viterbi Heuristic
def ViterbiPipe(words: Sequence[str], all_tags: Sequence[str],
                emission_prob_func: Callable[[str, str], float],
                transition_prob_func: Callable[[str, str], float],
                prob_calculator: Callable[[float, float], float], is_testing: bool = False) -> Sequence[Tuple[str, str]]:
    state = []
    for word_idx, word in enumerate(words):
        p = []
        for tag_idx, tag in enumerate(set(all_tags)):
            transition_p = transition_prob_func(all_tags[tag_idx-1] if word_idx>0 else ".", tag, is_testing)
            emission_p = emission_prob_func(word, tag, is_testing)
            state_probability = prob_calculator(emission_p, transition_p)
            p.append(state_probability)

        max_probability = max(p)
        # getting state for which probability is maximum
        state_max = all_tags[p.index(max_probability)]
        state.append(state_max)
    return list(zip(words, state))


# compute word given tag: Emission Probability
def word_given_tag(word, tag, train_bag=train_tagged_words) -> Tuple[int, int]:
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    count_w_given_tag = len(w_given_tag_list)

    return (count_w_given_tag, count_tag)

# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability


def t2_given_t1(t2, t1, tags) -> Tuple[int, int]:
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index] == t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


def prepare_eval_metrics(predicted_word_tags: Sequence[Tuple[str, str]], actual_word_tags: Sequence[Tuple[str, str]]) -> Tuple[float, Sequence[Sequence]]:

    # accuracy
    check = [i for i, j in zip(predicted_word_tags, actual_word_tags) if i == j]
    accuracy = len(check)/len(predicted_word_tags)
    incorrect_tagged_cases = [[actual_word_tags[i-1], j] for i, j in enumerate(zip(predicted_word_tags, actual_word_tags)) if j[0] != j[1]]
    return accuracy, incorrect_tagged_cases


train_tags = list([pair[1] for pair in train_tagged_words])
train_words = list([pair[0] for pair in train_tagged_words])
test_tags = list([pair[1] for pair in test_tagged_words])
test_words = list([pair[0] for pair in test_tagged_words])

word_tag_counts: Dict[str, Tuple[int, int]] = {}


def emission_prob(word: str, tag: str, apply_smoothing: bool = False, epsilon: float = 1e-4, is_testing: bool = False) -> float:
    word_tag_key = f"{word}_{tag}"
    dict_query_res = word_tag_counts.get(word_tag_key, None)
    if dict_query_res:
        wc, tc = dict_query_res
    else:
        wc, tc = word_given_tag(word, tag)
        if not is_testing:
            word_tag_counts[word_tag_key] = (wc, tc)

    return wc/tc if tc > 0. else epsilon if apply_smoothing else np.nan


tag_tag_counts: Dict[str, Tuple[int, int]] = {}


# Boostrapping tag_tag_counts
train_atg_set = set(train_tags)

for i, t1 in enumerate(list(train_atg_set)):
    for j, t2 in enumerate(list(train_atg_set)):
        tag_tag_counts[f"{t2}_{t1}"] = t2_given_t1(t2, t1, train_tags)



def transition_prob(t2: str, t1: str, apply_smoothing: bool = False, epsilon: float = 1e-4, print_count: bool = False, is_testing: bool = False) -> float:
    tag_tag_key = f"{t2}_{t1}"
    dict_query_res = tag_tag_counts.get(tag_tag_key, None)
    if dict_query_res:
        count_t2_t1, count_t1 = dict_query_res
    else:
        count_t2_t1, count_t1 = t2_given_t1(t2, t1, train_tags)
        if not is_testing:
            tag_tag_counts[tag_tag_key] = (count_t2_t1, count_t1)
    if print_count:
        print(f"count_t2_t1: {count_t2_t1}, count_t1: {count_t1}")
    return count_t2_t1/count_t1 if count_t1 > 0. else epsilon if apply_smoothing else np.nan

# TODO: word_tag_counts,tag_tag_counts could be pre-populated
# def bootstrap():
#     train_words, train_tags

# %% [markdown]
# ### Build the vanilla Viterbi based POS tagger
# %%


def emission_prob_default(word: str, tag: str, is_testing: bool = False) -> float:
    return emission_prob(word, tag, is_testing)


def transition_prob_default(t2: str, t1: str, is_testing: bool = False) -> float:
    return transition_prob(t2, t1, is_testing)


def probability_calculator_default(emission_p: float, transition_p: float) -> float:
    return emission_p * transition_p


# %%
start = time.time()
default_train_viterbi_res = ViterbiPipe(train_words, train_tags, emission_prob_default, transition_prob_default, probability_calculator_default)
time_taken = time.time() - start
print(f"Time Taken: {time_taken:.2f}")

# %%
#  Calculate 5 most frequent TAGs
top_5_tags = pd.Series([p[1] for p in default_train_viterbi_res]).value_counts().sort_values(ascending=False).head()
display(top_5_tags)
top_tag = list(top_5_tags[: 1].index)[0]
top_tag

# %%
start = time.time()
default_test_viterbi_res = ViterbiPipe(test_words, test_tags, emission_prob_default, transition_prob_default, probability_calculator_default, is_testing=True)
time_taken = time.time() - start
print(f"Time Taken: {time_taken:.2f}")

# %%
len(default_test_viterbi_res), len(test_tagged_words)

# %%
accuracy_train_default, incorrect_tagged_cases_test_default = prepare_eval_metrics(default_train_viterbi_res, train_tagged_words)
display(accuracy_train_default)
accuracy_test_default, incorrect_tagged_cases_test_default = prepare_eval_metrics(default_test_viterbi_res, test_tagged_words)
display(accuracy_test_default)
# %% [markdown]
# ### Solve the problem of unknown words

# %%
# As we can see that ADJ is most frequent result which is stored in variable "top_tag", we'll assign ADJ tag to missing words

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

# %%
