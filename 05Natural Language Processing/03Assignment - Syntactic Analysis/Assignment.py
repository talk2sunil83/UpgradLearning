# ## 1. Data Preparation

# %%
# Importing libraries
from typing import Sequence, Tuple
import nltk
import numpy as np
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


# %%
# reading the Treebank tagged sentences
wsj = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))


# %%
# Splitting into train and validation
random.seed(1234)
train_set, test_set = train_test_split(wsj, test_size=0.05)

# %%
# Getting list of tagged words
train_tagged_words = [tup for sent in train_set for tup in sent]
len(train_tagged_words)


# %%
# Getting list of tagged words
validation = [tup for sent in test_set for tup in sent]


# %%
# tokens
tokens = [pair[0] for pair in train_tagged_words]


# %%
# vocabulary
V = set(tokens)

# %%
# number of tags
T = set([pair[1] for pair in train_tagged_words])

# %% [markdown]
# ### Emission Probabilities

# %%
# computing P(w/t) and storing in T x V matrix
t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))


# %%
# compute word given tag: Emission Probability
def word_given_tag(word:str, tag:str, train_bag:Sequence[Tuple[str, str]]=train_tagged_words)-> Tuple[int, int]:
    """Calculate the count of occurrence of word for given tag and total occurrence of tag

    Args:
        word (str): word
        tag (str): tag
        train_bag (Sequence[Tuple[str, str]], optional): tagged words of training set. Defaults to train_tagged_words.

    Returns:
        Tuple[int, int]: (count of occurrence of word for given tag, total occurrence of tag)
    """    
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    count_w_given_tag = len(w_given_tag_list)

    return (count_w_given_tag, count_tag)

# %% [markdown]
# ### Transition Probabilities

# %%
# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability


def t2_given_t1(t2:str, t1:str, train_bag:Sequence[Tuple[str, str]]=train_tagged_words)-> Tuple[int, int]:
    """ Calculate the count of occurrence of T1-T2 and total occurrence of T1

    Args:
        t2 (str): Second tag
        t1 (str): First tag
        train_bag (Sequence[Tuple[str, str]], optional): tagged words of training set. Defaults to train_tagged_words.

    Returns:
        Tuple[int, int]: (count of occurrence of T1-T2, total occurrence of T1)
    """    
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t == t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index] == t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


# %%
# creating t x t transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)

tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)):
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]

tags_matrix


# %%
# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns=list(T), index=list(T))
tags_df


# %% [markdown]
# ## 2. Build the vanilla Viterbi based POS tagger


# %%
# Viterbi Heuristic
def Viterbi(words:Sequence[str], train_bag:Sequence[Tuple[str, str]]=train_tagged_words)-> Sequence[Tuple[str, str]]:
    """Vennila Implementation of Viterbi Heuristic


    Args:
        words (Sequence[str]): list of words
        train_bag (Sequence[Tuple[str, str]], optional): tagged words of training set. Defaults to train_tagged_words.

    Returns:
        Sequence[Tuple[str, float]]: words with tag
    """
    state = []
    tags_set = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in tags_set:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p_parts = word_given_tag(word, tag)
            emission_p = emission_p_parts[0]/emission_p_parts[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)

        p_max = max(p)
        # getting state for which probability is maximum
        state_max = tags_set[p.index(p_max)]
        state.append(state_max)
    return list(zip(words, state))

# %% [markdown]
# #### Evaluating on Validation Set

# %%
# list of tagged words
validation_run_base = [tup for sent in test_set for tup in sent]

# list of untagged words
validation_tagged_words = [tup[0] for sent in test_set for tup in sent]
print(len(test_set))
print(len(validation_tagged_words))


# %%
# tagging the test sentences
start = time.time()
validation_vit_default = Viterbi(validation_tagged_words)
end = time.time()
difference = end-start
print(f"Time taken in seconds: {difference:.2f}")
print(len(validation_tagged_words))



# %%
# accuracy
check = [i for i, j in zip(validation_vit_default, validation_run_base) if i == j]
accuracy = len(check)/len(validation_vit_default)

print(accuracy)
perf_metric = [("Vennila", accuracy)]

# %%
incorrect_tagged_cases = [[validation_run_base[i-1], j] for i, j in enumerate(zip(validation_vit_default, validation_run_base)) if j[0] != j[1]]
incorrect_tagged_cases

# %%
#  Read test data file
file_text = ""
with open("Test_sentences.txt", "rt") as f:
    file_text = "".join([line.strip() for line in f.readlines()])

# %%
# Testing
test_words = word_tokenize(file_text)
tagged_seq = Viterbi(test_words)

# %%
print(tagged_seq)
print(difference)

# %%
# Calculate most common tag in train data set
most_common_tags = pd.Series([p[1] for p in train_tagged_words]).value_counts().sort_values(ascending=False).head()
most_common_tag = list(most_common_tags.index)[0]
print(most_common_tag)
print(most_common_tags)
# %% [markdown]
# ## 3. Solve the problem of unknown words
# %% [markdown]
# ### Viterbi Modification -2: Implementation of Viterbi Heuristic with using only transition probability if word is missing from training corpus


# %%


def Viterbi_Most_Common_Tag(tagged_seq:Sequence[str],most_common_tag:str, train_bag:Sequence[Tuple[str, str]]=train_tagged_words)-> Sequence[Tuple[str, str]]:
    """Implementation of Viterbi Heuristic with most occuring TAG for missing words


    Args:
        tagged_seq (Sequence[str]): list of words
        most_common_tag (str): most common tag
        train_bag (Sequence[Tuple[str, str]], optional): tagged words of training set. Defaults to train_tagged_words.

    Returns:
        Sequence[Tuple[str, float]]: words with tag
    """
    V = list(set([pair[0] for pair in train_bag]))
    words = [pair[0] for pair in tagged_seq]
    tags = [pair[1] for pair in tagged_seq]

    for word_index, word in enumerate(words):
        if word not in V:
            tags[word_index] = most_common_tag

    return list(zip(words, tags))


# %%
validation_most_common_tag = Viterbi_Most_Common_Tag(validation_vit_default, most_common_tag)

# %%
# accuracy
check_most_common_tag = [i for i, j in zip(validation_most_common_tag, validation_run_base) if i == j]
accuracy = len(check_most_common_tag)/len(validation_most_common_tag)
print(accuracy)
perf_metric.append(("With Most Common Tag", accuracy))

# %%
tagged_most_common_test = Viterbi_Most_Common_Tag(tagged_seq, most_common_tag)
tagged_most_common_test


# %% [markdown]
# ### Viterbi Modification -2: Implementation of Viterbi Heuristic with using only transition probability if word is missing from training corpus

# %%
def Viterbi_Transition(words:Sequence[str], train_bag:Sequence[Tuple[str, str]]=train_tagged_words)-> Sequence[Tuple[str, str]]:
    """Implementation of Viterbi Heuristic with using only transition probability if word is missing from training corpus


    Args:
        tagged_seq (Sequence[str]): list of words
        most_common_tag (str): most common tag
        train_bag (Sequence[Tuple[str, str]], optional): tagged words of training set. Defaults to train_tagged_words.

    Returns:
        Sequence[Tuple[str, float]]: words with tag
    """
    state = []
    all_tags = list(set([pair[1] for pair in train_bag]))

    for word_idx, word in enumerate(words):
        # initialise list of probability column for a given observation
        p = []
        for tag in all_tags:
            if word_idx == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p_parts = word_given_tag(word, tag)
            emission_p = emission_p_parts[0]/emission_p_parts[1]

            if word in V:
                state_probability = transition_p * emission_p
            else:
                state_probability = transition_p

            p.append(state_probability)

        p_max = max(p)
        # getting state for which probability is maximum
        state_max = all_tags[p.index(p_max)]
        state.append(state_max)
    return list(zip(words, state))


# %%
# tagging the test sentences
validation_transition = Viterbi_Transition(validation_tagged_words)

# %%
# accuracy
check = [i for i, j in zip(validation_transition, validation_run_base) if i == j]
accuracy = len(check)/len(validation_transition)
print(accuracy)
perf_metric.append(("With Transition Prob", accuracy))


# %%
tagged_transition_test = Viterbi_Transition(test_words)
tagged_transition_test

# %%
pd.DataFrame(perf_metric, columns=["Algorithm", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
# %%
