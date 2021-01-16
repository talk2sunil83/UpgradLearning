# %%
# Importing libraries
from collections import Counter
import nltk
import numpy as np
import pandas as pd
import pprint
import time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import math


# %%
# reading the Treebank tagged sentences
wsj = list(nltk.corpus.treebank.tagged_sents())


# %%
# samples: Each sentence is a list of (word, pos) tuples
wsj[:3]

# %% [markdown]
# In the list mentioned above, each element of the list is a sentence. Also, note that each sentence ends with a full stop '.' whose POS tag is also a '.'. Thus, the POS tag '.' demarcates the end of a sentence.
#
# Also, we do not need the corpus to be segmented into sentences, but can rather use a list of (word, tag) tuples. Let's convert the list into a (word, tag) tuple.

# %%
# converting the list of sents to a list of (word, pos tag) tuples
tagged_words = [tup for sent in wsj for tup in sent]
print(len(tagged_words))
tagged_words[:10]

# %% [markdown]
# We now have a list of about 100676 (word, tag) tuples. Let's now do some exploratory analyses.
# %% [markdown]
# ### 2. Exploratory Analysis
#
# Let's now conduct some basic exploratory analysis to understand the tagged corpus. To start with, let's ask some simple questions:
# 1. How many unique tags are there in the corpus?
# 2. Which is the most frequent tag in the corpus?
# 3. Which tag is most commonly assigned to the following words:
#     - "bank"
#     - "executive"
#

# %%
# question 1: Find the number of unique POS tags in the corpus
# you can use the set() function on the list of tags to get a unique set of tags,
# and compute its length
tags = [pair[1] for pair in tagged_words]
unique_tags = set(tags)
len(unique_tags)


# %%
# question 2: Which is the most frequent tag in the corpus
# to count the frequency of elements in a list, the Counter() class from collections
# module is very useful, as shown below

tag_counts = Counter(tags)
tag_counts


# %%
# the most common tags can be seen using the most_common() method of Counter
tag_counts.most_common(5)

# %% [markdown]
# Thus, NN is the most common tag followed by IN, NNP, DT, -NONE- etc. You can read the exhaustive list of tags using the NLTK documentation as shown below.

# %%
# list of POS tags in NLTK
nltk.help.upenn_tagset()


# %%


# %%
# question 3: Which tag is most commonly assigned to the word w.
bank = [pair for pair in tagged_words if pair[0].lower() == 'bank']
bank


# %%
# question 3: Which tag is most commonly assigned to the word w.
executive = [pair for pair in tagged_words if pair[0].lower() == 'executive']
executive

# %% [markdown]
# ### 2. Exploratory Analysis Contd.
#
# Until now, we were looking at the frequency of tags assigned to particular words, which is the basic idea used by lexicon or unigram taggers. Let's now try observing some rules which can potentially be used for POS tagging.
#
# To start with, let's see if the following questions reveal something useful:
#
# 4. What fraction of words with the tag 'VBD' (verb, past tense) end with the letters 'ed'
# 5. What fraction of words with the tag 'VBG' (verb, present participle/gerund) end with the letters 'ing'

# %%
# 4. how many words with the tag 'VBD' (verb, past tense) end with 'ed'
past_tense_verbs = [pair for pair in tagged_words if pair[1] == 'VBD']
ed_verbs = [pair for pair in past_tense_verbs if pair[0].endswith('ed')]
print(len(ed_verbs) / len(past_tense_verbs))
ed_verbs[:20]


# %%
# 5. how many words with the tag 'VBG' end with 'ing'
participle_verbs = [pair for pair in tagged_words if pair[1] == 'VBG']
ing_verbs = [pair for pair in participle_verbs if pair[0].endswith('ing')]
print(len(ing_verbs) / len(participle_verbs))
ing_verbs[:20]

# %% [markdown]
# ## 2. Exploratory Analysis Continued
#
# Let's now try observing some tag patterns using the fact the some tags are more likely to apper after certain other tags. For e.g. most nouns NN are usually followed by determiners DT ("The/DT constitution/NN"), adjectives JJ usually precede a noun NN (" A large/JJ building/NN"), etc.
#
# Try answering the following questions:
# 1. What fraction of adjectives JJ are followed by a noun NN?
# 2. What fraction of determiners DT are followed by a noun NN?
# 3. What fraction of modals MD are followed by a verb VB?

# %%
# question: what fraction of adjectives JJ are followed by a noun NN

# create a list of all tags (without the words)
tags = [pair[1] for pair in tagged_words]

# create a list of JJ tags
jj_tags = [t for t in tags if t == 'JJ']

# create a list of (JJ, NN) tags
jj_nn_tags = [(t, tags[index+1]) for index, t in enumerate(tags)
              if t == 'JJ' and tags[index+1] == 'NN']

print(len(jj_tags))
print(len(jj_nn_tags))
print(len(jj_nn_tags) / len(jj_tags))


# %%
# question: what fraction of determiners DT are followed by a noun NN
dt_tags = [t for t in tags if t == 'DT']
dt_nn_tags = [(t, tags[index+1]) for index, t in enumerate(tags)
              if t == 'DT' and tags[index+1] == 'NN']

print(len(dt_tags))
print(len(dt_nn_tags))
print(len(dt_nn_tags) / len(dt_tags))


# %%
# question: what fraction of modals MD are followed by a verb VB?
md_tags = [t for t in tags if t == 'MD']
md_vb_tags = [(t, tags[index+1]) for index, t in enumerate(tags)
              if t == 'MD' and tags[index+1] == 'VB']

print(len(md_tags))
print(len(md_vb_tags))
print(len(md_vb_tags) / len(md_tags))

# %% [markdown]
# Thus, we see that the probability of certain tags appearing after certain other tags is quite high, and this fact can be used to build quite efficient POS tagging algorithms.
# %% [markdown]
# ## 3. Lexicon and Rule-Based Models for POS Tagging
#
# Let's now see lexicon and rule-based models for POS tagging. We'll first split the corpus into training and test sets and then use built-in NLTK taggers.
#
# ### 3.1 Splitting into Train and Test Sets

# %%
# splitting into train and test
random.seed(1234)
train_set, test_set = train_test_split(wsj, test_size=0.3)

print(len(train_set))
print(len(test_set))
print(train_set[:2])

# %% [markdown]
# ### 3.2 Lexicon (Unigram) Tagger
#
# Let's now try training a lexicon (or a unigram) tagger which assigns the most commonly assigned tag to a word.
#
# In NLTK, the `UnigramTagger()`  can be used to train such a model.

# %%
# Lexicon (or unigram tagger)
unigram_tagger = nltk.UnigramTagger(train_set)
unigram_tagger.evaluate(test_set)

# %% [markdown]
# Even a simple unigram tagger seems to perform fairly well.
# %% [markdown]
# ### 3.3. Rule-Based (Regular Expression) Tagger
#
# Now let's build a rule-based, or regular expression based tagger. In NLTK, the `RegexpTagger()` can be provided with handwritten regular expression patterns, as shown below.
#
# In the example below, we specify regexes for gerunds and past tense verbs (as seen above), 3rd singular present verb (creates, moves, makes etc.), modal verbs MD (should, would, could), possesive nouns (partner's, bank's etc.), plural nouns (banks, institutions), cardinal numbers CD and finally, if none of the above rules are applicable to a word, we tag the most frequent tag NN.

# %%
# specify patterns for tagging
# example from the NLTK book
patterns = [
    (r'.*ing$', 'VBG'),              # gerund
    (r'.*ed$', 'VBD'),               # past tense
    (r'.*es$', 'VBZ'),               # 3rd singular present
    (r'.*ould$', 'MD'),              # modals
    (r'.*\'s$', 'NN$'),              # possessive nouns
    (r'.*s$', 'NNS'),                # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')                    # nouns
]


# %%
regexp_tagger = nltk.RegexpTagger(patterns)
# help(regexp_tagger)


# %%
regexp_tagger.evaluate(test_set)

# %% [markdown]
# ### 3.4 Combining Taggers
#
# Let's now try combining the taggers created above. We saw that the rule-based tagger by itself is quite ineffective since we've only written a handful of rules. However, if we could combine the lexicon and the rule-based tagger, we can potentially create a tagger much better than any of the individual ones.
#
# NLTK provides a convenient way to combine taggers using the 'backup' argument. In the following code, we create a regex tagger which is used as a backup tagger to the lexicon tagger, i.e. when the tagger is not able to tag using the lexicon (in case of a new word not in the vocabulary), it uses the rule-based tagger.
#
# Also, note that the rule-based tagger itself is backed up by the tag 'NN'.
#

# %%
# rule based tagger
rule_based_tagger = nltk.RegexpTagger(patterns)

# lexicon backed up by the rule-based tagger
lexicon_tagger = nltk.UnigramTagger(train_set, backoff=rule_based_tagger)

lexicon_tagger.evaluate(test_set)
