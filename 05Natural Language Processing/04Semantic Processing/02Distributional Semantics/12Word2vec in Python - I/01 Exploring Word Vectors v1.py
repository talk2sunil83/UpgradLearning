# %%
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.corpus import brown, movie_reviews
import random
from sklearn.decomposition import PCA
import nltk
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec


# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# %% [markdown]
# ### Creating our sentences to train the word vectors

# %%
TextCorpus = ["I like Upgrad",
              "Upgrad has a good ML program",
              "Upgrad has good faculty",
              "Rahim is that good faculty",
              "I like ML"
              ]


# %%
text_tokens = [sent.split() for sent in TextCorpus]
text_tokens[:2]

# %% [markdown]
# #### Training the word vectors

# %%
model = Word2Vec(text_tokens, min_count=1)


# %%
model.wv['ML']


# %%
len(model.wv['ML'])

# %% [markdown]
# Similarity between word vectors is measures using Cosine similarity

# %%
model.wv.most_similar("faculty", topn=5)

# %% [markdown]
# ### Training our word vectors on the text8 corpus
# Cleaned Wikipedia text, compressed to 100MB
# %% [markdown]
# #### Importing the raw text file

# %%
sentences = word2vec.Text8Corpus('../text8')


# %%
type(sentences)

# %% [markdown]
# #### Using all default parameters

# %%
model = Word2Vec(sentences)


# %%
model.wv.most_similar("happiness")


# %%
model.wv.most_similar("queen")

# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
# #### Semantic regularities captured in word embeddings

# %%
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)


# %%
model.wv.most_similar(positive=['woman', 'hero'], negative=['man'], topn=5)

# %% [markdown]
# ### Visualizing these word vectors

# %%
X = model.wv[model.wv.vocab]

# %% [markdown]
# ##### We'll use PCA to reduce and visualize in 2 dimensions

# %%


# %%
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# %% [markdown]
# Randomly select 100 words

# %%
random.seed(2)


# %%
ind = random.sample(range(0, len(X)), 100)
result_random = result[ind]
all_words = list(model.wv.vocab)
words = [all_words[i] for i in ind]


# %%
plt.figure(figsize=(12, 12))
plt.scatter(result_random[:, 0], result_random[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result_random[i, 0], result_random[i, 1]))
plt.show()

# %% [markdown]
# ## Effect of vector dimension
# %% [markdown]
# #### Reducing the length to 50

# %%
model = Word2Vec(sentences, size=50)


# %%
model.wv.most_similar("money")


# %%
model.wv.most_similar("queen", topn=5)


# %%
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)

# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
#
# %% [markdown]
# #### Inreasing vector length to 300

# %%
model = word2vec.Word2Vec(sentences, size=300)


# %%
model.wv.most_similar("money")


# %%
model.wv.most_similar("queen")


# %%
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)

# %% [markdown]
# ## Skip gram vs CBOW
# %% [markdown]
# Simply put, the CBOW model learns the embedding by predicting the current word based on its context. The skip-gram model learns by predicting the surrounding words given a current word.
# %% [markdown]
# ![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Word2Vec-Training-Models.png)
# %% [markdown]
# #### Using Skipgram method

# %%
model_sg = word2vec.Word2Vec(sentences, size=100, sg=1)


# %%
model_sg.wv.most_similar("money")


# %%
model_sg.wv.most_similar("queen")


# %%
model_sg.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)


# %%
model_sg.wv.most_similar("meandering")

# %% [markdown]
# #### Learning:
# In CBOW the vectors from the context words are averaged before predicting the center word. In skip-gram there is no averaging of embedding vectors. It seems like the model can learn better representations for the rare words when their vectors are not averaged with the other context words in the process of making the predictions.
# %% [markdown]
# ## Word vectors trained on different contexts
#  - We'll load different corpora, from different contexts and see how the embeddings vary
#  - The text8 corpus is wikipedia pages, while Brown corpus is from 15 different topics, and movie reviews are from IMDB

# %%


# %%
model_brown = Word2Vec(brown.sents(), sg=1)
model_movie = Word2Vec(movie_reviews.sents(), sg=1, window=5)


# %%
model_sg.wv.most_similar('money', topn=5)


# %%
model_brown.wv.most_similar('money', topn=5)


# %%
model_movie.wv.most_similar('money', topn=5)

# %% [markdown]
#
# I hope it’s pretty clear from the above examples that the semantic similarity of words can vary greatly depending on the textual context.
# %% [markdown]
# ## Using pre-trained word vectors
# %% [markdown]
# ### A quick note on Glove:
#    - Developed by Stanford by training on 6 Billion tokens
#    - Objective is slightly different
#    - End result very similar to Google's word2vec
# %% [markdown]
# https://nlp.stanford.edu/projects/glove/
# %% [markdown]
# - We'll use the 100D vectors for this example.
# - The trained vectors are available in a text file
# - The format is slightly different from that of word2vec, necessitating the use of a utility to format accordingly

# %%
glove_input_file = '../glove.6B.100d.txt'
word2vec_output_file = '../glove.6B.100d.w2vformat.txt'
glove2word2vec(glove_input_file, word2vec_output_file)


# %%
glove_model = KeyedVectors.load_word2vec_format("../glove.6B.100d.w2vformat.txt", binary=False)

# %% [markdown]
# #### Now you can use all the methods you used with word2vec models

# %%
glove_model.most_similar("king")


# %%
glove_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)


# %%
glove_model.most_similar(positive=['woman', 'hero'], negative=['man'], topn=5)

# %%
