# %%
import nltk.corpus
# %%
print(str(nltk.corpus.brown).replace('\\\\', '/'))

# %%
print(str(nltk.corpus.treebank).replace('\\\\', '/'))
# %%
print(str(nltk.corpus.names).replace('\\\\', '/'))
# %%
print(str(nltk.corpus.inaugural).replace('\\\\', '/'))
# %%
nltk.corpus.treebank.fileids()
# %%
