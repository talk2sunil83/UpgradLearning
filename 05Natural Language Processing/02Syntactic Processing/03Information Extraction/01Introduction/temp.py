# %%
from nltk.corpus import conll2000
from nltk import conlltags2tree, tree2conlltags
# %%
labels = [('Donald', 'NNP', 'B-PERSON'),
          ('Trump', 'NNP', 'I-PERSON'),
          ('is', 'VBP', 'O'),
          ('the', 'DT', 'O'),
          ('President', 'NNP', 'B-ROLE'),
          ('of', 'IN', 'O'),
          ('USA', 'NNP', 'B-COUNTRY')]
# %%
tree = conlltags2tree(labels)
print(tree)
# %%


labels = [('show', 'VB', 'O'),
          ('me', 'PRP', 'O'),
          ('the', 'DT', 'O'),
          ('least', 'JJS', 'B-cost_relative'),
          ('expensive', 'JJ', 'B-cost'),
          ('single', 'NN', 'B-round_trip'),
          ('trips', 'NNS', 'I-round_trip'),
          ('from', 'IN', 'O'),
          ('baltimore', 'NN', 'B-fromloc.city_name'),
          ('to', 'TO', 'O'),
          ('Dallas', 'VB', 'B-toloc.city_name'),
          ('Fort', 'NN', 'I-toloc.city_name'),
          ('Worth', 'NN', 'I-toloc.city_name')]
tree = conlltags2tree(labels)
print(tree)
# %%
