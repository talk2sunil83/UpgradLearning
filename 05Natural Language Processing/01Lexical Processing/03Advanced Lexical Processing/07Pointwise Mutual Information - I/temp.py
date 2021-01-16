# %%
from functools import reduce
from nltk.tokenize import sent_tokenize
from math import log10
# %%
text = "The Nobel Prize is a set of five annual international awards bestowed in several categories by Swedish and Norwegian institutions in recognition of academic, cultural, or scientific advances. In the 19th century, the Nobel family who were known for their innovations to the oil industry in Azerbaijan was the leading representative of foreign capital in Baku. The Nobel Prize was funded by personal fortune of Alfred Nobel. The Board of the Nobel Foundation decided that after this addition, it would allow no further new prize."

# %%
docs = sent_tokenize(text.lower())
docs
# %%
#  log ( P(New Delhi)/P(New)P(Delhi) )


def get_word_prob(word):
    return len([sent for sent in docs if word.lower() in sent]) / len(docs)


# %%
get_word_prob('nobel')

# %%
get_word_prob('prize')


# %%
get_word_prob('Nobel Prize')

# %%


def get_pmi(term: str, round_till: int = 3) -> float:
    return round(log10(get_word_prob(term) / reduce((lambda p1, p2: p1*p2), [get_word_prob(w) for w in term.split()])), round_till)


# %%
print(get_pmi("Nobel Prize"))

# %%
log10(0.5/0.75)
# %%
