# %%
import nltk
from nltk.parse import ShiftReduceParser
# %%
grammar = nltk.CFG.fromstring('''
    S ->  Wh-NP VP
    NP -> N NP1
    NP -> Det N
    NP -> N PP
    NP1 -> ε
    VP -> V VP1
    VP -> V PP
    VP1 ->ε
    PP -> P NP
    N -> 'airlines' | 'Bangalore' | 'Bangkok'
    Wh -> 'What'
    V -> 'fly'
    P -> 'from'| 'to'
    ''')
sent_parts = list("What airlines fly from Bangalore to Bangkok".split())

# %%
srp = ShiftReduceParser(grammar)

for t in srp.parse(sent_parts):
    print(t)
# %%
