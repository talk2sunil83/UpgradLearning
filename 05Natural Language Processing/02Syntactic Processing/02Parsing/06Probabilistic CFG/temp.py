# %%
from nltk.parse import pchart
import nltk
# %%
pcfg_grammar = nltk.PCFG.fromstring("""
S -> NP VP [1.0]
PP -> P NP [1.0]
VP -> V NP [0.3]| VP PP [0.4] | V [0.3]
NP -> DT N [0.4] | N [0.35] | NP PP [0.25]
P -> 'on' [0.4] | 'with' [0.6]
V -> 'sleeps' [1.0]
N -> 'child' [0.5]| 'bed' [0.3] | 'cushion' [0.2]
DT -> 'the' [1.0]
    """)
# %%
str = "the child sleeps on the bed with cushion"
# %%

parser = pchart.InsideChartParser(pcfg_grammar)

# print all possible trees, showing probability of each parse
for t in parser.parse(str.split()):
    print(t)

# %%
pcfg_grammar = nltk.PCFG.fromstring("""
S -> NP VP [1.0]
NP -> Det N [0.7] | Det N PP [0.3]
VP -> V [0.4] | V NP [0.4] |V NP PP [0.2]
PP -> P NP [1.0]
Det -> 'a' [0.4]| 'the' [0.6]
N -> 'man' [0.5] | 'dog'  [0.3]| 'park' [0.1]| 'telescope' [0.1]
V -> 'saw' [1.0]
P -> 'with' [0.6]| 'in' [0.4]                                
    """)

str = "the man saw a dog in the park with a telescope"
parser = pchart.InsideChartParser(pcfg_grammar)

# print all possible trees, showing probability of each parse
for t in parser.parse(str.split()):
    print(t)

# %%
grammar = nltk.PCFG.fromstring(
    '''
        S -> NP VP [1.0]
        PP -> P NP [1.0]
        VP -> V NP [0.3]| VP PP [0.4] | V [0.3]
        NP -> DT N [0.4] | N [0.2] | NP PP [0.25] | N V [0.15]
        P -> 'until' [0.4] | 'with' [0.6]
        V -> 'chased' [0.4] | 'stumbled' [0.2] | 'fell' [0.4]
        N -> 'lion' [0.5]| 'deer' [0.4] | 'it' [0.1]
        DT -> 'the' [1.0]
    ''')
rdpar = pchart.InsideChartParser(grammar)

sent = list('the lion chased the deer until it fell'.split())
for tree in rdpar.parse(sent):
    print(tree)
# %%
