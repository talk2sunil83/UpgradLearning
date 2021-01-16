# %%
from nltk import parse, CFG
# %%

grammar = CFG.fromstring("""
S -> NP VP
NP -> DT N
VP -> V | V PP
PP -> P NP
DT -> 'the'
N -> 'child' | 'kitchen'
V -> 'ran'
P -> 'to'
    """)

str_val = "the child ran to the kitchen"
srp = parse.ShiftReduceParser(grammar, trace=2)

for res in srp.parse(list(str_val.lower().split())):
    if res is None:
        print("No Res")
    print(res)

# %%
# import nltk
# nltk.app.srparser()
# %%


grammar = CFG.fromstring("""
S -> NP VP
PP -> P NP
VP -> V NP| VP PP | V
NP -> DT N  | N | NP PP
P ->  'with'
V -> 'caught'
N -> 'man'| 'fish' | 'net'
DT -> 'the' | 'a'
    """)

str_val = list("the man caught fish with a net".lower().split())
srp = parse.ShiftReduceParser(grammar, trace=2)

for t in srp.parse(str_val):
    print(t)
# %%
grammar = CFG.fromstring("""
S -> NP VP
PP -> P NP
VP -> V NP| V PP | V
NP -> DT N  | N | NP PP
P ->  'over'
V -> 'jumped'
N -> 'bear'| 'table'
DT -> 'the'
    """)

sr = parse.ShiftReduceParser(grammar, trace=3)
str_val = list("the bear jumped over the table".split())

for t in sr.parse(str_val):
    print(t)

# %%
