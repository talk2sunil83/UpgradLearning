# Copyright 2021 dev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
from nltk.parse import RecursiveDescentParser
import nltk
from nltk import tree
# %%

groucho_grammar = nltk.CFG.fromstring(
    '''
        S -> NP VP
        NP -> Det N| Det Adj N
        VP -> V NP |V
        
        Det -> 'the'
        V -> 'booked'
        N -> 'man' | 'flight'
        Adj -> 'old'
    ''')
# %%
sent = list('The old man booked the flight'.lower().split())
parser = nltk.ChartParser(groucho_grammar)
for tree in parser.parse(sent):
    print(tree)
# %%
grammar = nltk.CFG.fromstring(
    '''
        S -> NP VP
        NP ->  NP PP | DT NN
        VP -> V JJ
        PP -> IN NP
        
        DT -> 'The' | 'the'
        NN -> 'work' | 'analyst'
        V -> 'is'
        JJ -> 'good'
        IN -> 'of'
    ''')

# %%
rdpar = RecursiveDescentParser(grammar)
# %%
sent = list('The work of the analyst is good'.lower().split())
# for tree in rdpar.parse(sent):
#     print(tree)
# %%
nltk.app.rdparser()
# %%


# %%
grammar = nltk.CFG.fromstring(
    '''
        S -> NP VP
        NP ->  NP PP | DT NN
        VP -> V JJ
        PP -> IN NP
        DT -> 'The' | 'the'
        NN -> 'work' | ‘analyst’

        V -> 'is'

        JJ -> 'good'

        IN -> 'of'
    ''')
rdpar = RecursiveDescentParser(grammar)

sent = list('The work of the analyst is good'.lower().split())
for tree in rdpar.parse(sent):
    print(tree)

# %%
