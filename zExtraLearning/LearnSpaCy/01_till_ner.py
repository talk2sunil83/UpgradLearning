# %%
import spacy
import en_core_web_sm
import textacy
from spacy import displacy
# %%
with open("./data/alice.txt", "r", encoding='utf-8') as f:
    chapters = f.read().replace("\n\n", " ").replace("\n", " ").split("CHAPTER ")[1:]

print(len(chapters))
# %%

spacy.prefer_gpu()
nlp = en_core_web_sm.load()  # en_core_web_trf
print("model loaded")

# %%
chapter1 = chapters[0]

# %%
doc = nlp(chapter1)
# %%
sentences = list(doc.sents)
sentences[0]
# %%
sentence = sentences[1]
ents = list(sentence.ents)
print(ents)

# %%
# ents = list(doc.ents)
# print(ents)
# %%
'''
doc --> sents
(doc, sent) --> ents
ents --> ent[label, label_, text]
sent --> token[text, pos, pos_]

doc --> noun_chunk
import textacy
patterns = [{"POS":"ADV"}, {"POS":"VERB"}]
'''
patterns = [{"POS": "ADV"}, {"POS": "VERB"}]  # ADV --> VERB
# patterns = [[{"POS": "NOUN"}, {"POS": "VERB"}, {"POS": "ADV"}], [{"POS": "PRON"}, {"POS": "VERB"}, {"POS": "ADV"}]]
# patterns = [{"POS": "VERB"}]
verb_phrases = textacy.extract.matches(doc, patterns=patterns)
for verb_phrase in verb_phrases:
    print(verb_phrase)
# %%
sentence = sentences[8]
for word in sentence:
    if word.pos_ == "VERB":
        print(word, word.lemma_)

# %%
displacy.render(sentence, style='dep')
# with open("data_viz.html", "w") as f:
#     f.write(html)

# %%
displacy.render(sentence, style='ent')

# %%
displacy.render(doc, style='ent')

# %%
colors = {"PERSON": "#4E0000"}
options = {'ents': ["PERSON"], "colors": colors}
displacy.render(doc, style='ent', options=options)

# %%
colors = {"PERSON": "linear-gradient(90deg,#aa9cfc, #fc9ce7)"}
options = {'ents': ["PERSON"], "colors": colors}
displacy.render(doc, style='ent', options=options)

# %%
