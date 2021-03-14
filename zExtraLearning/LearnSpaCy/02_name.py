# %%
from typing import Any, Iterator, Generator
import spacy
import en_core_web_sm
import textacy
from spacy import displacy
from spacy.tokens.span import Span

spacy.prefer_gpu()
# %%
with open("./data/alice.txt", "r", encoding='utf-8') as f:
    chapters = f.read().replace("\n\n", " ").replace("\n", " ").split("CHAPTER ")[1:]

print(len(chapters))
# %%
chapter1 = chapters[0]


def find_sents(text: str) -> Iterator[Span]:  # Generator[Span, None, None]:  #
    nlp = en_core_web_sm.load()
    return nlp(text).sents


for sent in find_sents(chapter1):
    print(sent)
# %%
