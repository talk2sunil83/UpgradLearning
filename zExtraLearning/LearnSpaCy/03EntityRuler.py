# %%

from os import name
from typing import List
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import json
from functools import reduce
# %%


def load_data(file: str):
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


replace_texts = ["THE", "The", "the", "AND", "And", "and", "Sr.", "Jr.", ',', "(", ")"]
titles = ["Dr.", "Professor", "Mr.", "Mrs.", "Ms.", "Miss", "Aunt", "Uncle", "Mr. and Mrs."]
unwanted_chars = ["(", ")"]


# def rem_words_str(item, replace_texts): return " ".join([v for v in item.split() if v not in replace_texts])
def split_and_merge_names(a, b): return (a if isinstance(a, list) else a.split()) + b.split()


def get_single_names(names: List[str]) -> List[str]:
    res = []
    for name in names:
        for name_part in name.split():
            res.append(name_part)
    return res


def rem_str(item, unwanted_chars):
    for c in unwanted_chars:
        item = item.replace(c, "")
    return item


def generate_better_characters(file: str):
    data = load_data(file)
    clean_data = [name.strip() for name in [rem_str(item, replace_texts) for item in data]]
    print(len(clean_data))
    # single_names = get_single_names(clean_data)
    single_names = reduce(split_and_merge_names, clean_data)
    print(len(single_names))
    new_characters = [i for i in single_names]
    [[new_characters.append(f"{t} {c}") for t in titles] for c in single_names]
    # [[new_characters.append(f"{t} {c}") for c in single_names] for t in titles]
    print(len(new_characters))
    new_characters = sorted(list(set(new_characters)))
    print(len(new_characters))
    return new_characters


char_file_path = "./data/hp_characters.json"
# hp_chars = load_data(char_file_path)
# print(hp_chars)
all_names = generate_better_characters(char_file_path)
training_data = [{"label": "PERSON", "pattern": name} for name in all_names]
training_data
# %%
