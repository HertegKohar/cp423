"""
Herteg Kohar
"""
from Constants.constants import (
    TOPICS,
    DOCUMENTS_PATH,
    INVERTED_INDEX_PATH,
    MAPPING_PATH,
)

import os
import json
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from Soundex.soundex import compute_soundex


def update_inverted_index(topics):
    if os.path.exists(INVERTED_INDEX_PATH) and os.path.exists(MAPPING_PATH):
        print("Loading existing inverted index...")
        with open(INVERTED_INDEX_PATH, "r", encoding="utf-8") as f:
            inverted_index = json.load(f)
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        index = len(mapping)
    else:
        print("Creating new inverted index...")
        inverted_index = {}
        mapping = {}
        index = 0
    tokenizer = RegexpTokenizer(r"\w+")
    for topic in topics:
        for file in os.listdir(os.path.join(DOCUMENTS_PATH, topic)):
            with open(
                os.path.join(DOCUMENTS_PATH, topic, file), "r", encoding="utf-8"
            ) as f:
                text = f.read()
            hash_ = file.split(".")[0]
            if hash_ not in mapping:
                mapping[hash_] = f"H{index}"
                index += 1
                tokens = tokenizer.tokenize(text)
                counter = Counter(tokens)
                for (
                    token,
                    count,
                ) in counter.items():
                    if token not in inverted_index:
                        inverted_index[token] = {"soundex": compute_soundex(token)}
                        inverted_index[token]["occurences"] = []

                    inverted_index[token]["occurences"].append(
                        (mapping[hash_], count, topic)
                    )

    with open(INVERTED_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f)
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    print(f"Inverted index saved to {INVERTED_INDEX_PATH}")
    print(f"Mapping saved to {MAPPING_PATH}")


# Debugging
if __name__ == "__main__":
    update_inverted_index(TOPICS)
