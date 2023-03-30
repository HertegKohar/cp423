import os
import json
from nltk.tokenize import RegexpTokenizer
from collections import Counter

TOPICS = ["Astronomy", "Health", "Economy"]


def update_inverted_index(topics):
    if os.path.exists("inverted_index.json") and os.path.exists("mapping.json"):
        print("Loading existing inverted index...")
        with open("inverted_index.json", "r", encoding="utf-8") as f:
            inverted_index = json.load(f)
        with open("mapping.json", "r", encoding="utf-8") as f:
            mapping = json.load(f)
        index = len(mapping) - 1
    else:
        print("Creating new inverted index...")
        inverted_index = {}
        mapping = {}
        index = 0
    tokenizer = RegexpTokenizer(r"\w+")
    for topic in topics:
        for file in os.listdir(topic):
            with open(f"{topic}/{file}", "r", encoding="utf-8") as f:
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
                        inverted_index[token] = []

                    inverted_index[token].append((mapping[hash_], count))

    with open("inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(inverted_index, f)
    with open("mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    print(f"Inverted index saved to inverted_index.json")
    print(f"Mapping saved to mapping.json")


# Debugging
if __name__ == "__main__":
    update_inverted_index(TOPICS)
