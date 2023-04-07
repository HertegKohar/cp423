"""
Herteg Kohar
"""
import os
import re
import json
from nltk.tokenize import RegexpTokenizer
from collections import Counter

TOPICS = ["Astronomy", "Health", "Economy"]


def compute_soundex(term):
    """
    Compute the soundex code for a given term
    -----
    Args:
        term (str): Term to compute soundex code for
    Returns:
        soundex (str): Soundex code for 'term'
    """
    soundex = ""
    soundex += term[0].upper()
    for char in term[1:].lower():
        if char in "bfpv":
            soundex += "1"
        elif char in "cgjkqsxz":
            soundex += "2"
        elif char in "dt":
            soundex += "3"
        elif char in "l":
            soundex += "4"
        elif char in "mn":
            soundex += "5"
        elif char in "r":
            soundex += "6"
        else:
            soundex += "0"
    soundex = re.sub(r"(.)\1+", r"\1", soundex)
    soundex = re.sub(r"0", "", soundex)
    soundex = soundex[:4].ljust(4, "0")
    return soundex


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
                        inverted_index[token] = {"soundex": compute_soundex(token)}
                        inverted_index[token]["occurences"] = []

                    inverted_index[token]["occurences"].append(
                        (mapping[hash_], count, topic)
                    )

    with open("inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(inverted_index, f)
    with open("mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    print(f"Inverted index saved to inverted_index.json")
    print(f"Mapping saved to mapping.json")


# Debugging
if __name__ == "__main__":
    update_inverted_index(TOPICS)
