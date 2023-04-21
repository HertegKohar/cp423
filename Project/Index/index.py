"""
Author: Herteg Kohar
"""
from Constants.constants import (
    TOPICS,
    DOCUMENTS_PATH,
    INVERTED_INDEX_PATH,
    MAPPING_PATH,
    INDENT,
    HASH_TO_URL_PATH,
    LOGGER_PATH,
)

import os
import json
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from Soundex.soundex import compute_soundex


def remake_hash_to_url():
    """Remakes the hash to url mapping file from the crawler log file."""
    hash_to_url = {}
    with open(LOGGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split(" ")
            hash_to_url[line[2]] = line[1]
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Make sure that all hashes are in the mapping
    for hash_ in hash_to_url:
        if hash_ not in mapping:
            hash_to_url.pop(hash_)

    with open(HASH_TO_URL_PATH, "w", encoding="utf-8") as f:
        json.dump(hash_to_url, f, indent=4)


def update_inverted_index(topics):
    """Updates the inverted index and mapping.

    Args:
        topics (list[str]): List of topics to be used for the inverted index. And directory names.
    """
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
        json.dump(mapping, f, indent=INDENT)
    remake_hash_to_url()
    print(f"Inverted index saved to {INVERTED_INDEX_PATH}")
    print(f"Mapping saved to {MAPPING_PATH}")
    print(f"Hash to url mapping saved to {HASH_TO_URL_PATH}")


# Debugging
if __name__ == "__main__":
    update_inverted_index(TOPICS)
