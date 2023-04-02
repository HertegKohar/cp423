"""
Authors:
    Kelvin Kellner
    Herteg Kohar
Resource: https://norvig.com/spell-correct.html
"""
import argparse
from collections import Counter
import json
from nltk.tokenize import RegexpTokenizer
import os
import re
import string

FILENAME = "wikipedia.token"
DATA_FOLDER = "data_wikipedia"


def P(word):
    """
    Probability of 'word' occurring in the corpus
    -----
    Args:
        word (str): Word to compute probability for
    Returns:
        probability (float): Probability of 'word' occurring in the corpus
                             (proportion of corpus tokens that are 'word')
    """
    N_TOKENS = sum(WORD_FREQ_VECTOR.values())
    return 0 if N_TOKENS == 0 else WORD_FREQ_VECTOR[word] / N_TOKENS 


def correction(word): 
    """
    Return the highest probability spelling correction for 'word'
    -----
    Args:
        word (str): Word to correct
    Returns:
        correction (str): Highest probability spelling correction for 'word'    
    """
    return max(candidates(word), key=P)


def candidates(word): 
    """
    Return set of best spelling correction candidates for 'word'
    -----
    Args:
        word (str): Word to generate candidates for
    Returns:
        candidates (set[str]): Set of best candidates for 'word'
    """
    # candidates are generated using the following rules:
    # 1. if 'word' is a 'known' word, return it as there is no need to correct it
    # 2. otherise return the set of known words that are only 1 edit distance away from 'word'
    # 3. if there are none, then return the set of known words that are only 2 edit distance away
    # 4. if there are still none, then return the original 'word' as there is no correction to suggest
    return (known_words([word]) or known_words(edit_distance_of_1(word)) or known_words(edit_distance_of_2(word)) or [word])


def known_words(words):
    """
    The subset of 'words' that appear in the corpus, and thus, in 'WORD_FREQ_VECTOR'
    -----
    Args:
        words (list[str]): List of words to check for in the corpus
    Returns:
        known (set[str]): Subset of 'words' that are present in the corpus
    """
    return set(w for w in words if w in WORD_FREQ_VECTOR)


def edit_distance_of_1(word):
    """
    Generate all possible edits that are 1 edit distance away from 'word'
    -----
    Args:
        word (str): Word to generate edits for
    Returns:
        ed1 (list[str]): List of all possible edits that are 1 edit distance away from 'word'
    """
    # perform all edits of all types on 'word' and create a set of the results
    alphabet = string.ascii_lowercase
    pieces = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    insertions = [left + letter + right for left, right in pieces for letter in alphabet]
    deletions = [left + right[1:] for left, right in pieces if right]
    replacements = [left + letter + right[1:] for left, right in pieces if right for letter in alphabet]
    transpositions = [left + right[1] + right[0] + right[2:] for left, right in pieces if len(right) > 1]
    return set(insertions + deletions + replacements + transpositions)


def edit_distance_of_2(word): 
    """
    Generate all possible edits that are 2 edit distance away from 'word'
    -----
    Args:
        word (str): Word to generate edits for
    Returns:
        ed2 (list[str]): List of all possible edits that are 2 edit distance away from 'word'
    """
    # compute all combinations of 1 more edit on all edits of 1 distance from 'word'
    return (ed2 for ed1 in edit_distance_of_1(word) for ed2 in edit_distance_of_1(ed1))


def tokenize_corpus(path):
    """
    Load articles from the given path for wikipedia data
    Tokenize the text of each article and save to a file
    -----
    Args:
        path (str): Path to directory containing wikipedia data
    Returns:
        tokens (list[str]): List of tokens from the corpus
    """
    all_articles = []
    corpus = ""
    for file in os.listdir(path):
        with open(f"{path}/{file}", "r", encoding="utf-8") as f:
            articles = json.load(f)
            print(f"Number of articles in {file}: {len(articles)}")
            for i, article in enumerate(articles):
                corpus += article["text"]
                if i % 1000 == 0:
                    print(f"Processed {i} articles")
            all_articles.extend(articles)
        # Break if here to prevent memory error
        break
    tokens = RegexpTokenizer(r"\w+").tokenize(corpus)
    with open(FILENAME, "a", encoding="utf-8") as f:
        f.write(" ".join(tokens))
        f.write("\n")
    print(f"Processed {len(tokens)} tokens and saved to {FILENAME}")
    return tokens


if __name__ == "__main__":
    # word_freq_vector is the term frequency vector for all words in the inverted index
    # to make word_freq_vecto: word is the key, frequency is the sum of all inverted_index[word]['occurences'][1]
    word_freq_vector = []
    with open('inverted_index.json', 'r') as f:
        inverted_index = json.load(f)
        for word in inverted_index:
            word_freq_vector[word] = sum([occ[1] for occ in inverted_index[word]['occurences']])
            print(word, word_freq_vector[word])
    exit(0)

    



    # 1. Process and load the corpus
    exists = os.path.exists(FILENAME)
    overwrite_file = args.overwrite
    file_is_empty = os.stat(FILENAME).st_size == 0
    if not exists or overwrite_file or file_is_empty:
        print(f"\nTokenized corpus text file is empty or not found.\nProcessing and saving corpus to '{FILENAME}'...")
        with open(FILENAME, "w", encoding="utf-8") as f:
            pass
        tokens = tokenize_corpus(DATA_FOLDER)
        print()
    # term frequency vector for all words in the corpus
    # [(word: string, count: int), ...]
    with open(FILENAME, encoding="utf-8") as f:
        WORD_FREQ_VECTOR = Counter(tokenize(f.read()))

    # 2. Output the corrections or probabilities of the given words as needed
    mode = "correct" if args.correct is not None else "proba"
    words = args.correct if args.correct is not None else args.proba
    if mode == "correct":
        for word in words:
            print(f"{word} -> {correction(word)}")
    if mode == "proba":
        for word in words:
            print(f"P({word}) = {P(word)}")
    
    # All done :)
