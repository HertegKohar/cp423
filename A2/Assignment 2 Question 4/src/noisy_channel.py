"""
Authors:
    Kelvin Kellner
    Herteg Kohar
Resources:
    https://norvig.com/spell-correct.html
    https://medium.com/mlearning-ai/build-spell-checking-models-for-any-language-in-python-aa4489df0a5f
"""
import argparse
from collections import Counter
import json
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, PlaintextCorpusReader
import os
import re
import sys

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('wikipedia.token.stop', encoding="utf-8").read())) #TODO: confirm file name

def load_articles(path):
    """Load articles from the given path to directory for wikipedia data
    Args:
        path (str): Path to directory containing wikipedia data
    Returns:
        list[dict]: List of articles
        str: The corpus of all articles
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
        break
    return corpus, all_articles

def remove_stop_words(tokens):
    """Remove stopwords from the given corpus

    Args:
        tokens (list[str]): List of tokens from the corpus
    """
    # TODO: see if stopwords should be removed or not
    # stop_words = set(stopwords.words("english"))
    stop_words = set()
    filtered_tokens = [token for token in tokens if not token.lower() in stop_words]
    with open("wikipedia.token.stop", "a", encoding="utf-8") as f:
        f.write(" ".join(filtered_tokens))
        f.write("\n")
    return

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def noisy_channel(words):
    '''
    TODO: docstring
    '''
    for word in words:
        print(f"Correcting {word} to {correction(word)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noisy Channel Mode Spell-Checking Script")
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--correct', nargs="+", help="Words to display correct spelling for")
    options.add_argument('--proba', nargs="+", help="Words to display probability of spell correction for")
    args = parser.parse_args()

    mode = "correct" if args.correct is not None else "proba"
    words = args.correct if args.correct is not None else args.proba
    print (mode, words)
    # noisy_channel(words)
    
    # TODO: integrate with rest of code
    exit()
    stopwords = True
    if stopwords:
        with open("wikipedia.token.stop", "w", encoding="utf-8") as f:
            pass
    corpus, all_articles = load_articles("../../Assignment 2 Question 1/src/data_wikipedia") # TODO: check file folder location
    tokens = RegexpTokenizer(r"\w+").tokenize(corpus)
    if stopwords:
        print("Removing stopwords")
        nltk.download("stopwords")
        remove_stop_words(tokens)
        print("Saved corpus without stopwords to wikipedia.token.stop")
