"""
Authors:
    Kelvin Kellner
    Herteg Kohar
Resource: https://norvig.com/spell-correct.html
"""
import argparse
from collections import Counter
import json
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os
import re
import string

#TODO: confirm file name and path
FILENAME = "wikipedia.token"
DATA_FOLDER = "data_wikipedia"

def tokenize(text):
    "Return a list of all words from the text file."
    return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of 'word' occurring in the corpus."
    N_TOKENS = sum(WORD_FREQ_VECTOR.values())
    return WORD_FREQ_VECTOR[word] / N_TOKENS

def correction(word): 
    "Returns highest probability spelling correction canditate for 'word'."
    return max(candidates(word), key=P)

def candidates(word): 
    "Return set of best spelling correction candidates for 'word'."
    # candidates are generated using the following rules:
    # 1. if 'word' is a 'known' word, return it as there is no need to correct it
    # 2. otherise return the set of words that are only 1 edit distance away from 'word'
    # 3. if there are none, then return the set of words that are only 2 edit distance away
    # 4. if there are still none, then return the original 'word' as there is no correction to suggest
    return (known_words([word]) or known_words(edit_distance_of_1(word)) or known_words(edit_distance_of_2(word)) or [word])

def known_words(words): 
    "The subset of 'words' that appear in the corpus, and thus, in 'WORD_FREQ_VECTOR'."
    return set(w for w in words if w in WORD_FREQ_VECTOR)

def edit_distance_of_1(word):
    "All possible edits that are 1 edit distance away from 'word'."
    alphabet = string.ascii_lowercase
    pieces = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    insertions = [left + letter + right for left, right in pieces for letter in alphabet]
    deletions = [left + right[1:] for left, right in pieces if right]
    replacements = [left + letter + right[1:] for left, right in pieces if right for letter in alphabet]
    transpositions = [left + right[1] + right[0] + right[2:] for left, right in pieces if len(right) > 1]
    return set(insertions + deletions + replacements + transpositions)

def edit_distance_of_2(word): 
    "All possible edits that are two edits away from 'word'."
    # compute all combinations of 1 more edit on all edits of 1 distance from 'word'
    return (e2 for e1 in edit_distance_of_1(word) for e2 in edit_distance_of_1(e1))

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
        break # TODO: remove break to process all files
    return corpus, all_articles

def process_and_save_corpus(tokens):
    """Remove stopwords from the given corpus and output to a file

    Args:
        tokens (list[str]): List of tokens from the corpus
    """
    # TODO: see if stop word removal is needed
    # stop_words = set(stopwords.words("english"))
    # filtered_tokens = [token for token in tokens if not token.lower() in stop_words]
    stop_words = set()
    filtered_tokens = tokens
    with open(FILENAME, "a", encoding="utf-8") as f:
        f.write(" ".join(filtered_tokens))
        f.write("\n")
    print(f"Processed {len(filtered_tokens)} tokens and saved to {FILENAME}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noisy Channel Mode Spell-Checking Script")
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the existing corpus file')
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--correct', nargs="+", help="List of Words to spell-correct")
    options.add_argument('--proba', nargs="+", help="List of Words to show proportion of occurrences in the corpus for")
    args = parser.parse_args()

    # 1. Process and load the corpus
    exists = os.path.exists(FILENAME)
    overwrite_file = args.overwrite
    if not exists or overwrite_file:
        print(f"\nCorpus text file not found.\nProcessing and saving corpus to '{FILENAME}'...")
        with open(FILENAME, "w", encoding="utf-8") as f:
            pass
        corpus, all_articles = load_articles(DATA_FOLDER)
        tokens = RegexpTokenizer(r"\w+").tokenize(corpus)
        process_and_save_corpus(tokens)
        print()
    # term frequency vector for all words in the corpus
    # [(word: string, count: int), ...]
    WORD_FREQ_VECTOR = Counter(tokenize(open(FILENAME, encoding="utf-8").read()))

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
