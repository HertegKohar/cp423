from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import nltk
import string
import json
import argparse
import os


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


def plot_zipfs(tokens):
    """Plot Zipf's Law for the given corpus

    Args:
        tokens (list[str]): List of tokens from the corpus
    """
    token_counts = Counter(tokens)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    counts = [x[1] for x in sorted_tokens]
    ranks = range(1, len(counts) + 1)
    plt.loglog(ranks, counts, base=10)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf's Law")
    plt.savefig("zipfs.png")
    print("Saved Zipf's Law plot to zipfs.png")
    plt.show()
    return


def tokenize_corpus(tokens):
    with open("wikipedia.token", "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))
    return


def remove_stop_words(tokens):
    """Remove stopwords from the given corpus

    Args:
        tokens (list[str]): List of tokens from the corpus
    """
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    filtered_tokens = [token for token in tokens if not token.lower() in stop_words]
    with open("wikipedia.token.stop", "w", encoding="utf-8") as f:
        f.write(" ".join(filtered_tokens))
    return


def stemming(tokens):
    """Apply stemming to tokens from the given corpus

    Args:
        tokens (list[str]): List of tokens from the corpus
    """
    porter_stemmer = PorterStemmer()
    stemmed_words = [porter_stemmer.stem(token) for token in tokens]
    with open("wikipedia.token.stemm", "w", encoding="utf-8") as f:
        f.write(" ".join(stemmed_words))
    return


def create_inverted_index(articles):
    """Create an inverted index for the given articles

    Args:
        articles (list[dict]): List of articles

    Returns:
        dict: Inverted index
    """
    inverted_index = defaultdict(list)
    for article in articles:
        tokens = word_tokenize(article["text"])
        counter = Counter(tokens)
        for token, count in counter.items():
            inverted_index[token].append((article["id"], count))
    with open("inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(inverted_index, f)
    return inverted_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command Line Interface for Wikipedia Processing"
    )
    parser.add_argument(
        "--zipf",
        help="Plot Zipf's Law",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--tokenize",
        help="Tokenize corpus",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--stopwords",
        help="Remove stopwords",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--stemming",
        help="Stem tokens",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--invertedindex",
        help="Create inverted index",
        action="store_true",
        default=False,
        required=False,
    )
    args = parser.parse_args()
    corpus, all_articles = load_articles("wikipedia_data")
    tokens = word_tokenize(corpus)
    if args.stopwords:
        print("Removing stopwords")
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
        remove_stop_words(tokens)
    if args.zipf:
        print("Plotting Zipf's Law")
        plot_zipfs(tokens)
    if args.tokenize:
        print("Tokenizing corpus")
        tokenize_corpus(tokens)
    if args.stemming:
        print("Applying stemming")
        nltk.download("punkt")
        stemming(tokens)
    if args.invertedindex:
        print("Creating inverted index")
        create_inverted_index(all_articles)
