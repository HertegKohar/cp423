"""
Author: Herteg Kohar
"""
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, PlaintextCorpusReader
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import nltk
import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


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


def load_articles_incremented(path, increment):
    """Load articles from the given path to directory for wikipedia data

    Args:
        path (str): Path to directory containing wikipedia data

    Returns:
        list[dict]: List of articles
        str: The corpus of all articles
    """
    file_paths = []
    for i, file in enumerate(os.listdir(path)):
        file_paths.append(f"{path}/{file}")
        if i % increment == 0 and i != 0:
            yield file_paths
            file_paths = []
    yield file_paths


def write_article_text(article, path):
    with open(f"{path}/{article['id']}.txt", "w", encoding="utf-8") as f:
        f.write(article["text"])


def load_article_texts_threaded(path):
    start = time.time()
    count = 0
    if not os.path.exists("article_texts"):
        os.mkdir("article_texts")
    for file in os.listdir(path):
        with open(f"{path}/{file}", encoding="utf-8") as input_file:
            articles = json.load(input_file)
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(write_article_text, article, "article_texts")
                    for article in articles
                ]
                for _ in as_completed(futures):
                    pass
        count += 1
        print("Finished file: ", count)
    end = time.time()
    print(f"Time taken: {end - start}")


def load_corpus(path):
    fileids = os.listdir(path)
    corpus = PlaintextCorpusReader(path, fileids, encoding="utf-8")
    return corpus


def tokenize_generator(tokenizer, text):
    for token in tokenizer.tokenize(text):
        yield token


def tokenize_threaded(corpus):
    start = time.time()
    count = 0
    chunk_size = 10000
    max_workers = os.cpu_count()
    tokenizer = RegexpTokenizer(r"\w+")
    futures = []
    with open("wikipedia.token", "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, fileid in enumerate(corpus.fileids()):
                if (i + 1) % chunk_size == 0:
                    # If memory error write to file here
                    for future in as_completed(futures):
                        for token in future.result():
                            f.write(token)
                            f.write("\n")
                    count += 1
                    print("Finished chunk: ", count)
                    futures = []
                futures.append(
                    executor.submit(tokenize_generator, tokenizer, corpus.raw(fileid))
                )
            print("Last chunk")
            if futures:
                for future in as_completed(futures):
                    for token in future.result():
                        f.write(token)
                        f.write("\n")
                futures.clear()
    end = time.time()
    print(f"Time taken: {end - start}")


def stemming_generator(porter_stemmer, tokenizer, text):
    for token in tokenize_generator(tokenizer, text):
        yield porter_stemmer.stem(token)


def tokenize_and_stemm_threaded(corpus):
    start = time.time()
    count = 0
    chunk_size = 10000
    max_workers = os.cpu_count()
    porter_stemmer = PorterStemmer()
    futures = []
    tokenizer = RegexpTokenizer(r"\w+")
    with open("wikipedia.token.stemm", "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, fileid in enumerate(corpus.fileids()):
                if (i + 1) % chunk_size == 0:
                    # If memory error write to file here
                    for future in as_completed(futures):
                        for token in future.result():
                            f.write(token)
                            f.write("\n")
                    count += 1
                    print("Finished chunk: ", count)
                    futures = []
                futures.append(
                    executor.submit(
                        stemming_generator,
                        porter_stemmer,
                        tokenizer,
                        corpus.raw(fileid),
                    )
                )
            print("Last chunk")
            if futures:
                for future in as_completed(futures):
                    for token in future.result():
                        f.write(token)
                        f.write("\n")
                futures.clear()
    end = time.time()
    print(f"Time taken: {end - start}")


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


def save_tokenized_corpus(tokens):
    with open("wikipedia.token", "a", encoding="utf-8") as f:
        for token in tokens:
            f.write(token)
            f.write("\n")
    return


def remove_stop_words(tokens):
    """Remove stopwords from the given corpus

    Args:
        tokens (list[str]): List of tokens from the corpus
    """
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if not token.lower() in stop_words]
    with open("wikipedia.token.stop", "a", encoding="utf-8") as f:
        f.write(" ".join(filtered_tokens))
        f.write("\n")
    return


def stemming(tokens):
    """Apply stemming to tokens from the given corpus

    Args:
        tokens (list[str]): List of tokens from the corpus
    """
    porter_stemmer = PorterStemmer()
    stemmed_words = [porter_stemmer.stem(token) for token in tokens]
    with open("wikipedia.token.stemm", "a", encoding="utf-8") as f:
        f.write(" ".join(stemmed_words))
        f.write("\n")
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


def open_and_create_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        corpus = " ".join([article["text"] for article in articles])
        tokens = word_tokenize(corpus)
    return tokens


def open_and_create_corpus_threaded(path, max_workers):
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(word_tokenize, article["text"]) for article in articles
            ]
            for future in as_completed(futures):
                yield future.result()


def tokenize_and_stem(text, porter_stemmer):
    for token in word_tokenize(text):
        yield porter_stemmer.stem(token)


def stemming_threaded(path, porter_stemmer, max_workers):
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(tokenize_and_stem, article["text"], porter_stemmer)
                for article in articles
            ]
            for future in as_completed(futures):
                yield future.result()


def perform_tasks_threaded(**kwargs):
    if kwargs["tokenize"]:
        count = 0
        max_workers = 3
        start = time.time()
        with open("wikipedia.token", "a", encoding="utf-8") as f:
            for file_paths in load_articles_incremented("data_wikipedia", max_workers):
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            open_and_create_corpus_threaded, path, max_workers + 1
                        )
                        for path in file_paths
                    ]
                    for future in as_completed(futures):
                        count += 1
                        for token in future.result():
                            f.write(token)
                            f.write("\n")
                        print("Processed", count, "files")
        print("Tokenized corpus in", time.time() - start, "seconds")
    if kwargs["stemming"]:
        # Don't open the file just tokenize and stem in the multi-level threading (second level)
        # Opening token file makes it too slow
        porter_stemmer = PorterStemmer()
        max_workers = 3
        count = 0
        start = time.time()
        with open("wikipedia.token.stemm", "a", encoding="utf-8") as f:
            for file_paths in load_articles_incremented("data_wikipedia", max_workers):
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            stemming_threaded, path, porter_stemmer, max_workers + 1
                        )
                        for path in file_paths
                    ]
                    for future in as_completed(futures):
                        count += 1
                        for token in future.result():
                            f.write(" ".join(token))
                            f.write("\n")
                        print("Processed", count, "files")
        end = time.time()
        print("Time taken for stemming:", end - start)

        # print("Processing", len(articles), "articles")
        # for article in articles:
        #     tokens = word_tokenize(article["text"])
        #     if kwargs["stopwords"]:
        #         remove_stop_words(tokens)
        #     if kwargs["tokenize"]:
        #         tokenize_corpus(tokens)
        #     if kwargs["stemming"]:
        #         stemming(tokens)
        # count += len(articles)
        # print("Processed", count, "articles")
    # if kwargs["invertedindex"]:
    #     create_inverted_index(all_articles)
    # if kwargs["zipf"]:
    #     plot_zipfs(tokens)


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
    # Clear token files
    if args.tokenize:
        with open("wikipedia.token", "w", encoding="utf-8") as f:
            pass
    if args.stopwords:
        with open("wikipedia.token.stop", "w", encoding="utf-8") as f:
            pass
    if args.stemming:
        with open("wikipedia.token.stemm", "w", encoding="utf-8") as f:
            pass
    # print("Loading corpus")
    # start = time.time()
    # corpus = load_corpus("article_texts")
    # end = time.time()
    # print("Time taken to load corpus:", end - start)
    corpus, all_articles = load_articles("data_wikipedia")
    tokens = RegexpTokenizer(r"\w+").tokenize(corpus)
    if args.stopwords:
        print("Removing stopwords")
        nltk.download("stopwords")
        remove_stop_words(tokens)
        print("Saved corpus without stopwords to wikipedia.token.stop")
    if args.tokenize:
        print("Tokenizing corpus")
        save_tokenized_corpus(tokens)
        print("Saved tokenized corpus to wikipedia.token")
    if args.stemming:
        print("Applying stemming")
        nltk.download("punkt")
        stemming(tokens)
        print("Saved stemmed corpus to wikipedia.token.stemm")
    if args.invertedindex:
        print("Creating inverted index")
        create_inverted_index(all_articles)
        print("Saved inverted index to inverted_index.json")
    if args.zipf:
        print("Plotting Zipf's Law")
        plot_zipfs(tokens)
