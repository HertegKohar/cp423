import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import nltk


def concatenate_data(imdb, amazon, yelp):
    data = []
    columns = ["sentence", "label"]
    path = "sentiment labelled sentences"
    if imdb:
        data.append(
            pd.read_csv(
                os.path.join(path, "imdb_labelled.txt"), sep="\t", names=columns
            )
        )
    if amazon:
        data.append(
            pd.read_csv(
                os.path.join(path, "amazon_cells_labelled.txt"), sep="\t", names=columns
            )
        )
    if yelp:
        data.append(
            pd.read_csv(
                os.path.join(path, "yelp_labelled.txt"), sep="\t", names=columns
            )
        )
    return pd.concat(data)


def preprocess(text_data):
    tokenizer = RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words("english"))
    text_data["sentence"] = text_data["sentence"].apply(
        lambda x: tokenizer.tokenize(x.lower())
    )
    text_data["sentence"] = text_data["sentence"].apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    text_data["sentence"] = text_data["sentence"].apply(lambda x: " ".join(x))
    X_train, X_test, y_train, y_test = train_test_split(
        text_data["sentence"], text_data["label"], test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis Training")

    parser.add_argument(
        "--imdb",
        action="store_true",
        default=False,
        help="Include IMDB data set",
    )

    parser.add_argument(
        "--amazon",
        action="store_true",
        default=False,
        help="Include Amazon data set",
    )

    parser.add_argument(
        "--yelp",
        action="store_true",
        default=False,
        help="Include Yelp data set",
    )

    model_group = parser.add_mutually_exclusive_group(required=True)

    model_group.add_argument("--naive", action="store_true")
    model_group.add_argument("--knn", action="store_true")
    model_group.add_argument("--svm", action="store_true")
    model_group.add_argument("--decisiontree", action="store_true")

    parser.add_argument("k_value", nargs="?", type=int, help="K value for KNN")

    args = parser.parse_args()

    if not args.imdb and not args.amazon and not args.yelp:
        parser.error("No data sets selected")

    if args.knn and not args.k_value:
        parser.error("KNN requires a K value")

    data = concatenate_data(args.imdb, args.amazon, args.yelp)
    nltk.download("stopwords")

    if args.knn:
        print(f"Training KNN with K={args.k_value}")
    elif args.naive:
        print("Training Naive Bayes")
    elif args.svm:
        print("Training SVM")
    elif args.decisiontree:
        print("Training Decision Tree")
