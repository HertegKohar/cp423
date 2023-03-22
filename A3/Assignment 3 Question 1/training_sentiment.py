import pandas as pd
import argparse
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Analysis Training")
    parser.add_argment("--imdb", action="store_true", default=False, required=False)

    parser.add_argument("--amazon", action="store_true", default=False, required=False)

    parser.add_argument("--yelp", action="store_true", default=False, required=False)

    model_group = parser.add_mutually_exclusive_group(required=True)

    model_group.add_argument("--naive", action="store_true")
    model_group.add_argument("--knn", action="store_true")
    model_group.add_argument("--svm", action="store_true")
    model_group.add_argument("--decisiontree", action="store_true")

    parser.add_argument("--k", type=int, help="K value for KNN")

    args = parser.parse_args()

    if args.knn and not args.k:
        parser.error("KNN requires a K value")

    if args.knn:
        print(f"Training KNN with K={args.k}")
    elif args.naive:
        print("Training Naive Bayes")
    elif args.svm:
        print("Training SVM")
    elif args.decisiontree:
        print("Training Decision Tree")
