import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import argparse

MODEL_PATH = "sentiment_model.joblib"
SETTINGS_PATH = "settings.json"


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


def create_bag_of_words(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf


def cross_validate_and_train(model, X_train_tfidf, y_train):
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }
    K_folds = 5
    cv_results = cross_validate(
        model,
        X_train_tfidf,
        y_train,
        cv=K_folds,
        scoring=scoring,
        return_estimator=True,
    )
    for i in range(K_folds):
        print(f"Fold: {i+1}")
        print(f"Accuracy: {cv_results['test_accuracy'][i]}")
        print(f"Precision: {cv_results['test_precision'][i]}")
        print(f"Recall: {cv_results['test_recall'][i]}")
        print(f"F1 Score: {cv_results['test_f1_score'][i]}")
        print("\n")
    print("Mean Metrics")
    print(f"Mean Accuracy: {np.mean(cv_results['test_accuracy'])}")
    print(f"Mean Precision: {np.mean(cv_results['test_precision'])}")
    print(f"Mean Recall: {np.mean(cv_results['test_recall'])}")
    print(f"Mean F1 Score: {np.mean(cv_results['test_f1_score'])}")
    print("\n")
    mean_scores = {
        metric: np.mean(cv_results[f"test_{metric}"]) for metric in scoring.keys()
    }
    best_metric = max(mean_scores, key=mean_scores.get)
    best_estimator_index = np.argmax(cv_results["test_" + best_metric])
    best_estimator = cv_results["estimator"][best_estimator_index]
    # print(f"Best Estimator: {best_estimator_index}")
    return best_estimator


def test_model(model, X_test_tfidf, y_test):
    print("Test Metrics")
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot()
    # RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.show()


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

    settings = {"imdb": args.imdb, "amazon": args.amazon, "yelp": args.yelp}

    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f)

    data = concatenate_data(args.imdb, args.amazon, args.yelp)
    nltk.download("stopwords", quiet=True)
    X_train, X_test, y_train, y_test = preprocess(data)
    X_train_tfidf, X_test_tfidf = create_bag_of_words(X_train, X_test)
    print("Included data sets:", ", ".join([k for k, v in settings.items() if v]))
    if args.knn:
        print(f"Training KNN with K={args.k_value}")
        clf = KNeighborsClassifier(n_neighbors=args.k_value)
    elif args.naive:
        print("Training Naive Bayes")
        clf = GaussianNB()
        X_train_tfidf = X_train_tfidf.toarray()
        X_test_tfidf = X_test_tfidf.toarray()
    elif args.svm:
        print("Training SVM")
        clf = SVC(kernel="linear")
    elif args.decisiontree:
        print("Training Decision Tree")
        clf = DecisionTreeClassifier()

    model = cross_validate_and_train(clf, X_train_tfidf, y_train)
    test_model(model, X_test_tfidf, y_test)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Settings saved to {SETTINGS_PATH}")
