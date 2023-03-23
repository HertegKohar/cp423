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


def preprocess_test_data(test_sentence, imdb, amazon, yelp, dense):
    test_df = pd.DataFrame({"sentence": [test_sentence]})
    data = concatenate_data(imdb, amazon, yelp)
    X_train, _, _, _ = preprocess(data)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train)
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r"\w+")
    test_df["sentence"] = test_df["sentence"].apply(
        lambda x: tokenizer.tokenize(x.lower())
    )
    test_df["sentence"] = test_df["sentence"].apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    test_df["sentence"] = test_df["sentence"].apply(lambda x: " ".join(x))
    test_df_tfid = vectorizer.transform(test_df["sentence"])
    if dense:
        test_df_tfid = test_df_tfid.toarray()
    return test_df_tfid


def predict_sentiment(model, model_name, test_sentence, imdb, amazon, yelp):
    dense = False
    if model_name == "naive":
        dense = True
    test_df_tfid = preprocess_test_data(test_sentence, imdb, amazon, yelp, dense)
    prediction = model.predict(test_df_tfid)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "text", type=str, help="Text to predict sentiment (must be in quotes)"
    )

    args = parser.parse_args()

    if os.path.isfile(MODEL_PATH) and os.path.isfile(SETTINGS_PATH):
        print(f"Sentiment prediction for: {args.text}")
        model = joblib.load(MODEL_PATH)
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        nltk.download("stopwords", quiet=True)

        prediction = predict_sentiment(
            model,
            settings["model"],
            args.text,
            settings["imdb"],
            settings["amazon"],
            settings["yelp"],
        )

        if prediction == 1:
            print("Prediction: Positive")
        else:
            print("Prediction: Negative")
    else:
        print("Model or settings not found. Please run train_sentiment.py first.")
