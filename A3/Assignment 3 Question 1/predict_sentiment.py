"""
Author: Herteg Kohar
"""
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import argparse

MODEL_PATH = "sentiment_model.joblib"
TFIDF_PATH = "tfidf.joblib"
SETTINGS_PATH = "settings.json"


def preprocess_test_data(test_sentence, tfidf_vectorizer, dense):
    """Preprocess the test data

    Args:
        test_sentence (str): Sentence to predict sentiment
        tfidf_vectorizer (TfidVectorizer): Vectorizer used to train the model
        dense (bool): Boolean to indicate whether matrix is dense or not

    Returns:
        Matrix: Matrix with the test data
    """
    test_df = pd.DataFrame({"sentence": [test_sentence]})
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r"\w+")
    test_df["sentence"] = test_df["sentence"].apply(
        lambda x: tokenizer.tokenize(x.lower())
    )
    test_df["sentence"] = test_df["sentence"].apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    test_df["sentence"] = test_df["sentence"].apply(lambda x: " ".join(x))
    test_df_tfid = tfidf_vectorizer.transform(test_df["sentence"])
    if dense:
        test_df_tfid = test_df_tfid.toarray()
    return test_df_tfid


def predict_sentiment(model, model_name, tfidf_vectorizer, test_sentence):
    """Predict the sentiment of a sentence

    Args:
        model (sklearn model): Model used to predict the sentiment
        model_name (str): Name of the model
        tfidf_vectorizer (Matrix): Matrix with the test data
        test_sentence (str): Sentence to predict sentiment

    Returns:
        int: Prediction (1 for positive, 0 for negative)
    """
    dense = False
    if model_name == "naive":
        dense = True
    test_df_tfid = preprocess_test_data(test_sentence, tfidf_vectorizer, dense)
    prediction = model.predict(test_df_tfid)
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "text", type=str, help="Text to predict sentiment (must be in quotes)"
    )

    args = parser.parse_args()

    if (
        os.path.isfile(MODEL_PATH)
        and os.path.isfile(SETTINGS_PATH)
        and os.path.isfile(TFIDF_PATH)
    ):
        print(f"Sentiment prediction for: {args.text}")
        model = joblib.load(MODEL_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
        nltk.download("stopwords", quiet=True)

        prediction = predict_sentiment(
            model,
            settings["model"],
            tfidf_vectorizer,
            args.text,
        )

        if prediction == 1:
            print("Prediction: Positive")
        else:
            print("Prediction: Negative")
    else:
        print(
            "Model, vectorizer or settings not found. Please run train_sentiment.py first."
        )
