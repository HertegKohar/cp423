"""
Author: Herteg Kohar
"""
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from Constants.constants import (
    TOPICS,
    TOPICS_MAP,
    MODEL_PATH,
    TFID_PATH,
    DOCUMENTS_PATH,
    MODELS_PATH,
)

# TOPICS = ["Astronomy", "Health", "Economy"]
# TOPICS_MAP = {"Astronomy": 0, "Health": 1, "Economy": 2}

# MODEL_PATH = "classifier.joblib"
# TFID_PATH = "tfidf.joblib"


def create_dataset(topics):
    data = {"hash": [], "topic": [], "text": []}
    for topic in topics:
        for file in os.listdir(os.path.join(DOCUMENTS_PATH, topic)):
            with open(
                os.path.join(DOCUMENTS_PATH, topic, file), "r", encoding="utf-8"
            ) as f:
                text = f.read()
                data["hash"].append(file)
                data["topic"].append(topic)
                data["text"].append(text)
    return pd.DataFrame(data)


def preprocess(data, topics_map):
    tokenizer = RegexpTokenizer(r"\w+")
    data["text"] = data["text"].apply(lambda x: tokenizer.tokenize(x.lower()))
    data["text"] = data["text"].apply(lambda x: " ".join(x))
    data["label"] = data["topic"].map(topics_map)
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


def create_tf_idf(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    joblib.dump(vectorizer, TFID_PATH)
    return X_train_tfidf, X_test_tfidf


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def cross_validate_and_train(model, X_train_tfidf, y_train):
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="macro"),
        "recall": make_scorer(recall_score, average="macro"),
        "f1_score": make_scorer(f1_score, average="macro"),
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
    # for i in range(K_folds):
    #     print(f"Fold: {i+1}")
    #     print(f"Accuracy: {cv_results['test_accuracy'][i]}")
    #     print(f"Precision: {cv_results['test_precision'][i]}")
    #     print(f"Recall: {cv_results['test_recall'][i]}")
    #     print(f"F1 Score: {cv_results['test_f1_score'][i]}")
    #     print("\n")
    print("Mean Metrics")
    print(f"Mean Accuracy: {np.mean(cv_results['test_accuracy'])}")
    print(f"Mean Precision: {np.mean(cv_results['test_precision'])}")
    print(f"Mean Recall: {np.mean(cv_results['test_recall'])}")
    print(f"Mean F1 Score: {np.mean(cv_results['test_f1_score'])}")

    mean_scores = {
        metric: np.mean(cv_results[f"test_{metric}"]) for metric in scoring.keys()
    }
    best_metric = max(mean_scores, key=mean_scores.get)
    best_estimator_index = np.argmax(cv_results["test_" + best_metric])
    best_estimator = cv_results["estimator"][best_estimator_index]
    # print(f"Best Estimator: {best_estimator_index}")
    return best_estimator


def test_model(model, X_test_tfidf, y_test, plot=False):
    print("Test Metrics")
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    if plot:
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=model.classes_
        ).plot()
        plt.show()
    return report


def train_and_test_models(X_train_tfidf, y_train, X_test_tfidf, y_test):
    models = [
        SVC(probability=True),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        LogisticRegression(),
    ]
    reports = {}
    for model in models:
        print(f"Model: {model.__class__.__name__}")
        best_estimator = cross_validate_and_train(model, X_train_tfidf, y_train)
        report = test_model(best_estimator, X_test_tfidf, y_test, plot=False)
        reports[model.__class__.__name__] = [best_estimator, report]
        print("\n")
    return reports


def choose_best_model(reports):
    # Choose best model based on highest accuracy, precision, recall and f1 score
    best_model = None
    best_score = 0
    model_name = None
    for model in reports:
        report = reports[model][1]
        score = (
            report["accuracy"]
            + report["macro avg"]["precision"]
            + report["macro avg"]["recall"]
            + report["macro avg"]["f1-score"]
        )
        if score > best_score:
            best_score = score
            best_model = reports[model][0]
            model_name = model
    return model_name, best_model


def training_pipeline():
    data = create_dataset(TOPICS)
    X_train, X_test, y_train, y_test = preprocess(data, TOPICS_MAP)
    X_train_tfidf, X_test_tfidf = create_tf_idf(X_train, X_test)
    reports = train_and_test_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    model_name, best_model = choose_best_model(reports)
    print(f"Best Model: {model_name}")
    joblib.dump(best_model, MODEL_PATH)


def predict_new_text(text):
    vectorizer = joblib.load(TFID_PATH)
    model = joblib.load(MODEL_PATH)
    test_df = pd.DataFrame({"text": [text]})
    tokenizer = RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words("english"))
    test_df["text"] = test_df["text"].apply(lambda x: tokenizer.tokenize(x.lower()))
    test_df["text"] = test_df["text"].apply(
        lambda x: [w for w in x if not w in stop_words]
    )
    test_df["text"] = test_df["text"].apply(lambda x: " ".join(x))
    X_test_tfidf = vectorizer.transform(test_df["text"])
    y_pred = model.predict_proba(X_test_tfidf)
    print("Predictions")
    for probs in y_pred:
        for i in range(len(probs)):
            print(f"{TOPICS[i]}: {probs[i]*100}%")


# Debugging
if __name__ == "__main__":
    training_pipeline()
    predict_new_text(
        "Stock market rallies as investors respond positively to new economic stimulus measures. Analysts predict strong growth for the tech sector as demand for digital services and products continues to rise. Meanwhile, concerns about inflation and supply chain disruptions persist, leading some experts to advise caution in certain areas. In the latest jobs report, unemployment drops to a record low, signaling a tightening labor market and potential wage pressures. As global economies recover from the pandemic, companies are exploring new opportunities for growth and investment, with emerging markets and renewable energy sectors attracting particular attention."
    )
