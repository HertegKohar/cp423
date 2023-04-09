"""
Authors:
    Herteg Kohar
    Kelvin Kellner
"""
from Constants.constants import (
    N_DOCUMENTS,
    EXTENSION,
    DOCUMENTS_PATH,
    INVERTED_INDEX_PATH,
    MAPPING_PATH,
    HASH_TO_URL_PATH,
)
from Spell_Correct.spell_correct import spell_correct_query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from dataclasses import dataclass, field
from colorama import Fore, Style
import pandas as pd
import json
import os

# Make a class to hold document path and similarity score
@dataclass
class Document:
    path: str = field(default=None)
    score: float = field(default=None)
    hash_: str = field(default=None)
    url: str = field(default=None)


def preprocess_query(query):
    df = pd.DataFrame({"text": [query]})
    tokenizer = RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words("english"))
    df["text"] = df["text"].apply(lambda x: tokenizer.tokenize(x.lower()))
    df["text"] = df["text"].apply(lambda x: [w for w in x if not w in stop_words])
    df["text"] = df["text"].apply(lambda x: " ".join(x))
    return df["text"]


def get_docs(query_df, inverted_index):
    docs = []
    for word in query_df[0].split():
        if word in inverted_index:
            docs.append(inverted_index[word])
    return docs


def initialize_documents(docs, reversed_mapping, hash_to_url_dict):
    seen = set()
    documents = []
    for doc in docs:
        for occurence in doc["occurences"]:
            if occurence[0] not in seen:
                document = Document()
                path = os.path.join(
                    DOCUMENTS_PATH,
                    occurence[2],
                    reversed_mapping[occurence[0]] + EXTENSION,
                )
                document.path = path
                document.hash_ = occurence[0]
                document.url = hash_to_url_dict[reversed_mapping[occurence[0]]]
                documents.append(document)
                seen.add(occurence[0])
    return documents


def compute_similarity(documents, query_df):
    for document in documents:
        with open(document.path, "r", encoding="utf-8") as f:
            text = f.read()
        df = pd.DataFrame({"text": [text]})
        tokenizer = RegexpTokenizer(r"\w+")
        # stop_words = set(stopwords.words("english"))
        df["text"] = df["text"].apply(lambda x: tokenizer.tokenize(x.lower()))
        # df["text"] = df["text"].apply(lambda x: [w for w in x if not w in stop_words])
        df["text"] = df["text"].apply(lambda x: " ".join(x))
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(df["text"])
        X_test_tfidf = vectorizer.transform(query_df)
        score = cosine_similarity(X_train_tfidf, X_test_tfidf)
        document.score = score[0][0]
    documents.sort(key=lambda x: x.score, reverse=True)
    # Only return top 3
    return documents[:N_DOCUMENTS]


def get_snippet(text, query_words):
    # Split text into sentences
    sentences = sent_tokenize(text)
    # Find sentences containing query words
    relevant_sentences = [
        s for s in sentences if any(q.lower() in s.lower() for q in query_words)
    ]
    # Join relevant sentences to form snippet
    snippet = "\n\n".join(relevant_sentences)
    return snippet


def display_highlighted_terms(documents, query):
    for document in documents:
        with open(document.path, "r", encoding="utf-8") as f:
            text = f.read()
        query_words = query.split()
        text = get_snippet(text, query_words)
        highlighted_document = text
        for term in query_words:
            highlighted_document = highlighted_document.replace(
                term, f"{Fore.GREEN}{term}{Style.RESET_ALL}"
            )
        print(
            f"Document: {document.hash_}, Path: {document.path}, URL: {document.url}\n"
        )
        print(f"{highlighted_document}\n")


def query_documents(query):
    query = query.lower()
    query_df = preprocess_query(query)

    with open(INVERTED_INDEX_PATH, "r") as f:
        inverted_index = json.load(f)
    with open(HASH_TO_URL_PATH, "r") as f:
        hash_to_url_dict = json.load(f)
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    reversed_mapping = {v: k for k, v in mapping.items()}

    query_df[0] = spell_correct_query(query_df[0], inverted_index)
    print("\nSpell corrected query to:", query_df[0], end="\n\n")

    documents = get_docs(query_df, inverted_index)

    documents = initialize_documents(documents, reversed_mapping, hash_to_url_dict)

    documents = compute_similarity(documents, query_df)

    display_highlighted_terms(documents, query_df[0])
