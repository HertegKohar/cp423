"""
Author: Kelvin Kellner
Reference: https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python
"""
import argparse
import joblib
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
import pandas as pd
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
)
from sklearn.preprocessing import MinMaxScaler

MAX_FEATURES = 1000
N_GRAM_RANGE = (1,2)
MIN_DF = 10
MAX_DF = 1000

def concatenate_data():
    data = []
    columns = ["news_group", "article_number", "cluster", "text"]
    path = "20_newsgroups"
    for news_group in os.listdir(path):
        # append all articles in news group to data
        for article in os.listdir(os.path.join(path, news_group)):
            with open(os.path.join(path, news_group, article), "r") as f:
                text = f.read()
            data.append([news_group, article, -1, text])
    data = pd.DataFrame(data, columns=columns)
    return data

def preprocess(text_data):
    tokenizer = RegexpTokenizer(r"\w+")
    # remove headers
    text_data = text_data.apply(
        lambda x: re.sub(r'^.*?\n\n', '', x, flags=re.DOTALL)
    )
    # tokenize
    text_data = text_data.apply(
        lambda x: tokenizer.tokenize(x.lower())
    )
    text_data = text_data.apply(lambda x: " ".join(x))
    return text_data

def load_or_create_preprocessed():
    # use preprocessed data file if it exists, else create it
    if os.path.exists("data/articles_preprocessed.csv"):
        df_articles = pd.read_csv("data/articles_preprocessed.csv")
    else:
        df_articles = concatenate_data()
        df_articles["text"] = preprocess(df_articles["text"])
        df_articles.to_csv("data/articles_preprocessed.csv")
    return df_articles

def load_or_create_tfidf(df_articles):
    # use tfidf data file if it exists, else create it
    if os.path.exists("data/articles_tfidf.csv"):
        df_articles_tfidf = pd.read_csv("data/articles_tfidf.csv")
    else:
        # creating a new TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES, ngram_range=N_GRAM_RANGE, strip_accents="unicode", min_df=MIN_DF, max_df=MAX_DF)
        tfidf_article_array = tfidf.fit_transform(df_articles["text"])
        df_articles_tfidf = pd.DataFrame(tfidf_article_array.toarray(), index=df_articles.index, columns=tfidf.get_feature_names_out())
        df_articles_tfidf.describe()
        df_articles_tfidf.to_csv("data/articles_tfidf.csv")
    return df_articles_tfidf

def load_or_create_pca(df_articles_tfidf):
    # use pca data file if it exists, else create it
    if os.path.exists("data/articles_pca.csv"):
        df_articles_pca = pd.read_csv("data/articles_pca.csv")
    else:
        # using PCA to reduce the dimensionality
        scaler = MinMaxScaler()
        data_rescaled = scaler.fit_transform(df_articles_tfidf)
        # 99% of variance
        pca = PCA(n_components = 0.99)
        pca.fit(data_rescaled)
        reduced = pca.transform(data_rescaled)
        df_articles_pca = pd.DataFrame(data=reduced)
        df_articles_pca.describe()
        df_articles_pca.to_csv("data/articles_pca.csv")
    return df_articles_pca

def print_metrics(df_articles, df_articles_labeled):
    # print metrics
    print(f"Adjusted Mutual Information: {adjusted_mutual_info_score(df_articles['news_group'], df_articles_labeled['cluster'])}")
    print(f"Adjusted Rand Score: {adjusted_rand_score(df_articles['news_group'], df_articles_labeled['cluster'])}")
    print(f"Completeness Score: {completeness_score(df_articles['news_group'], df_articles_labeled['cluster'])}")
    print()

def findOptimalEps(n_neighbors, data):
    '''
    function to find optimal eps distance when using DBSCAN; based on this article: https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
    '''
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

def cluster_using_kmeans(df_articles, df_articles_pca, ncluster):
    # using KMeans clustering
    print(f"Clustering using KMeans with {ncluster} clusters")
    kmeans = KMeans(n_clusters=ncluster, n_init='auto')
    articles_labels = kmeans.fit_predict(df_articles_pca)
    df_articles_labeled = df_articles.copy()
    df_articles_labeled["cluster"] = articles_labels
    # TODO: save model
    return df_articles_labeled

def cluster_using_whc(df_articles, df_articles_pca, ncluster):
    # using Ward Hierarchical Clustering
    print(f"Clustering using Ward Hierarchical Clustering with {ncluster} clusters")
    whc = AgglomerativeClustering(n_clusters=ncluster, linkage="ward")
    articles_labels = whc.fit_predict(df_articles_pca)
    df_articles_labeled = df_articles.copy()
    df_articles_labeled["cluster"] = articles_labels
    # TODO: save model
    return df_articles_labeled

def cluster_using_ac(df_articles, df_articles_pca, ncluster):
    # using Agglomerative Clustering
    print(f"Clustering using Agglomerative Clustering with {ncluster} clusters")
    ac = AgglomerativeClustering(n_clusters=ncluster, linkage="average")
    articles_labels = ac.fit_predict(df_articles_pca)
    df_articles_labeled = df_articles.copy()
    df_articles_labeled["cluster"] = articles_labels
    # TODO: save model
    return df_articles_labeled

def cluster_using_dbscan(df_articles, df_articles_tfidf, df_articles_pca, ncluster):
    # using DBSCAN clustering
    print(f"Clustering using DBSCAN with {ncluster} clusters")
    findOptimalEps(2, df_articles_tfidf)
    dbscan = DBSCAN(eps=0.2, metric="euclidean")
    articles_labels = dbscan.fit_predict(df_articles_pca)
    df_articles_labeled = df_articles.copy()
    df_articles_labeled["cluster"] = articles_labels
    # TODO: save model
    return df_articles_labeled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News Article Clustering")
    parser.add_argument(
        '--ncluster',
        type=int,
        nargs='+',
        default=[20],
        help='Number of clusters, multiple numbers repeats clustering for each'
    )
    parser.add_argument(
        "--kmeans",
        action="store_true",
        default=False,
        help="Use KMeans clustering",
    )
    parser.add_argument(
        "--whc",
        action="store_true",
        default=False,
        help="Use Ward Hierarchical Clustering",
    )
    parser.add_argument(
        "--ac",
        action="store_true",
        default=False,
        help="Use Agglomerative Clustering",
    )
    parser.add_argument(
        "--dbscan",
        action="store_true",
        default=False,
        help="Use DBSCAN clustering",
    )
    args = parser.parse_args()
    if not args.ncluster:
        parser.error("Number of clusters required")
    if not args.kmeans and not args.whc and not args.ac and not args.dbscan:
        parser.error("No clustering model selected")

    # prepare data
    df_articles = load_or_create_preprocessed()
    df_articles_tfidf = load_or_create_tfidf(df_articles)
    df_articles_pca = load_or_create_pca(df_articles_tfidf)

    # for each number of clusters
    for ncluster in args.ncluster:
        print(f"\n----- {ncluster} CLUSTERS -----\n")

        # cluster using the appropriate model(s)
        if args.kmeans:
            kmeans = cluster_using_kmeans(df_articles, df_articles_pca, ncluster)
            print_metrics(df_articles, kmeans)
        if args.whc:
            whc = cluster_using_whc(df_articles, df_articles_pca, ncluster)
            print_metrics(df_articles, whc)
        if args.ac:
            ac = cluster_using_ac(df_articles, df_articles_pca, ncluster)
            print_metrics(df_articles, ac)
        if args.dbscan:
            dbscan = cluster_using_dbscan(df_articles, df_articles_tfidf, df_articles_pca, ncluster)
            print_metrics(df_articles, dbscan)
