"""
Author: Kelvin Kellner
Reference: https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python
"""
import argparse
import joblib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
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

# TF-IDF parameters
MAX_FEATURES = 250
MIN_DF = 20
MAX_DF = 500

MODEL_PATH = "models\\"

def concatenate_data():
    data = []
    columns = ["news_group", "article_number", "cluster", "text"]
    path = "20_newsgroups"
    news_group_index = -1
    for news_group in os.listdir(path):
        news_group_index += 1
        # append all articles in news group to data
        for article in os.listdir(os.path.join(path, news_group)):
            with open(os.path.join(path, news_group, article), "r") as f:
                text = f.read()
            data.append([news_group_index, article, -1, text])
    data = pd.DataFrame(data, columns=columns)
    return data

def preprocess(text_data):
    nltk.download("stopwords", quiet=True)
    tokenizer = RegexpTokenizer(r"\w+")
    stop_words = set(stopwords.words("english"))
    # remove headers
    text_data = text_data.apply(
        lambda x: re.sub(r'^.*?\n\n', '', x, flags=re.DOTALL)
    )
    # tokenize and lowercase
    text_data = text_data.apply(
        lambda x: tokenizer.tokenize(x.lower())
    )
    # remove stop words
    text_data = text_data.apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    text_data = text_data.apply(lambda x: " ".join(x))
    return text_data

def load_or_create_preprocessed():
    print("Loading or creating preprocessed data...")
    # use preprocessed data file if it exists, else create it
    if os.path.exists("data/articles_preprocessed.csv"):
        df_articles_1 = pd.read_csv("data/articles_preprocessed.csv")
    # else:
    df_articles = concatenate_data()
    df_articles["text"] = preprocess(df_articles["text"])
    df_articles.to_csv("data/articles_preprocessed.csv", index=False)

    print(df_articles)
    print(df_articles_1)
    comp = df_articles.compare(df_articles_1, align_axis=0)
    print(comp)
    exit(0)
    return df_articles

def load_or_create_tfidf(df_articles):
    print("Loading or creating TF-IDF data...")
    # use tfidf data file if it exists, else create it
    # if os.path.exists("data/articles_tfidf.csv"):
    #     df_articles_tfidf = pd.read_csv("data/articles_tfidf.csv")
    # else:
    # creating a new TF-IDF matrix
    # tfidf = TfidfVectorizer(stop_words="english", strip_accents="unicode", min_df=MIN_DF)
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, strip_accents="unicode", min_df=MIN_DF, max_df=MAX_DF)
    tfidf_article_array = tfidf.fit_transform(df_articles["text"])
    df_articles_tfidf = pd.DataFrame(tfidf_article_array.toarray(), index=df_articles.index, columns=tfidf.get_feature_names_out())
    df_articles_tfidf.to_csv("data/articles_tfidf.csv")
    return df_articles_tfidf

def load_or_create_pca(df_articles_tfidf):
    print("Loading or creating PCA data...")
    # use pca data file if it exists, else create it
    if os.path.exists("data/articles_pca.csv"):
        df_articles_pca = pd.read_csv("data/articles_pca.csv")
    else:
        # using PCA to reduce the dimensionality
        scaler = MinMaxScaler()
        data_rescaled = scaler.fit_transform(df_articles_tfidf)
        # variance explained by 90% of components
        pca = PCA(n_components = 0.90)
        pca.fit(data_rescaled)
        reduced = pca.transform(data_rescaled)
        df_articles_pca = pd.DataFrame(data=reduced)
        df_articles_pca.to_csv("data/articles_pca.csv")
        # Calculate the variance explained by principle components
        print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))
        print(' Number of components:', pca.n_components_)
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

def cluster_using_kmeans(df_articles, df_for_prediction, ncluster):
    # using KMeans clustering
    print(f"Clustering using KMeans with {ncluster} clusters")
    kmeans = KMeans(n_clusters=ncluster, n_init='auto')
    articles_predictions = kmeans.fit_predict(df_for_prediction)
    df_articles_predicted = df_articles.copy()
    df_articles_predicted["cluster"] = articles_predictions
    # TODO: save model
    joblib.dump(kmeans, MODEL_PATH+'kmeans_n' +str(ncluster)+'.joblib')
    return df_articles_predicted

def cluster_using_whc(df_articles, df_for_prediction, ncluster):
    # using Ward Hierarchical Clustering
    print(f"Clustering using Ward Hierarchical Clustering with {ncluster} clusters")
    whc = AgglomerativeClustering(n_clusters=ncluster, linkage="ward")
    articles_predictions = whc.fit_predict(df_for_prediction)
    df_articles_predicted = df_articles.copy()
    df_articles_predicted["cluster"] = articles_predictions
    # TODO: save model
    joblib.dump(whc, MODEL_PATH+'whc_n' +str(ncluster)+'.joblib')
    return df_articles_predicted

def cluster_using_ac(df_articles, df_for_prediction, ncluster):
    # using Agglomerative Clustering
    print(f"Clustering using Agglomerative Clustering with {ncluster} clusters")
    ac = AgglomerativeClustering(n_clusters=ncluster, linkage="average")
    articles_predictions = ac.fit_predict(df_for_prediction)
    df_articles_predicted = df_articles.copy()
    df_articles_predicted["cluster"] = articles_predictions
    # TODO: save model
    joblib.dump(ac, MODEL_PATH+'ac_n' +str(ncluster)+'.joblib')
    return df_articles_predicted

def cluster_using_dbscan(df_articles, df_for_prediction, ncluster):
    # using DBSCAN clustering
    print(f"Clustering using DBSCAN with {ncluster} clusters")
    # findOptimalEps(2, df_for_prediction)
    dbscan = DBSCAN(eps=1.6, metric="cosine") # metric: euclidean or cosine
    articles_predictions = dbscan.fit_predict(df_for_prediction)
    df_articles_predicted = df_articles.copy()
    df_articles_predicted["cluster"] = articles_predictions
    # TODO: save model
    joblib.dump(dbscan, MODEL_PATH+'dbscan_n' +str(ncluster)+'.joblib')
    return df_articles_predicted

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
    # df_articles_pca = load_or_create_pca(df_articles_tfidf)
    df_for_prediction = df_articles_tfidf

    # for each number of clusters
    for ncluster in args.ncluster:
        print(f"\n----- {ncluster} CLUSTERS -----\n")

        # cluster using the appropriate model(s)
        if args.kmeans:
            kmeans = cluster_using_kmeans(df_articles, df_for_prediction, ncluster)
            print_metrics(df_articles, kmeans)
        if args.whc:
            whc = cluster_using_whc(df_articles, df_for_prediction, ncluster)
            print_metrics(df_articles, whc)
        if args.ac:
            ac = cluster_using_ac(df_articles, df_for_prediction, ncluster)
            print_metrics(df_articles, ac)
        if args.dbscan:
            dbscan = cluster_using_dbscan(df_articles, df_for_prediction, ncluster)
            print_metrics(df_articles, dbscan)
