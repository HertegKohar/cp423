# **Program Instructions**  

Install Dependencies
```Bash
pip install -r requirements.txt
```

For boolean flags, use --flagname to set to True and exclude flag altogether to set to False:
```Bash
python program.py --flagname    # "flagname" is true
python program.py               # "flagname" is false
```


## **Question 1**

**This section goes over training scroll down to see how to predict**


Sample Command Line Use
```Bash
python training_sentiment.py [-h] [--imdb] [--amazon] [--yelp] (--naive | --knn | --svm | --decisiontree) [k_value]
```

**Any combination of the datasets can be used but at least one must be specified.**

Example Execution of KNN with K=5

**KNN needs a value to be specified for K.**

Example Execution of KNN with K=5:
```Bash
python training_sentiment.py --imdb --yelp --knn 5
```
Example Execution of SVM:
```Bash
python training_sentiment.py --imdb --amazon --yelp --svm
```

Example Execution of Naive Bayes Classifier:

```Bash
python training_sentiment.py --yelp --amazon --naive
```

Example Execution of Decision Tree Classifier:

```Bash
python training_sentiment.py --imdb --amazon --decisiontree
```

**Predicting Program Instructions**

You must execute the training program first before executing the predicting program. It generates the model file as well as the settings in order to predict new text

**Text must be in quotes**

Sample Command Line Use
```Bash
python predict_sentiment.py [-h] text
```

Example Execution of Predicting Program:

```Bash
python predict_sentiment.py "I love this movie"
```

```Bash
python cluster_news.py --kmeans --whc --ac --dbscan --ncluster 10 20 40
```


## **Question 2**

Sample Command Line Use
```Bash
python cluster_news.py [-h] [--kmeans] [--whc] [--ac] [--dbscan] [--ncluster n1 [n2 ...]]
```

Example Execution of K-Means Clustering with 10 clusters:
```Bash
python cluster_news.py --kmeans --ncluster 10
```
Example Execution of K-Means Clustering with 20 clusters:
```Bash
python cluster_news.py --kmeans --ncluster 20
```
Example Execution of K-Means Clustering with 40 clusters:
```Bash
python cluster_news.py --kmeans --ncluster 40
```

Example Execution of K-Means Clustering with 10, 20, and 40 clusters:
```Bash
python cluster_news.py --kmeans --ncluster 10 20 40
```

Example Execution of Ward Hierarchical Clustering with 10, 20, and 40 clusters:
```Bash
python cluster_news.py --whc --ncluster 10 20 40
```

Example Execution of Agglomerative Clustering with 10, 20, and 40 clusters:
```Bash
python cluster_news.py --ac --ncluster 10 20 40
```

Example Execution of DBSCAN Clustering with 10, 20, and 40 clusters:
```Bash
python cluster_news.py --dbscan --ncluster 10 20 40
```
