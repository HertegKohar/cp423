## **Program Instructions**


Sample Command Line Use
```Bash
python cluster_news.py [-h] [--kmeans] [--whc] [--ac] [--dbscan] [--ncluster n1 [n2 ...]]
```
For boolean flags, use --flagname to set to True and exclude flag altogether to set to False:
```Bash
python program.py --flagname    # "flagname" is true
python program.py               # "flagname" is false
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
