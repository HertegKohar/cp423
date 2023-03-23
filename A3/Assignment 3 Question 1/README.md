**Program Instructions**


**This section goes over training scroll down to see how to predict**


Sample Command Line Use
```Bash
python training_sentiment.py [-h] [--imdb] [--amazon] [--yelp] (--naive | --knn | --svm | --decisiontree) [k_value]
```
For boolean flags, use --flagname to set to True and exclude flag altogether to set to False:
```Bash
python program.py --flagname    # "flagname" is true
python program.py               # "flagname" is false
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


