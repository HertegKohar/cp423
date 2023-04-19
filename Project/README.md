# **CP423 Final Project**
### **Search engine program implemented in Python.**

**Overview:** This program is a basic search engine implemented in Python. It includes different fundamental components we talked in class for building Indexing and Query Processing pipelines. The program starts from the command line using `python search_engine.py`. The user is prompted with the following message, and they can type the appropriate number in the console and press enter to select the option they want to do. The program will then prompt the user for the appropriate input and will run perform the selected task. The program will continue to prompt the user for input until they select the option to exit the program.

    Select an option:
    1- Collect new documents.
    2- Index documents.
    3- Search for a query.
    4- Train ML classifier.
    5- Predict a link.
    6- Your story!
    7- Exit


The program collects documents from websites that fit under any one of the three program topics of Astronomy, Economy, and Health. The list of source websites can be modified in `source.txt`.

---


## **Program Instructions**

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Run Program
```Bash
python search_engine.py
```
Running the program displays a brief welcome message. After which, the user enters the option menu event loop, as shown below.

Option menu console output:

    Select an option:
    1- Collect new documents.
    2- Index documents.
    3- Search for a query.
    4- Train ML classifier.
    5- Predict a link.
    6- Your story!
    7- Exit


### 1- Collect new documents

The program collects new documents for each of our topics in our search engine. We use the source.txt file to begin crawling the links for new pages on the website as our trusted source. This is done for the `TOPIC_DOCUMENT_LIMIT` constant in `constants.py`. The collected documents are then placed in the `data` directory and then in the subdirectory according to the topic. When a document is collected the url of the document is then hashed and the filename is the hash as the filename as a text file. The hash to url map is also saved for each saved document. The way the crawler works it iterates over all the sources to try and have a nicely distributed documents collected from the trusted sources.

Output:

    Collecting new documents...
    Crawling Astronomy...
    Finished crawling Astronomy
    Crawling Economy...
    Finished crawling Economy
    Crawling Health...
    Finished crawling Health
After the task is complete, the program re-enters the option menu event loop.


### 2- Index documents


The program iterates over the `data` directory and all of the topic subdirectories and indexes all of the tokens after tokenizing. The program then saves the hash of the document in a list within a dictionary of the token mapping back in the inverted index. Since the actual hashes are very long a short form is used for each hash which can be found in `mapping.json` located in the `Index Data` directory. The program then saves the inverted index (`inverted_index.json`)and the mapping to the `Index Data` directory. After this the program uses the `crawler.log` to create a mapping of the hashes to the url it came from to display when querying. The outline of the occurences field of the inverted index includes all of the occurences of the token within documents as well as the token's frequency in the document. The token's soundex code is also stored in the inverted index as well for each token to be used for spell correction in the querying portion.

Output:

    Indexing documents...
    Loading existing inverted index...
    Inverted index saved to Index Data\inverted_index.json
    Mapping saved to Index Data\mapping.json
    Hash to url mapping saved to Index Data\hash_to_url.json
After the task is complete, the program re-enters the option menu event loop.


### 3- Search for a query

The program prompts the user to enter a search query. The users query will be spell corrected using soundex similarity, and edit distance and term frequency are used for fallback conditions. The program will then perform the term-at-a-time algorithm and use the inverted index to display the top 3 highest ranked documents (ranked using cosine similarity) that contain any of the spell corrected query terms. A link to the web page is given for each document, as well as a snippet of matching text blocks with query terms highlighted in unique colours.

Example output (screenshot):

![search for a query - example output](https://user-images.githubusercontent.com/19508210/233170942-b92b08e9-a3ce-44ae-84a1-1b19eed3c47e.png)

After the task is complete, the program re-enters the option menu event loop.


### 4- Train ML classifier

The model to be trained is KNN, we found KNN to perform the best when predicting new links and classifying the link to pertain to a certain topic. We used cross-validation and classification metrics to support this decision as well. The program first collects all the documents and puts them into a pandas dataframe with their text contents. The text contents are then preprocessed by tokenizing and removing stopwords. The data is then split into train and test with 80% for training and 20% for testing. After this the TFIDF vectorizer is then fitted to the training data and both the training and test set are vectorized. The vectorizer is then saved for future use in predictions. The model is then trained using a grid search to identify the best parameters, in this case the neighbourhood size. Once this is done the model is then tested on making prediction with the test set and the metrics are returned.

Output:

    Indexing documents...
    Loading existing inverted index...
    Inverted index saved to Index Data\inverted_index.json
    Mapping saved to Index Data\mapping.json
After the task is complete, the program re-enters the option menu event loop.


### 5- Predict a link

The program prompts the user to input a link, which is then crawled to extract its textual content. The content is then vectorized using the same vectorizer that was utilized during classifier training. The program then loads the saved classifier and uses it to make a classification prediction, printing each of the three program topics along with a corresponding probability/confidence score indicating the likelihood of the link belonging to that topic.

Output:

    Predicting a link...
    Enter link: http://sten.astronomycafe.net/2023/03/
    Predictions
    Astronomy: 100.0%
    Health: 0.0%
    Economy: 0.0%
After the task is complete, the program re-enters the option menu event loop.


### 6- Your story!

The program displays a brief message containing general information about the search engine and details about how we have applied our knowledge from the course to build the various components of this system.

After the task is complete, the program re-enters the option menu event loop.


### 7- Exit

The option menu event loop will conclude, exiting the progam.

Output:

    Exiting search engine...
