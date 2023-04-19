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
Output:

    Select an option:
    1- Collect new documents.
    2- Index documents.
    3- Search for a query.
    4- Train ML classifier.
    5- Predict a link.
    6- Your story!
    7- Exit


### 1- Collect new documents

The program will... for each of the 3 topics...  
// TODO: Add description

Output:

    Collecting new documents...
    Crawling Astronomy...
    Finished crawling Astronomy
    Crawling Economy...
    Finished crawling Economy
    Crawling Health...
    Finished crawling Health
Then, the program will return to the option selection menu (event loop).


### 2- Index documents

The program will...   
// TODO: Add description

Output:

    Indexing documents...
    Loading existing inverted index...
    Inverted index saved to Index Data\inverted_index.json
    Mapping saved to Index Data\mapping.json
Then, the program will return to the option selection menu (event loop).


### 3- Search for a query

The program will prompt the user to enter a search query. The users query will be spell corrected using soundex similarity, and edit distance and term frequency are used for fallback conditions. The program will then perform the term-at-a-time algorithm and use the inverted index to display the top 3 highest ranked documents (ranked using cosine similarity) that contain any of the spell corrected query terms. A link to the web page is given for each document, as well as a snippet of matching text blocks with query terms highlighted in unique colours.

Example output (screenshot):

![screenshot01](https://user-images.githubusercontent.com/19508210/233170942-b92b08e9-a3ce-44ae-84a1-1b19eed3c47e.png)

Then, the program will return to the option selection menu (event loop).


### 4- Train ML classifier

The program will...   
// TODO: Add description

Output:

    Indexing documents...
    Loading existing inverted index...
    Inverted index saved to Index Data\inverted_index.json
    Mapping saved to Index Data\mapping.json
Then, the program will return to the option selection menu (event loop).


### 5- Predict a link

The program will...   
// TODO: Add description

Output:

    Indexing documents...
    Loading existing inverted index...
    Inverted index saved to Index Data\inverted_index.json
    Mapping saved to Index Data\mapping.json
Then, the program will return to the option selection menu (event loop).


### 6- Your story!

The program will display a brief message containing general information about the search engine and details about how we have applied our knowledge from the course to build the various components of this system.

Output:

// TODO: either add story here, or remove the output block from this section  
Then, the program will return to the option selection menu (event loop).


### 7- Exit

The option menu event loop will conclude, exiting the progam.

Output:

    Exiting search engine...
