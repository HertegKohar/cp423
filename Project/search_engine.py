from crawler import crawl_all_topics

options = """
1 - Collect new documents
2 - Index documents
3 - Search for query
4 - Train ML Classifier
5 - Predict a link
6 - Your Story!
7 - Exit
"""

STORY_PATH = "story.txt"

TOPIC_DOCUMENT_LIMIT = 5

if __name__ == "__main__":
    print("Welcome to the search engine!")
    print("Options:")
    print(options)
    while (user_input := input("Enter option: ")) != "7":
        if user_input == "1":
            print("Collecting new documents...")
            crawl_all_topics(TOPIC_DOCUMENT_LIMIT)
        elif user_input == "2":
            print("Indexing documents...")
            raise NotImplementedError
        elif user_input == "3":
            print("Searching for query...")
            raise NotImplementedError
        elif user_input == "4":
            print("Training ML Classifier...")
            raise NotImplementedError
        elif user_input == "5":
            print("Predicting a link...")
            raise NotImplementedError
        elif user_input == "6":
            with open(STORY_PATH, "r") as f:
                print(f.read())

    print("Exiting search engine...")
