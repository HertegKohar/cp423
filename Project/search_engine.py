"""
Author: Herteg Kohar
"""
from crawler import crawl_all_topics, crawl_new_link
from index import update_inverted_index
from classify import training_pipeline, predict_new_text
from query import query_documents

OPTIONS = """
Select an option:
1 - Collect new documents
2 - Index documents
3 - Search for query
4 - Train ML Classifier
5 - Predict a link
6 - Your Story!
7 - Exit
"""

STORY_PATH = "story.txt"

TOPIC_DOCUMENT_LIMIT = 10
TOPICS = ["Astronomy", "Health", "Economy"]

if __name__ == "__main__":
    print("Welcome to the search engine!")
    print("Options:")
    print(OPTIONS)
    while (user_input := input("Enter option: ")) != "7":
        if user_input == "1":
            print("Collecting new documents...")
            crawl_all_topics(TOPIC_DOCUMENT_LIMIT)
        elif user_input == "2":
            print("Indexing documents...")
            update_inverted_index(TOPICS)
        elif user_input == "3":
            query = input("Enter query: ")
            print(f"Searching for query '{query}'...")
            query_documents(query)
        elif user_input == "4":
            print("Training ML Classifier...")
            training_pipeline()
        elif user_input == "5":
            print("Predicting a link...")
            link = input("Enter link: ")
            text = crawl_new_link(link)
            if text is None:
                print("Link is not valid or no text was able to be extracted.")
            else:
                predict_new_text(text)
        elif user_input == "6":
            with open(STORY_PATH, "r") as f:
                print(f.read())
        print(OPTIONS)

    print("Exiting search engine...")
