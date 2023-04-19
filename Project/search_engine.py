"""
Author: Herteg Kohar
"""
from Constants.constants import TOPICS, TOPIC_DOCUMENT_LIMIT, STORY_TEXT, OPTIONS
from Crawler.crawler import crawl_all_topics, crawl_new_link
from Index.index import update_inverted_index
from Classify.classify import training_pipeline, predict_new_text
from Query.query import query_documents


if __name__ == "__main__":
    print("Welcome to the search engine!")
    print(f"Topics: {', '.join(TOPICS)}")
    print("Options:")
    print(OPTIONS)
    user_input = input("Enter option: ")
    while user_input != "7":
        if user_input == "1":
            print("Collecting new documents...")
            crawl_all_topics(TOPIC_DOCUMENT_LIMIT)
        elif user_input == "2":
            print("Indexing documents...")
            update_inverted_index(TOPICS)
        elif user_input == "3":
            query = input("Enter query: ")
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
            print(STORY_TEXT)
        else:
            print("Invalid option!")
        print(OPTIONS)
        user_input = input("Enter option: ")

    print("Exiting search engine...")
