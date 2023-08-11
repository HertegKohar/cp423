"""
Author: Herteg Kohar
"""
from Constants.constants import TOPICS, TOPIC_DOCUMENT_LIMIT
from Crawler.crawler import crawl_all_topics, crawl_new_link
from Index.index import update_inverted_index
from Classify.classify import training_pipeline, predict_new_text
from Query.query import query_documents

if __name__ == "__main__":
    print("Collecting new documents...")
    crawl_all_topics(TOPIC_DOCUMENT_LIMIT)

    print("Indexing documents...")
    update_inverted_index(TOPICS)

    print("Training ML Classifier...")
    training_pipeline()
