"""
Author: Herteg Kohar
"""
from Constants.constants import (
    HEADERS,
    LOGGER_PATH,
    SOURCE_PATH,
    TOPIC_DOCUMENT_LIMIT,
    COOLDOWN,
    DOCUMENTS_PATH,
    TOPICS,
    HASH_TO_URL_PATH,
)
import requests
from random import shuffle
from bs4 import BeautifulSoup
from datetime import datetime
import hashlib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict, deque
from urllib.parse import urlparse, urljoin
import justext
import os
import json


# from requests_html import HTMLSession

FILES = set()

for topic in TOPICS:
    for file in os.listdir(os.path.join(DOCUMENTS_PATH, topic)):
        FILES.add(os.path.join(DOCUMENTS_PATH, topic, file))

STOP_WORDS = set(stopwords.words("english"))


def log(message):
    with open(LOGGER_PATH, "a", encoding="utf-8") as f:
        f.write(message)
        f.write("\n")


def hash_and_save(topic, url, content, hash_to_url_dict):
    hash_object = hashlib.sha256(url.encode())
    hex_dig = hash_object.hexdigest()
    file_path = os.path.join(DOCUMENTS_PATH, topic, f"{hex_dig}.txt")
    if file_path in FILES:
        return False
    content = content.lower()
    paragraphs = justext.justext(content, justext.get_stoplist("English"))
    text_to_write = ""
    for paragraph in paragraphs:
        if not paragraph.is_boilerplate:
            text = paragraph.text
            words = word_tokenize(text)
            words = [word for word in words if word not in STOP_WORDS]
            text = " ".join(words)
            text_to_write += text
    if len(text_to_write) == 0:
        return False
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text_to_write)
    log(f"{topic} {url} {hex_dig} {datetime.now()}")
    FILES.add(file_path)
    hash_to_url_dict[hex_dig] = url
    return True


def create_topics_dict(path):
    topics_and_urls = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        # Skip the header
        next(f)
        for line in f:
            topic, url = line.split(",")
            topic = topic.strip()
            url = url.strip()
            topics_and_urls[topic].append(url)
    return topics_and_urls


def crawl_new_link(url):
    r = requests.get(url, headers=HEADERS)
    if r.status_code == COOLDOWN:
        return
    text = ""
    paragraphs = justext.justext(r.text, justext.get_stoplist("English"))
    for paragraph in paragraphs:
        if not paragraph.is_boilerplate:
            text += paragraph.text
    if len(text) == 0:
        return
    with open("predicted_link_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text


def source_switch_crawl(topic, urls, limit):
    print(f"Crawling {topic}...")
    count = 0
    hash_to_url_dict = {}
    base_urls = defaultdict(list)
    for url in urls:
        base_urls[urlparse(url).netloc].append(url)
    visited = set()
    base_urls_list = list(base_urls.keys())
    shuffle(base_urls_list)
    base_urls_deque = deque(base_urls_list)
    while count != limit:
        base_url = base_urls_deque[0]
        base_urls_deque.rotate(-1)
        if len(base_urls[base_url]) == 0:
            print(f"No more urls for {base_url}")
            continue
        shuffle(base_urls[base_url])
        url = base_urls[base_url].pop(0)
        r = requests.get(url, headers=HEADERS)
        if r.status_code == COOLDOWN:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")
            absolute_url = urljoin(url, href)
            if (
                href is not None
                and urlparse(absolute_url).netloc in base_urls
                and absolute_url not in visited
            ):
                base_urls[urlparse(absolute_url).netloc].append(absolute_url)
        visited.add(url)
        saved = hash_and_save(topic, url, r.text, hash_to_url_dict)
        if saved:
            count += 1
    with open(HASH_TO_URL_PATH, "r+", encoding="utf-8") as f:
        original = json.load(f)
        original.update(hash_to_url_dict)
        f.seek(0)
        f.truncate()
        json.dump(original, f)
    print(f"Finished crawling {topic}")


def crawl(topic, urls, limit):
    print(f"Crawling {topic}...")
    count = 0
    hash_to_url_dict = {}
    queue = urls.copy()
    base_urls = {urlparse(url).netloc for url in urls}
    visited = set()
    while count != limit and len(queue) != 0:
        url = queue.pop(0)
        r = requests.get(url, headers=HEADERS)
        if r.status_code == COOLDOWN:
            break
        soup = BeautifulSoup(r.text, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")
            absolute_url = urljoin(url, href)
            if (
                href is not None
                and urlparse(absolute_url).netloc in base_urls
                and absolute_url not in visited
            ):
                queue.append(absolute_url)
        visited.add(url)
        saved = hash_and_save(topic, url, r.text)
        if saved:
            count += 1
    with open(HASH_TO_URL_PATH, "r+", encoding="utf-8") as f:
        original = json.load(f)
        original.update(hash_to_url_dict)
        f.seek(0)
        f.truncate()
        json.dump(original, f)
    print(f"Finished crawling {topic}")


# To try and get to pages that require javascript to load
# def crawl_session(topic, urls, limit):
#     print(f"Crawling {topic}...")
#     count = 0
#     queue = urls.copy()
#     base_urls = {urlparse(url).netloc for url in urls}
#     visited = set()
#     session = HTMLSession()
#     while count != limit and len(queue) != 0:
#         url = queue.pop(0)
#         r = session.get(url)
#         r.html.render(wait=15)
#         if r.status_code == COOLDOWN:
#             break
#         for link in r.html.links:
#             absolute_url = urljoin(url, link)
#             if (
#                 urlparse(absolute_url).netloc in base_urls
#                 and absolute_url not in visited
#             ):
#                 queue.append(absolute_url)
#         visited.add(url)
#         saved = hash_and_save(topic, url, r.html.html)
#         if saved:
#             count += 1
#         print("Documents collected", count)
#         print(queue)
#     session.close()
#     print(f"Finished crawling {topic}")


def crawl_all_topics(limit):
    topics_and_urls = create_topics_dict(SOURCE_PATH)
    for topic, urls in topics_and_urls.items():
        # crawl(topic, urls, limit)
        source_switch_crawl(topic, urls, limit)


# Debugging
if __name__ == "__main__":
    crawl_all_topics(TOPIC_DOCUMENT_LIMIT)
