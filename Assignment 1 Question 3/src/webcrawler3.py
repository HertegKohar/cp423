"""
Authors:
    Kelvin Kellner
    Herteg Kohar
"""
import requests
import argparse
from bs4 import BeautifulSoup
import hashlib
import datetime
import re
import os

# Regex to match URLs
URL_REGEX = url_regex = re.compile(
    r"(http|https)://[\w-]+(.[\w-]+)+([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
)
# Regex to match HTML tag
HTML_TAG_REGEX = re.compile(r"<[^>]*>")
# Regex to match any word except for "1"
TOKEN_REGEX = re.compile(r"(?:(?!\b1\b)\w+)")
NON_NUMERIC_REGEX = re.compile(r"[^0-9]")

# Path to the log file
LOGGER_PATH = "crawler3.log"

# HTTP response code for too many requests
COOLDOWN = 429

# Set of files already seen
FILES = set()

for file in os.listdir("."):
    if file.endswith(".txt"):
        FILES.add(file.split(".txt")[0])


def log(message):
    """Log the download of a URL to a file.

    Args:
        message (str): A message which follows the assignment format of <H,URL,Download
        DateTime, HTTP Response Code>
    """
    with open(LOGGER_PATH, "a", encoding="utf-8") as f:
        f.write(message)


def binary_sequence_from_contents(document):
    """
    TODO: write docstring
    """
    print("A")
    print(document)
    codified_content = re.sub(HTML_TAG_REGEX, "1", document)
    print("B")
    print(codified_content)
    codified_content = re.sub(TOKEN_REGEX, "0", codified_content)
    print("C")
    print(codified_content)
    codified_content = re.sub(NON_NUMERIC_REGEX, "", codified_content)
    print("D")
    print(codified_content)
    # for link in soup.find_all("a"):
    #     href = link.get("href")
    #     if href is not None and URL_REGEX.match(href):
    #         crawl(href, maxdepth - 1, verbose, rewrite)
    #     elif href is not None:
    #         crawl(url + href, maxdepth - 1, verbose, rewrite)

def hash_and_save(url, status_code, content, rewrite):
    """Use hashlib to hash the URL and save the content to a file.

    Args:
        url (str): Current URL being crawled
        status_code (int): HTTP response code from GET request
        content (str): HTML content of the URL
        rewrite (bool): The rewrite flag from the command line to rewrite the HTML content within the .txt if already seen
    """
    hash_object = hashlib.sha256(url.encode())
    hex_dig = hash_object.hexdigest()
    filename = hex_dig + ".txt"
    log(f"{hex_dig} {url} {datetime.datetime.now()} {status_code}\n")
    if not rewrite and hex_dig in FILES:
        return
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    FILES.add(hex_dig)


def crawl(url, rewrite=False):
    """Crawl the URLs recursively from each page explored.
    TODO: update this docstring
    Args:
        url (str): The current URL being crawled
        rewrite (bool): Flag to rewrite the content files for the URL if recorded
    """
    r = requests.get(url)
    hash_and_save(url, r.status_code, r.text, rewrite)
    if r.status_code == COOLDOWN:
        print("Too many requests")
        return
    binary_sequence_from_contents(r.text)
    # soup = BeautifulSoup(r.text, "html.parser")
    # for link in soup.find_all("a"):
    #     href = link.get("href")
    #     if href is not None and URL_REGEX.match(href):
    #         crawl(href, maxdepth - 1, verbose, rewrite)
    #     elif href is not None:
    #         crawl(url + href, maxdepth - 1, verbose, rewrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command Line Web Crawler")
    parser.add_argument("initialURL", type=str, help="The initial URL to crawl")
    args = parser.parse_args()
    # Clear the log
    with open(LOGGER_PATH, "w", encoding="utf-8") as f:
        pass
    crawl(args.initialURL)
