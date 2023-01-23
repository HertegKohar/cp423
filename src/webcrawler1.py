"""
Author: Herteg Kohar
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
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*\.com)"
)

# Path to the log file
LOGGER_PATH = "crawler1.log"

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


def crawl(url, maxdepth, verbose, rewrite):
    """Crawl the URLs recursively from each page explored.

    Args:
        url (str): The current URL being crawled
        maxdepth (int): The maximum depth of the crawling (how many links deep)
        verbose (bool): Flag to output the URL and current depth
        rewrite (bool): Flag to rewrite the content files for the URL if recorded
    """
    if verbose:
        print(url, maxdepth)
    if maxdepth == 0:
        return
    r = requests.get(url)
    hash_and_save(url, r.status_code, r.text, rewrite)
    soup = BeautifulSoup(r.text, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href")
        if href is not None and URL_REGEX.match(href):
            crawl(href, maxdepth - 1, verbose, rewrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command Line Web Crawler")
    parser.add_argument(
        "--maxdepth", help="The maximum depth to crawl", type=int, default=1
    )
    parser.add_argument(
        "--rewrite",
        help="Rewrite the content files for the URL if recorded",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--verbose",
        help="Verbose output",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument("initialURL", type=str, help="The initial URL to crawl")
    args = parser.parse_args()
    # Clear the log
    with open(LOGGER_PATH, "w", encoding="utf-8") as f:
        pass
    crawl(args.initialURL, args.maxdepth, args.verbose, args.rewrite)
