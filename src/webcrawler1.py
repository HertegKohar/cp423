import requests
import argparse
from bs4 import BeautifulSoup
import hashlib
import datetime
import re
import os

URL_REGEX = url_regex = re.compile(
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*\.com)"
)

LOGGER_PATH = "crawler1.log"

FILES = set()

for file in os.listdir("."):
    if file.endswith(".txt"):
        FILES.add(file.split(".txt")[0])


def log(message):
    with open(LOGGER_PATH, "a", encoding="utf-8") as f:
        f.write(message)


def hash_and_save(url, status_code, content, rewrite):
    hash_object = hashlib.sha256(url.encode())
    hex_dig = hash_object.hexdigest()
    filename = hex_dig + ".txt"
    log(f"{hex_dig} {url} {datetime.datetime.now()} {status_code}\n")
    if not rewrite and hex_dig in FILES:
        return
    print("Rewriting", filename)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    FILES.add(hex_dig)


def crawl(url, maxdepth, verbose, rewrite):
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
