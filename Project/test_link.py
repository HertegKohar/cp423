import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from Constants.constants import HEADERS

if __name__ == "__main__":
    base_url = input("Enter a link: ")
    headers = HEADERS
    r = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href")
        absolute_url = urljoin(base_url, href)
        print(absolute_url)
    with open("test_link_content.txt", "w", encoding="utf-8") as f:
        f.write(soup.text)
