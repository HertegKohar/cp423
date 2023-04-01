import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

if __name__ == "__main__":
    base_url = input("Enter a link: ")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    r = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href")
        absolute_url = urljoin(base_url, href)
        print(absolute_url)
    with open("test_link_content.txt", "w", encoding="utf-8") as f:
        f.write(soup.text)
