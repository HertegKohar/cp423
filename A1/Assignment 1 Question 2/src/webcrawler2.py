"""
Authors: 
    Herteg Kohar 
    Bryan Gadd
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.options import Options
from bs4 import BeautifulSoup
import hashlib
import argparse
import json
from time import sleep

PREFIX_URL = "https://scholar.google.ca"


def hash_and_save_html_content(
    url,
    content,
):
    """Use hashlib to hash the URL and save the content to a file.

    Args:
        url (str): Current URL being crawled
        content (str): HTML content of the URL
    """
    hash_object = hashlib.sha256(url.encode())
    hex_dig = hash_object.hexdigest()
    filename = hex_dig + ".txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def hash_and_save_json(url, dictionary):
    """Saves the dictionary to a JSON file as well as naming the file by the hash of the URL.

    Args:
        url (str): Current URL of Google Scholar being crawled
        dictionary (dict): Python dictionary containing the information of the researcher
    """
    hash_object = hashlib.sha256(url.encode())
    hex_dig = hash_object.hexdigest()
    filename = hex_dig + ".json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dictionary, f)


def parse_page(content):
    """Parses the page content into a python dictionary in order to store the information.

    Args:
        content (str): The HTML content of the page

    Returns:
        dict: The dictionary containing the information of the researcher to be stored in a JSON file
    """
    page_info = {}
    soup = BeautifulSoup(content, "html.parser")
    name = soup.find("div", id="gsc_prf_in").text
    page_info["researcher_name"] = name
    info = soup.find_all("div", class_="gsc_prf_il")
    page_info["researcher_institution"] = info[0].text
    page_info["researcher_caption"] = info[1].text
    image_div = soup.find("div", id="gsc_prf_pua")
    image_url = image_div.find("img")["src"]
    page_info["researcher_image_url"] = image_url
    citations_table = soup.find("table", id="gsc_rsb_st")
    citations = citations_table.find_all("td", class_="gsc_rsb_std")
    page_info["researcher_citations"] = {
        "all": citations[0].text,
        "since2018": citations[1].text,
    }
    page_info["researcher_hindex"] = {
        "all": citations[2].text,
        "since2018": citations[3].text,
    }
    page_info["researcher_i10index"] = {
        "all": citations[4].text,
        "since2018": citations[5].text,
    }
    page_info["researcher_papers"] = []
    papers_table = soup.find("table", id="gsc_a_t")
    for row in papers_table.find_all("tr"):
        paper = {}
        cells = row.find_all("td")
        if cells == []:
            continue
        paper["paper_title"] = cells[0].find("a", class_="gsc_a_at").text.strip()
        authors_and_journal = cells[0].find_all("div", class_="gs_gray")
        paper["paper_authors"] = authors_and_journal[0].text.strip()
        paper["paper_journal"] = authors_and_journal[1].text.strip()
        if cells[1].find("a", class_="gsc_a_ac gs_ibl") is not None:
            paper["paper_citedby"] = (
                cells[1].find("a", class_="gsc_a_ac gs_ibl").text.strip()
            )
        else:
            paper["paper_citedby"] = ""
        paper["paper_year"] = (
            cells[2].find("span", class_="gsc_a_h gsc_a_hc gs_ibl").text.strip()
        )
        page_info["researcher_papers"].append(paper)
    page_info["researcher_coauthors"] = []
    coauthors = soup.find_all("span", class_="gsc_rsb_a_desc")
    for span in coauthors:
        coauthorJson = {}
        coauthorJson["coauthor_name"] = span.find("a").get_text()
        coauthorJson["coauthor_title"] = span.find(
            "span", class_="gsc_rsb_a_ext"
        ).get_text()
        coauthorJson["coauthor_link"] = PREFIX_URL + span.find("a").get("href")
        page_info["researcher_coauthors"].append(coauthorJson)
    return page_info


def parse_page_show_more(url):
    """Uses selenium in order to navigate through the show more button if applicable. Then
    saves the page content to a file and calls the parse_page function to parse the page content and store
    the information in a JSON file.

    Args:
        url (str): The researcher's Google Scholar URL
    """
    edge_options = Options()
    edge_options.add_experimental_option("detach", True)
    edge_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=edge_options)
    driver.get(url)
    show_more_button = driver.find_element(by=By.ID, value="gsc_bpf_more")
    while show_more_button.is_enabled():
        show_more_button.click()
        sleep(1)
    hash_and_save_html_content(url, driver.page_source)
    page_info = parse_page(driver.page_source)
    hash_and_save_json(url, page_info)
    # driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("researcherURL", type=str, help="The initial URL to crawl")
    args = parser.parse_args()
    parse_page_show_more(args.researcherURL)
    print("Done!")
