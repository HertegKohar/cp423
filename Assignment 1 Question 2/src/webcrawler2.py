"""
Author: Herteg Kohar 
"""
import requests
from bs4 import BeautifulSoup
import hashlib
import argparse
import json


def parse_page(content):
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
    page_info["researcher_coauthors"] = []
    coauthors = soup.find_all("span", class_="gsc_rsb_a_desc")
    for span in coauthors:
        coauthorJson = {}
        coauthorJson["coauthor_name"] = span.find("a").get_text()
        coauthorJson["coauthor_title"] = span.find("span", class_="gsc_rsb_a_ext").get_text()
        coauthorJson["coauthor_link"] = span.find("a").get("href")
        page_info["researcher_coauthors"].append(coauthorJson)
    
    return page_info


if __name__ == "__main__":
    with open(
        "Sample\_Guillermo Campitelli_ - _Google Scholar_.html"
    ) as file:
        content = file.read()
    page_info = parse_page(content)
    print(page_info)
