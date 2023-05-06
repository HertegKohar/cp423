import unittest
import json
import os
from Constants.constants import (
    MAPPING_PATH,
    HASH_TO_URL_PATH,
    TOPICS,
    DOCUMENTS_PATH,
    INVERTED_INDEX_PATH,
    EXTENSION,
)


# To run use python -m unittest discover -s testing -v
class TestFiles(unittest.TestCase):
    def test_files(self):
        files = set()
        # Check to see if all files are unique
        for topic in TOPICS:
            for file in os.listdir(os.path.join(DOCUMENTS_PATH, topic)):
                self.assertTrue(file not in files)
                files.add(file)
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # Check to see all files in mapping
        print("Number of documents", len(files))
        for file in files:
            hash_ = file.split(".")[0].strip()
            self.assertTrue(hash_ in mapping, msg=f"{hash_} not in mapping")


class TestInvertedIndex(unittest.TestCase):
    def test_inverted_index(self):
        with open(INVERTED_INDEX_PATH, "r", encoding="utf-8") as f:
            inverted_index = json.load(f)
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        reversed_mapping = {v: k for k, v in mapping.items()}
        files = {}
        for topic in TOPICS:
            files[topic] = set()
            for file in os.listdir(os.path.join(DOCUMENTS_PATH, topic)):
                files[topic].add(file)
        # Check to see if all files in inverted index exist
        for token in inverted_index:
            for occurence in inverted_index[token]["occurences"]:
                self.assertTrue(
                    reversed_mapping[occurence[0]] + EXTENSION in files[occurence[2]]
                )


class TestMappings(unittest.TestCase):
    def test_mappings(self):
        # Check to see if all hashes in mapping are in hash_to_url
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        with open(HASH_TO_URL_PATH, "r", encoding="utf-8") as f:
            hash_to_url = json.load(f)
        print("Mapping lengths:", len(mapping), len(hash_to_url))
        # self.assertEqual(len(mapping), len(hash_to_url), msg="Lengths not equal")
        failed = False
        not_included = []
        for key in mapping:
            if key not in hash_to_url:
                failed = True
                not_included.append(key)
        if failed:
            with open("missing.txt", "w") as f:
                f.write("\n".join(not_included))
        if failed:
            self.fail("keys not in hash_to_url, see missing.txt, run remove_missing.py")
        # self.assertTrue(key in hash_to_url, msg=f"{key} not in hash_to_url")
