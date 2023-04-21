import unittest
import json
from Constants.constants import MAPPING_PATH, HASH_TO_URL_PATH


class TestIndices(unittest.TestCase):
    def test_mappings(self):
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        with open(HASH_TO_URL_PATH, "r", encoding="utf-8") as f:
            hash_to_url = json.load(f)

        self.assertEqual(len(mapping), len(hash_to_url))
        for key in mapping:
            self.assertTrue(key in hash_to_url)
