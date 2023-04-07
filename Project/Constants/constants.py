import os
import nltk
import warnings

warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

OPTIONS = """
Select an option:
1 - Collect new documents
2 - Index documents
3 - Search for query
4 - Train ML Classifier
5 - Predict a link
6 - Your Story!
7 - Exit
"""

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
LOGGER_PATH = "crawler.log"

SOURCE_PATH = "source.txt"

TOPIC_DOCUMENT_LIMIT = 10

COOLDOWN = 429

DOCUMENTS_PATH = "Documents"

TOPICS = ["Astronomy", "Health", "Economy"]

STORY_PATH = "story.txt"
with open(STORY_PATH, "r", encoding="utf-8") as f:
    STORY_TEXT = f.read()

# Querying
N_DOCUMENTS = 3

EXTENSION = ".txt"

DOCUMENTS_PATH = "Documents"

# Classifying
TOPICS_MAP = {"Astronomy": 0, "Health": 1, "Economy": 2}
MODELS_PATH = "Models"
MODEL_PATH = os.path.join(MODELS_PATH, "classifier.joblib")
TFID_PATH = os.path.join(MODELS_PATH, "tfidf.joblib")

# Indexing
INDEX_DATA_PATH = "Index Data"
INVERTED_INDEX_PATH = os.path.join(INDEX_DATA_PATH, "inverted_index.json")
MAPPING_PATH = os.path.join(INDEX_DATA_PATH, "mapping.json")
