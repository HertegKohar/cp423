import os
import nltk
import warnings

# Filtering warnings
warnings.filterwarnings("ignore")

# Download punkt and stopwords if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading punkt")
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    print("Downloading stopwords")
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

# Crawling
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
LOGGER_PATH = "crawler.log"

SOURCE_PATH = "source.txt"

TOPIC_DOCUMENT_LIMIT = 10

COOLDOWN = 429

DOCUMENTS_PATH = "data"

TOPICS = ["Astronomy", "Health", "Economy"]

STORY_PATH = "story.txt"
with open(STORY_PATH, "r", encoding="utf-8") as f:
    STORY_TEXT = f.read()

# Querying
N_DOCUMENTS = 3

EXTENSION = ".txt"

# Classifying
TOPICS_MAP = {"Astronomy": 0, "Health": 1, "Economy": 2}
MODELS_PATH = "Models"
MODEL_PATH = os.path.join(MODELS_PATH, "classifier.joblib")
TFID_PATH = os.path.join(MODELS_PATH, "tfidf.joblib")
GAUSSIAN_MODEL = "GaussianNB"
PRODUCTION = True

# Indexing
INDEX_DATA_PATH = "Index Data"
INVERTED_INDEX_PATH = os.path.join(INDEX_DATA_PATH, "inverted_index.json")
MAPPING_PATH = os.path.join(INDEX_DATA_PATH, "mapping.json")
HASH_TO_URL_PATH = os.path.join(INDEX_DATA_PATH, "hash_to_url.json")
INDENT = 4
