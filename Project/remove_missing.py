import os
from Constants.constants import DOCUMENTS_PATH, TOPICS, EXTENSION

missing = []
with open("missing.txt") as f:
    for line in f:
        missing.append(line.strip())
    print(len(missing))

for file in missing:
    for topic in TOPICS:
        if os.path.exists(os.path.join(DOCUMENTS_PATH, topic, file + EXTENSION)):
            os.remove(os.path.join(DOCUMENTS_PATH, topic, file + EXTENSION))
