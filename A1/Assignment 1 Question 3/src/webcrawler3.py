"""
Authors:
    Kelvin Kellner
    Herteg Kohar
"""
import requests
import argparse
import hashlib
import datetime
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Regex to match URLs
URL_REGEX = url_regex = re.compile(
    r"(http|https)://[\w-]+(.[\w-]+)+([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
)
# Regex to match HTML tag
HTML_TAG_REGEX = re.compile(r"<[^>]*>")
# Regex to match any token
TOKEN_REGEX = re.compile(r"\S+")

# Path to the log file
LOGGER_PATH = "crawler3.log"

# HTTP response code for too many requests
COOLDOWN = 429

# Set of files already seen
FILES = set()

for file in os.listdir("."):
    if file.endswith(".txt"):
        FILES.add(file.split(".txt")[0])

def log(message):
    """
    Log the download of a URL to a file.
    -----
    Args:
        message (str): A message which follows the assignment format of <H,URL,Download
        DateTime, HTTP Response Code>
    """
    with open(LOGGER_PATH, "a", encoding="utf-8") as f:
        f.write(message)

def hash_and_save(url, status_code, content, rewrite):
    """
    Use hashlib to hash the URL and save the content to a file.
    -----
    Args:
        url (str): Current URL being crawled
        status_code (int): HTTP response code from GET request
        content (str): HTML content of the URL
        rewrite (bool): The rewrite flag from the command line to rewrite the HTML content within the .txt if already seen
    Returns:
        filename (str): The name of the file the HTML content was saved to (<hash>.txt)
    """
    hash_object = hashlib.sha256(url.encode())
    hex_dig = hash_object.hexdigest()
    filename = hex_dig + ".txt"
    log(f"{hex_dig} {url} {datetime.datetime.now()} {status_code}\n")
    if not rewrite and hex_dig in FILES:
        return filename
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    FILES.add(hex_dig)
    return filename

def plot_function_2d(function_outputs, max_score):
    """
    Displays a 2d plot of the function values
    -----
    Args:
        function_outputs: 2d array of function outputs for each combination of i and j input values
        max_score: Tuple (low: int, high: int) for the lowest and highest score of the optimization function
    """
    # Resource: https://stackoverflow.com/a/54088910
    n = len(function_outputs)
    # generate two 2d grids for the x & y bounds
    y, x = np.array(range(n)), np.array(range(n))
    z = function_outputs
    z_min, z_max = 0, max_score
    # set up appearance of the plot
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, z, cmap='YlOrRd', vmin=z_min, vmax=z_max)
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_title('Content Block Scores')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    # display the plot
    plt.show()

def optimize_sequence_mapping(sequence_mapping):
    """
    Optimizes the sequence mapping to find the optimal content block,
    the goal is to find the distribution with the most '0's inside and
    the most '1's inside. Return span uses token positions, not chars.
    -----
    Args:
        sequence_mapping: List of tuples (binary_mapping: int, start: int, end: int) for mapping each token as described
    Returns:
        optimal_tokens_span: Tuple (start: int, end: int) for optimal start and end token of the content block
        function_outputs: 2d array of function outputs for each combination of i and j input values
        max_score: Tuple (low: int, high: int) for the lowest and highest score of the optimization function
    """
    #
    # Brute forcing all combinations of i and j is too slow.
    #
    # Algorithm works based on the following observation:
    # each time i == j the score is the sum of the sequence
    # each time j moves forward a bit, the score is:
    #   the previous score + 1 if the token is 0, or the previous score -1 if the token is 1
    # So, we can calculate the score for each i and j combination by iterating through the sequence and
    # adding or subtracting 1 for each token, and resetting the score each time i moves forward a bit.
    # We keep track of the max score and optimal span along the way to get the correct answer in the end.
    #
    n_tokens = len(sequence_mapping)
    function_outputs = [[0 for _ in range(n_tokens)] for _ in range(n_tokens)]
    score_when_i_equals_j = sum([sequence_mapping[i][0] for i in range(n_tokens)])
    max_score = score_when_i_equals_j # assume for now
    optimal_tokens_span = (0,0)
    for i in range(0, n_tokens):
        current_score = score_when_i_equals_j
        for j in range(i+1, n_tokens):
            # if the 'consumed' bit: is 0 --> score += 1, if 1 --> score -= 1
            if sequence_mapping[j-1][0] == 0:
                current_score += 1
            else:
                current_score -= 1
            if current_score > max_score:
                max_score = current_score
                optimal_tokens_span = (i, j)
            function_outputs[j][i] = current_score
    return optimal_tokens_span, function_outputs, max_score

def generate_binary_sequence_mapping(token_spans, tag_spans):
    """
    Generates a list mapping of spanning token positions that describe
    whether each token is part of an HTML opening or closing tag (1),
    or the token is text content between or outside of any tags (0).
    -----
    Args:
        token_spans: List of tuples (start: int, end: int) for the span of each token in the text
        tag_spans: List of tuples (start: int, end: int) for the span of each opening or closing HTML tag in the text
    Returns:
        sequence_mapping: List of tuples (binary_mapping: int, start: int, end: int) for mapping each token as described
    """
    sequence_mapping = []
    tag_index = 0
    token_index = 0
    while token_index < len(token_spans):
        current_token_is_a_tag = False
        # if all HTML tags have already been processed, remaining tokens are mapped '0'
        if tag_index >= len(tag_spans):
            sequence_mapping.append((0, token_spans[token_index][0], token_spans[token_index][1]))
            token_index += 1
            continue
        # if current token is past current tag in position, move to next tag
        if token_spans[token_index][0] > tag_spans[tag_index][1]:
            tag_index += 1
            continue
        # if current token's span collides with current tag's span, then current token will be mapped '1' in next step
        if (token_spans[token_index][1] >= tag_spans[tag_index][0] and token_spans[token_index][1] <= tag_spans[tag_index][1]) \
            or (token_spans[token_index][0] <= tag_spans[tag_index][1] and token_spans[token_index][0] >= tag_spans[tag_index][0]):
            current_token_is_a_tag = True
        # map current token as either '1' or '0' depending on previous if statement
        sequence_mapping.append((1 if current_token_is_a_tag else 0, token_spans[token_index][0], token_spans[token_index][1]))
        token_index += 1
    return sequence_mapping
    
def create_list_of_tag_spans(document):
    """
    Produce a list of spanning (start and end character) positions of all HTML tags in the text.
    -----
    Args:
        document (str): The HTML content of the page
    Returns:
        tag_spans: List of tuples (start: int, end: int) for the span of each opening or closing HTML tag in the text
    """
    tag_spans = []
    for result in HTML_TAG_REGEX.finditer(document):
        tag_spans.append(result.span())
    # merge neighbouring spans, e.g. [(0, 29), (29, 50)] => [(0, 50)]
    i = 0
    while i+1 < len(tag_spans):
        # if current span's end == next span's start
        if tag_spans[i][1] == tag_spans[i+1][0]:
            # merge into single span
            tag_spans[i] = (tag_spans[i][0], tag_spans[i+1][1])
            tag_spans.pop(i+1)
        else:
            i+=1
    return tag_spans

def create_list_of_token_spans(document):
    """
    Produce a list of spanning (start and end character) positions of all tokens in the text.
    -----
    Args:
        document (str): The HTML content of the page
    Returns:
        token_spans: List of tuples (start: int, end: int) for the span of each token in the text
    """
    token_spans = []
    for result in TOKEN_REGEX.finditer(document):
        token_spans.append(result.span())
    return token_spans

def crawl(url, rewrite=False):
    """
    Crawl the URL to retrieve the HTML content of the page.
    -----
    Args:
        url (str): The current URL being crawled
        rewrite (bool): Flag to rewrite the content files for the URL if recorded
    Returns:
        document (str): The HTML content of the page
        filename (str): The name of the file the HTML content was saved to (<hash>.txt)
    """
    r = requests.get(url)
    if r.status_code == COOLDOWN:
        print("Too many requests")
        return
    document = r.text
    filename = hash_and_save(url, r.status_code, r.text, rewrite)
    return document, filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command Line Web Crawler")
    parser.add_argument("initialURL", type=str, help="The initial URL to crawl")
    args = parser.parse_args()
    
    # 1. crawl webpage to get text
    document, filename = crawl(args.initialURL)
    if document is None:
        print('Program is exiting prematurely as HTML content of the page could not be retrieved.')
        exit()

    # 2. create list of token and tag spans
    token_spans = create_list_of_token_spans(document)
    tag_spans = create_list_of_tag_spans(document)

    # 3. create binary sequence mapping using token and tag spans
    sequence_mapping = generate_binary_sequence_mapping(token_spans, tag_spans)
    sequence_as_string = ''.join([str(token[0]) for token in sequence_mapping])
    print(f'\nSEQUENCE MAPPING AS STRING\n\n{sequence_as_string}\n')

    # 4. apply any filters to modify the sequence as needed
    # (optionally apply filters to the sequence mapping here)

    # 5. run binary sequence through optimization function to find the best start and end positions for content block
    optimal_tokens_span, function_outputs, max_score = optimize_sequence_mapping(sequence_mapping)
    optimal_characters_span = (token_spans[optimal_tokens_span[0]][0], token_spans[optimal_tokens_span[1]][1])
    print('\nTHE OPTIMAL CONTENT BLOCK SPANS\n')
    print(f'TOKENS: {optimal_tokens_span[0]} to {optimal_tokens_span[1]} (of {len(token_spans)})')
    print(f'CHARACTERS: {optimal_characters_span[0]} to {optimal_characters_span[1]} (of {len(document)})')

    # 6. use optimal start and end positions to get content block from the document
    content_block = document[optimal_characters_span[0]:optimal_characters_span[1]]

    # 7. output content block to text file
    with open(f'{filename}', 'w', encoding='utf-8') as f:
        f.write(content_block)
    print(f'\nContent block saved to {filename}.')

    # 8. display optimization function outputs in a plot
    plot_function_2d(function_outputs, max_score)
    # plot_function_3d(function_outputs, optimal_tokens_span[-1])

    # All done :)
    print('\nProgram completed successfully.\n')
