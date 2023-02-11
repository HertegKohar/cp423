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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import os

# Regex to match URLs
URL_REGEX = url_regex = re.compile(
    r"(http|https)://[\w-]+(.[\w-]+)+([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
)
# Regex to match HTML tag
HTML_TAG_REGEX = re.compile(r"<[^>]*>")
# Regex to match any word except for "1"
TOKEN_REGEX = re.compile(r"\S+") # old regex: re.compile(r"(?:(?!\b1\b)\w+)")
NON_NUMERIC_REGEX = re.compile(r"[^0-9]")
DOC_REGEX = re.compile(r".*")

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
    """Log the download of a URL to a file.

    Args:
        message (str): A message which follows the assignment format of <H,URL,Download
        DateTime, HTTP Response Code>
    """
    with open(LOGGER_PATH, "a", encoding="utf-8") as f:
        f.write(message)

def hash_and_save(url, status_code, content, rewrite):
    """Use hashlib to hash the URL and save the content to a file.

    Args:
        url (str): Current URL being crawled
        status_code (int): HTTP response code from GET request
        content (str): HTML content of the URL
        rewrite (bool): The rewrite flag from the command line to rewrite the HTML content within the .txt if already seen
    """
    hash_object = hashlib.sha256(url.encode())
    hex_dig = hash_object.hexdigest()
    filename = hex_dig + ".txt"
    log(f"{hex_dig} {url} {datetime.datetime.now()} {status_code}\n")
    if not rewrite and hex_dig in FILES:
        return
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    FILES.add(hex_dig)

def generate_content_block_sequence(document):
    """
    TODO: write docstring
    """
    # TODO: confirm no tag filtering is needed
    # produce a list of all spanning positions of HTML tags
    tag_span = []
    fv = open('./tags-processed.txt', 'w')
    for result in HTML_TAG_REGEX.finditer(document):
        tag_span.append(result.span())
        try:
            fv.write(document[tag_span[-1][0]:tag_span[-1][1]])
        except:
            pass
        fv.write('\n')
    fv.close()
    # merge neighbouring spans into conglomerates
    i = 0
    while i+1 < len(tag_span):
        # if current end == next start
        if tag_span[i][1] == tag_span[i+1][0]:
            tag_span[i] = (tag_span[i][0], tag_span[i+1][1])
            tag_span.pop(i+1)
        else:
            i+=1
    print(f'\TAGS\n\n{tag_span[0:100]}\n...\n')
    # produce a list of all spanning positions of non-HTML-tag tokens
    token_span = []
    fv = open('./tokens-processed.txt', 'w')
    for result in TOKEN_REGEX.finditer(document):
        token_span.append(result.span())
        try:
            fv.write(document[token_span[-1][0]:token_span[-1][1]])
        except:
            pass
        fv.write('\n')
    fv.close()
    print(token_span[-1])
    print(f'\nTOKENS\n\n{token_span[0:100]}\n...\n')
    # generate 'smart sequence' of 1s and 0s using spans and document
    sequence_span = []
    index_tag = 0
    index_token = 0
    while index_token < len(token_span):
        current_token_is_a_tag = False
        # if all tags have already been processed
        if index_tag >= len(tag_span):
            sequence_span.append((0, token_span[index_token][0], token_span[index_token][1]))
            index_token += 1
            continue
        # if current token is past current tag, update tag
        if token_span[index_token][0] > tag_span[index_tag][1]:
            index_tag += 1
            continue
        # if current token's span collides with current tag
        if (token_span[index_token][1] >= tag_span[index_tag][0] and token_span[index_token][1] <= tag_span[index_tag][1]) \
            or (token_span[index_token][0] <= tag_span[index_tag][1] and token_span[index_token][0] >= tag_span[index_tag][0]):
            current_token_is_a_tag = True
        sequence_span.append((1 if current_token_is_a_tag else 0, token_span[index_token][0], token_span[index_token][1]))
        index_token += 1
    print(f'\nSEQUENCE WITH SPANS\n\n{sequence_span[0:100]}\n...\n')
    sequence_as_list = [item[0] for item in sequence_span]
    sequence_as_string = ''.join([str(item[0]) for item in sequence_span])
    print(f'\nSEQUENCE AS STRING\n\n{sequence_as_string}\n')
    return sequence_span, sequence_as_list, sequence_as_string

def plot_function_2d(function_values, max_score):
    n = len(function_values)
    # generate 2 2d grids for the x & y bounds
    y, x = np.array(range(n)), np.array(range(n))
    z = function_values

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    # z = z[:-1, :-1]
    z_min, z_max = 0, max_score[1]

    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_xlabel('j')
    ax.set_ylabel('i')
    ax.set_title('Content Block Scores')
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)

    plt.show()

def plot_function_3d(function_values, max_scores):

    n = len(function_values)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # as plot_surface needs 2D arrays as input
    y, x = np.array(range(n)), np.array(range(n))
    # we make a meshgrid from the x,y data
    X, Y = np.meshgrid(x, y)
    Z = np.array(function_values)

    # data_value shall be represented by color
    # data_value = np.random.rand(len(y), len(x))
    # map the data to rgba values from a colormap
    colors = cm.ScalarMappable(cmap = "viridis").to_rgba(Z)


    # plot_surface with points X,Y,Z and data_value as colors
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors, linewidth=0, antialiased=True)

    plt.show()
        


def optimize_content_block(sequence_span, sequence_as_list):
    # TODO: docstring :)
    # TODO: check parameters & call to see if all is necessary

    # method 2: much faster, gotta confirm it works :)
    n = len(sequence_span)
    print(f'\nN_words: {n}, N_chars: {sequence_span[-1][1]}\n')
    function_values = [[0 for j in range(n)] for i in range(n)]
    i_change_score = sum(sequence_as_list)
    max_score = i_change_score
    max_span = (0,0)
    for i in range(0, n):
        curr_score = i_change_score
        for j in range(i+1, n):
            # if the 'consumed' bit: is 0 -> score += 1, if 1 -> score -= 1
            if sequence_as_list[j-1] == 0:
                curr_score += 1
            else:
                curr_score -= 1
            if curr_score > max_score:
                max_score = curr_score
                max_span = (i, j)
            function_values[i][j] = curr_score
    char_span = (sequence_span[max_span[0]][1], sequence_span[max_span[1]][2])
    print(f'\nMAX_SCORE: {max_score}, MAX_SPAN: {max_span}, CHAR_SPAN: {char_span}')
    return function_values, max_span, char_span

    # TODO: track all (i, j, score) so we can plot the function :)


    # method 1: too slow
    # max_score = 0
    # max_span = (None, None)
    # n = len(sequence_span)
    # print(f'\nN: {n}\n')
    # for i in range(0, n):
    #     for j in range(i, n):
    #         score = objective_function(sequence_as_list, i, j)
    #         if score > max_score: # TODO: confirm whether to use > or >=
    #             max_score = score
    #             max_span = (i, j)
    # print(f'\nMAX_SCORE: {max_score}, MAX_SPAN: {max_span}')
    # return max_span



def crawl(url, rewrite=False):
    """Crawl the URLs recursively from each page explored.
    TODO: update this docstring
    Args:
        url (str): The current URL being crawled
        rewrite (bool): Flag to rewrite the content files for the URL if recorded
    """
    r = requests.get(url)
    hash_and_save(url, r.status_code, r.text, rewrite)
    if r.status_code == COOLDOWN:
        print("Too many requests")
        return
    sequence_span, sequence_as_list, sequence_as_string = generate_content_block_sequence(r.text)
    optimization_values, max_span, content_block_span = optimize_content_block(sequence_span, sequence_as_list)

    content_block = r.text[content_block_span[0]:content_block_span[1]]
    print(f'\nCONTENT_BLOCK:\n{content_block}\n')
    print('\nPlotting optimization scores, please wait...')
    plot_function_2d(optimization_values, max_span)
    # plot_function_3d(optimization_values, content_block_span)
    print('\nDone!')

    # TODO: consider moving some calls to the "main" and re-organizing program flow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command Line Web Crawler")
    parser.add_argument("initialURL", type=str, help="The initial URL to crawl")
    args = parser.parse_args()
    # Clear the log
    with open(LOGGER_PATH, "w", encoding="utf-8") as f:
        pass
    crawl(args.initialURL)
    # TODO: output results to file
