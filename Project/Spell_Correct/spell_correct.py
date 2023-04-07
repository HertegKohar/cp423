"""
Author: Kelvin Kellner
"""
import nltk
import re
import string


# TODO: make this a separate package
def compute_soundex(term):
    """
    Compute the soundex code for a given term
    -----
    Args:
        term (str): Term to compute soundex code for
    Returns:
        soundex (str): Soundex code for 'term'
    """
    soundex = ""
    soundex += term[0].upper()
    for char in term[1:].lower():
        if char in "bfpv":
            soundex += "1"
        elif char in "cgjkqsxz":
            soundex += "2"
        elif char in "dt":
            soundex += "3"
        elif char in "l":
            soundex += "4"
        elif char in "mn":
            soundex += "5"
        elif char in "r":
            soundex += "6"
        else:
            soundex += "0"
    soundex = re.sub(r"(.)\1+", r"\1", soundex)
    soundex = re.sub(r"0", "", soundex)
    soundex = soundex[:4].ljust(4, "0")
    return soundex


def edit_distance_of_1(word):
    """
    Generate all possible edits that are 1 edit distance away from 'word'
    -----
    Args:
        word (str): Word to generate edits for
    Returns:
        ed1 (list[str]): List of all possible edits that are 1 edit distance away from 'word'
    """
    # perform all edits of all types on 'word' and create a set of the results
    alphabet = string.ascii_lowercase
    pieces = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    insertions = [left + letter + right for left, right in pieces for letter in alphabet]
    deletions = [left + right[1:] for left, right in pieces if right]
    replacements = [left + letter + right[1:] for left, right in pieces if right for letter in alphabet]
    transpositions = [left + right[1] + right[0] + right[2:] for left, right in pieces if len(right) > 1]
    return set(insertions + deletions + replacements + transpositions)


def edit_distance_of_2(word): 
    """
    Generate all possible edits that are 2 edit distance away from 'word'
    -----
    Args:
        word (str): Word to generate edits for
    Returns:
        ed2 (list[str]): List of all possible edits that are 2 edit distance away from 'word'
    """
    # compute all combinations of 1 more edit on all edits of 1 distance from 'word'
    return (ed2 for ed1 in edit_distance_of_1(word) for ed2 in edit_distance_of_1(ed1))


def get_closest_match(term, matches, inverted_index):
    """
    Return the closest match to 'term' in 'matches'
    -----
    Args:
        term (str): Term to find closest match for
        matches (list[str]): List of terms to compare to 'term'
        inverted_index (dict): Inverted index to use for comparison
    Returns:
        closest_match (str): Closest match to 'term' in 'matches'
    """
    # 1. get all edit distance 1 words, if tie then...
    # 1.a. use the term with the highest term frequency TODO: confirm this is best (e.g. not user chooses)
    # 2. get all edit distance 2 words, if tie then...
    # 2.a. use the term with the highest term frequency TODO: confirm this is best (e.g. not user chooses)
    # 3. not using edit distance distance after all...
    # 3.a. use the term with the highest term frequency TODO: confirm this is best (e.g. not user chooses)

    term_freq = {match: sum(occ[1] for occ in inverted_index[match]["occurences"]) for match in matches}
    highest_term_freq_match = max(term_freq, key=term_freq.get)

    e1 = edit_distance_of_1(term)
    e1_matches = e1.intersection(matches)
    if len(e1_matches) > 0:
        if len(e1_matches) == 1:
            return e1_matches.pop()
        highest_term_freq_match = max(e1_matches, key=lambda x: term_freq[x])
        return highest_term_freq_match
    e2 = edit_distance_of_2(term)
    e2_matches = e2.intersection(matches)
    if len(e2_matches) > 0:
        if len(e2_matches) == 1:
            return e2_matches.pop()
        highest_term_freq_match = max(e2_matches, key=lambda x: term_freq[x])
        return highest_term_freq_match
    highest_term_freq_match = max(matches, key=lambda x: term_freq[x])
    return highest_term_freq_match




def spell_correct_query(query, inverted_index):
    # for each term in query_terms if it is in the inverted index then use it,
    # otherwise look for soundex matches and use closest match function to choose
    # if there are no soundex matches then remove the term from query_df TODO: confirm this is appropriate
    query_terms = query.split()
    spell_corrected_terms = []
    for i, term in enumerate(query_terms):
        matches = []
        for key in inverted_index.keys():
            if term == key:
                spell_corrected_terms.append(term)
                break
            if compute_soundex(term) == inverted_index[key]["soundex"]:
                matches.append(key)
        else:
            if len(matches) == 1:
                spell_corrected_terms.append(matches[0])
            elif len(matches) > 1:
                # print(f"Multiple matches found for {term}: {matches}, using first match")
                best_match = get_closest_match(term, matches, inverted_index)
                spell_corrected_terms.append(best_match)
    spell_corrected_query = " ".join(spell_corrected_terms)
    return spell_corrected_query