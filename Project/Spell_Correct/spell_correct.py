"""
Author: Kelvin Kellner
"""
from Soundex.soundex import compute_soundex
import string


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
    return set(ed2 for ed1 in edit_distance_of_1(word) for ed2 in edit_distance_of_1(ed1))


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
    # 1. get all edit distance 1 words
    # 1.a. if exactly 1 then use it, else...
    # 1.b. if none, then move to step 2, else...
    # 1.c. use the term with the highest term frequency
    # 2. get all edit distance 2 words
    # 2.a. if exactly 1 then use it, else...
    # 2.b. if none, then move to step 3, else...
    # 2.c. use the term with the highest term frequency
    # 3. not able to use edit distance distance after all, fallback is...
    # 3.a. use the term with the highest term frequency
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
    """
    Spell correct a query using the soundex similarity from the inverted index
    -----
    Args:
        query (str): Query to spell correct
        inverted_index (dict): Inverted index to use for spell correction
    Returns:
        spell_corrected_query (str): Spell corrected query
    """
    # for each term in query_terms if it is in the inverted index then use it,
    # otherwise look for soundex matches and use closest match function to choose
    # if there are no soundex matches then remove the term from the query
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