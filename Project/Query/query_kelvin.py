def query():
    raise NotImplementedError

import collections
import heapq
import json
import math
from soundex import compute_soundex

'''
procedure TermAtATimeRetrieval(Q, I, f , g k)
A ← HashTable()
L ← Array()
R ← PriorityQueue(k)
for all terms wi in Q do
li ← InvertedList(wi, I)
L.add( li )
end for
for all lists li ∈ L do
while li is not finished do
d ← li.getCurrentDocument()
Ad ← Ad + gi(Q)f (li)
li.moveToNextDocument()
end while
end for
for all accumulators Ad in A do
sd ← Ad # Accumulator contains the document score
R.add( sd, d )
end for
return the top k results from R
end procedure
'''

MAX_QUERY_RESULTS = 3

FILE_INVERTED_INDEX = 'inverted_index.json'
FILE_MAPPING = 'mapping.json'

def compute_term_document_score(term, document_code):
    # use tf-idf ranking model to calculate the score for a term in a document
    with open(FILE_INVERTED_INDEX, 'r') as f:
        inverted_index = json.load(f)
    if document_code not in inverted_index[term]['occurrences']:
        return 0
    term_frequency_in_document = inverted_index[term]['occurrences'][document_code]
    return 1 + math.log(term_frequency_in_document)

def compute_term_index_score(term):
    # use tf-idf ranking model to compute the score for a term in the collection
    with open(FILE_INVERTED_INDEX, 'r') as f:
        inverted_index = json.load(f)
    number_of_documents_containing_word = len(inverted_index[term]['occurrences'])
    number_of_documents_in_collection = len(inverted_index)
    return number_of_documents_in_collection / number_of_documents_containing_word

    

def search_for_word(word):
    matching_soundex = []
    with open(FILE_INVERTED_INDEX, 'r') as f:
        inverted_index = json.load(f)
        for key in inverted_index.keys():
            if word == key:
                return [word]
            if compute_soundex(word) == inverted_index[key]['soundex']:
                matching_soundex.append(key)
    return matching_soundex

def spell_correct_terms(query_terms):
    # for each term in query_terms check if it is in the inverted index, if not then look for a soundex match
    # if there is a soundex match then replace the term in query_terms with the soundex match
    # if there is no soundex match then remove the term from query_terms
    new_query_terms = []
    for term in query_terms:
        matches = search_for_word(term)
        if len(matches) == 1:
            new_query_terms.append(matches[0])
        elif len(matches) > 1:
            print(f"Multiple matches found for {term}: {matches}, using first match")
            new_query_terms.append(matches[0])
        else:
            print(f"No matches found for {term}")
    return new_query_terms

def term_at_a_time_retrieval(query_terms, func_f, func_g, max_results):
    A = {}
    R = []

    # Load inverted index and document mapping
    with open('inverted_index.json', 'r') as f:
        inverted_index = json.load(f)
    with open('mapping.json', 'r') as f:
        mapping = json.load(f)
    mapping_keys = list(mapping.keys())
    
    # Spell correct query terms
    correct_terms = spell_correct_terms(query_terms)
    print('Spell corrected query:', ' '.join(correct_terms))

    # Compile documents for each term
    L = []
    for term in correct_terms:
        document_hashes = []
        occurrences = inverted_index[term]['occurrences']
        for key in occurrences.keys():
            i = int(key[1:])
            document_hashes.append((key, term)) # occurrences[key]))
        L.append(document_hashes)

    # Calulate accumulator for each document

    # Process each inverted list
    for i, inv_list in enumerate(L):
        for doc_id, term in inv_list:
            # Compute feature functions f and g
            # print(doc_id, term)
            partial_score = func_f(term, doc_id) * func_g(term)
            
            # Update accumulator for the document
            if doc_id in A:
                A[doc_id] += partial_score
            else:
                A[doc_id] = partial_score

    # Compute final scores for each document
    for doc_id, score in A.items():
        final_score = score * len(correct_terms)
        heapq.heappush(R, (final_score, doc_id))

    # Return the top results
    return [heapq.heappop(R) for i in range(min(max_results, len(R)))]


    # # Iterate over all inverted lists
    # for li in document_hashes:
    #     while not li.finished():
    #         d = li.getCurrentDocument()
    #         Ad = A[d]
    #         Ad += feature_func_g(correct_terms) * feature_func_f(li)
    #         li.moveToNextDocument()
    
    # # Add all accumulators to priority queue
    # for d, Ad in A.items():
    #     sd = Ad # Accumulator contains the document score
    #     heapq.heappush(R, (-sd, d))
    
    # # Return top k results from priority queue
    # top_k = []
    # for i in range(max_results):
    #     if R:
    #         score, doc = heapq.heappop(R)
    #         top_k.append(doc)
    
    # return top_k

query = 'dow jones finance'
query_terms = query.split(' ')
results = term_at_a_time_retrieval(query_terms, compute_term_document_score, compute_term_index_score, MAX_QUERY_RESULTS)
print(results)