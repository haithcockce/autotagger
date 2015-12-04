#!/usr/bin/python

import sys, csv, copy, time, string, pdb
import numpy
import scipy.sparse as spsparse
from scipy.sparse import linalg as spsplinalg
import nltk
from nltk.corpus import stopwords as sw
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

if len(sys.argv) < 2:
    print("Usage: python svd.py <RAW DATA CSV>")
    sys.exit()

FILEPATH = sys.argv[1]
TAGS_INDEX = 0
BODY_INDEX = 1
TRANSTABLE = {ord(c): ' ' for c in string.punctuation}
DOC_COUNT = 0

dictionary = {}
popular_tags = {}
unqiue_tags = {}
records = []
stopwords = sw.words('english')

start_time = time.time()
print(start_time)

# most_popular:
#    param: tags
#       - array of strings
#    determine the most popular tag of the 
#    list of tags provided returning None 
#    if not found
def most_popular_tag(tags):
    for tag in popular_tags:
        if tag in tags:
            return tag
    return None


def build_clean_unique_tags(tags_dict):

    # Normalize the counts
    sigma = float(sum(list(tags_dict.values())))
    normalized_tags_dict = copy.deepcopy(tags_dict)
    for key in normalized_tags_dict:
        normalized_tags_dict[key] = float(normalized_tags_dict[key]) / sigma

    # Remove all tags that show up in 
    # less than 1% of all tag instances
    for i in range(0, len(normalized_tags_dict)):
        key = min(normalized_tags_dict, key=normalized_tags_dict.get)
        if normalized_tags_dict[key] < .01:
            normalized_tags_dict.pop(key, None)
        else:
            break

    # Build a sorted list of the tags in
    # order of most common to least common
    sorted_tags = []
    for i in range(0, len(normalized_tags_dict)):
        tag = max(normalized_tags_dict, key=normalized_tags_dict.get)
        print(tag)
        sorted_tags.append(tag)
        normalized_tags_dict.pop(tag, None)

    return sorted_tags




# Construct the records
csv_reader = csv.reader(open(FILEPATH, 'r', newline=''))
for row in csv_reader:

    # ignore non-questions
    if row[1] != '1':
        continue

    # Ignore records that have no tags
    if len(row[16]) == 0:          
        continue 

    records.append([row[16], row[8]])



print('DEBUG: Constructed records')
print(time.time() - start_time)



for record in records:
    # Construct the dictionary of unique monograms
    # Useful to gather the count later. 
    word_list = record[BODY_INDEX].lower().translate(TRANSTABLE).split(' ')
    for word in word_list:
        if (word not in dictionary) and (word not in stopwords) and (len(word) > 0):
            dictionary[word] = 0;

# Construct the dictionary of unique keys
tag_count = {}
for record in records:
    tag_list = record[TAGS_INDEX].lower().replace('<', '').replace('>', ' ').split(' ')
    for tag in tag_list:
        if len(tag) == 0:
            continue
        if tag not in tag_count:
            tag_count[tag] = 0
        else:
            tag_count[tag] += 1

popular_tags = build_clean_unique_tags(tag_count)


print('DEBUG: Created unique monograms and popular tags')
print(time.time() - start_time)


# After constructing unique monograms,
# start populating the document matrix
# The matrix will need to have duplicate 
# rows for each tag the record has. 
monogram_counts = {}
tags = []
iterations = 0
buff = []

for record in records:

    # if the tag is uncommon, don't use this record
    tag_list = record[TAGS_INDEX].lower().replace('<', '').replace('>', ' ').split(' ')
    tag = most_popular_tag(tag_list)
    if tag == None:
        continue
    record[TAGS_INDEX] = tag

    # copy the monogram template to count the words for this document 
    monogram_counts = copy.deepcopy(dictionary)
    monogram_list = record[BODY_INDEX].lower().translate(TRANSTABLE).split(' ')

    # begin counting the records. 
    for word in monogram_list:
        if (word not in stopwords) and (len(word) > 0) and (word in monogram_counts):
            monogram_counts[word] += 1

    # grab the classes (tags) of this record (document) and append
    # to the lists
    buff.append(list(monogram_counts.values()))
    tags.append(tag)    # This will be used as the classes
    DOC_COUNT += 1      # Useful because scipy is largely useless. 

    # Sanity check and concatenation. Jesus numpy is slow as shit. 
    if (iterations % 250) == 0:
        print('DEBUG: vstack-ed {:d} times'.format(iterations))
        try:
            temp_matrix = spsparse.vstack((temp_matrix, numpy.array(buff).astype('d')))
        except NameError:
            temp_matrix = spsparse.coo_matrix((numpy.array(buff).astype('d')))
        buff = []
    iterations += 1


print('DEBUG: Made entire matrix')
print(time.time() - start_time)

sparse_data = spsparse.csc_matrix(temp_matrix)

print('DEBUG: Made sparse matrix')
print(time.time() - start_time)


# At this point in time, there should be the following:
#   tags: 
#       TYPE list of strings
#       The classificaitons (tag) of each doc. Here, the 
#       ith tag represents the classification of the ith
#       document, werein the same document will be appended 
#       repeatedly for each tag that it may have. 
#   sparse_data:
#       TYPE scipy.sparse.compressed sparse column matrix of ints
#       The documents are represented here via monogram count. 
#       This will have n entries of the same document for n tags
#       that document has. 

#svd = TruncatedSVD(n_components=10000)
#svd.fit(sparse_data)

print('DEBUG: Trained sklearn\'s SVD on 10000 components')
print(time.time() - start_time)

#sparse_data_norm = spsparse.csc_matrix(normalize(sparse_data, axis=0))

#print('DEBUG: Normalized Matrix')
#print(time.time() - start_time)
#print('DEBUG: Calculated Singular values')
#print(time.time() - start_time)



