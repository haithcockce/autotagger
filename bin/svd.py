#!/usr/bin/python

import sys, csv, copy
import numpy
import scipy.sparse as spsparse
from scipy.sparse import linalg as spsplinalg
import nltk
from nltk.corpus import stopwords as sw
import time
import string

if len(sys.argv) < 2:
    print("Usage: python svd.py <RAW DATA CSV>")
    sys.exit()

FILEPATH = sys.argv[1]
TAGS_INDEX = 0
BODY_INDEX = 1
TRANSTABLE = {ord(c): ' ' for c in string.punctuation}

unique_monograms = {}
unqiue_tags = []
records = []
stopwords = sw.words('english')

# Construct the records
print(FILEPATH) 
start_time = time.time()
print(start_time)

csv_reader = csv.reader(open(FILEPATH, 'r', newline=''))
for row in csv_reader:

    # ignore non-questions
    if row[1] == '1':
        # Ignore records that have no tags
        if len(row[16]) == 0: 
            continue 

        records.append([row[16], row[8]])

for record in records:
    # Construct the dictionary of unique monograms
    # Useful to gather the count later. 
    word_list = record[BODY_INDEX].lower().translate(TRANSTABLE).split(' ')
    for word in word_list:
        if (word not in unique_monograms) and (word not in stopwords) and (len(word) > 0):
            unique_monograms[word] = 0;


# After constructing unique monograms,
# start populating the document matrix
# The matrix will need to have duplicate 
# rows for each tag the record has. 
monogram_counts = {}
tags = []
iterations = 0


for record in records:

    # copy the monogram template to count the words for this document 
    monogram_counts = copy.deepcopy(unique_monograms)
    monogram_list = record[BODY_INDEX].lower().translate(TRANSTABLE).split(' ')
    tag_list = record[TAGS_INDEX].lower().replace('<', '').replace('>', ' ').split(' ')

    # begin counting the records. 
    for word in monogram_list:
        if (word not in stopwords) and (len(word) > 0) and (word in monogram_counts):
            monogram_counts[word] += 1

    # for code legibility 
    monogram_counts = numpy.array(list(monogram_counts.values()))

     
    # grab the classes (tags) of this record (document) and append
    # to the lists
    for tag in tag_list:
        try:
            temp_matrix = numpy.vstack((temp_matrix, monogram_counts))
        except NameError:
            temp_matrix = copy.deepcopy(monogram_counts)
        tags.append(tag)    # This will be used as the classes


sparse_data = spsparse.csc_matrix(temp_matrix)

end_time = time.time()

print(end_time - start_time)

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

# TODO
# . actually perform SVD
#     - resulting matrices might be too large :c

#U, s = spsplinalg.svds(sparse_data, k=len(unique_monograms), return_singular_vectors="u")
