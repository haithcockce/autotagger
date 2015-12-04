import csv
import sys
import string
import copy
from sklearn import svm

csv.field_size_limit(sys.maxsize)
'''
TODO
2) Consider training a dumb classifier 
    - simply set the one_gram to be tagged with the tags with majoroty of occurrences 
    - From there, the top 3 words with the highest majorities are the tags
3) Consider eliminating words? 
    - Maybe eliminating one_grams with something close to a uniform dist of occurrence. 
'''

# MACROS
PROJECT_PATH = '/home/dev/School/CSc-522/Project/autotagger/'

# vars
records = []
document_db = []
unique_tags = []
unique_one_grams = {}

one_grams = []

# Open the raw CSV and read in all the records
csv_reader = csv.reader(open(PROJECT_PATH + 'data/tags-ngrams/1-tag-many-one-grams.csv', 'r', newline=''))
for row in csv_reader:
    records.append(row)

for i in range(1, len(records)):
    
    # Build the list of unqiue_tags
    unique_tags.append(copy.deepcopy(records[i][0]))
    
    # Build the base dict of unique_one_grams
    one_grams = records[i][1].split()
    for one_gram in one_grams:
        if one_gram not in unique_one_grams:
            unique_one_grams[one_gram] = 0
            
# At this point, we should have a base dict of all possible unique one
# grams and a unique list of tags (unique_one_grams.keys() and unique_
# tag). Here, we need to build the count of specific one grams for each
# list of one grams associated with a tag.
#
#for i in range(1, len(records)):
#    
#    ith_document = copy.deepcopy(unique_one_grams)
#   
#    one_grams = records[i][1].split()
#    for one_gram in one_grams:
#        ith_document[one_gram] += 1 
#    
#    document_db.append(list(ith_document.values()))
#
#unique_one_grams = list(unique_one_grams.keys())

# This should construct a document database of the count of one grams per tag. 
# Treat this as a module to reuse. 
# Things to use: 
#    - document_db: 
#          document_db[tags]: record is the count of each one gram for 'tag'
#          document_db[tags][one_gram]: columns are the counts of the 'one_gram' for 'tag'
#    - unique_one_grams:
#          list of unique one grams
#    - unique_tags:
#          list of unique tags
