#!/usr/bin/python

import sys, os, csv, string
import numpy
import scipy
from sklearn import svm

if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
    print("Usage: python cleaner.py <RAW DATA CSV>")
    sys.exit()

FILEPATH = sys.argv[1]
buff = []
word_count = {}
words = []
tags_per_document = []
#document_db 
    
print(FILEPATH) 
csv_reader = csv.reader(open(FILEPATH, 'r', newline=''))
for row in csv_reader:
    if row[1] == '1':
        words = row[8].split(' ')
       
# for words in the row, 
#     - increment the count of word[i] in unique_words
# for tags in row,
#     - append the tag to the tags_per_document[]
#     - append the word_count.values() to the sparse matrix
        

