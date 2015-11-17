#!/usr/bin/python

import sys, csv, copy
import numpy
import scipy
from sklearn import svm

if len(sys.argv) < 2:
    print("Usage: python svd.py <RAW DATA CSV>")
    sys.exit()

FILEPATH = sys.argv[1]
TAGS_INDEX = 0
BODY_INDEX = 1
buff = []
records = []
unique_monograms = {}
unqiue_tags = []

classifications = []

    
# Construct the records
print(FILEPATH) 
csv_reader = csv.reader(open(FILEPATH, 'r', newline=''))
for row in csv_reader:
    if row[1] == '1':
        records.append(row[8])
        records.append(row[16])

# Construct the dictionary of unique monograms
# Useful to gather the count later. 
for record in records:
    buff = record[BODY_INDEX].split(' ')
    for word in buff:
        if word not in unique_monograms:
            unique_monograms[word] = 0;

# After constructing unique monograms,
# start populating the document matrix
# The matrix will need to have duplicate 
# rows for each tag the record has. 
buff_dict = {}

for record in records:
    buff_dict = copy.deepcopy(unique_monograms)
    buff = record[BODY_INDEX].split(' ')
    tags = record[TAGS_INDEX].replace('<', '').replace('>', ' ').split(' ')
    for word in buff:
        buff_dict[word] += 1
     
    for tag in tags:
        clas
