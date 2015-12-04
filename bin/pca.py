import csv
import sys
import string
import copy
from sklearn.decomposition import PCA
import documentdb
from documentdb import document_db

csv.field_size_limit(sys.maxsize)
'''
TODO
1) Multi sampling PCA:
INIT:
Get a nxm Matrix going: n is interations of analysis while m is attr (unique words) and cells are the sum of the PCA values
Get a submatrix going: keys are the tags, values are dicts: keys are the unique words and values are the count of words
Get a temp list for PCA component values
Calculate the size of partitions
    o 13219 total unique tags, so 10 total paritions?
    
For n iterations of analysis:
  - Partition data randomly into even partitions and calculate the PCA values for each partition and sum the PCA values
    - shuffle unique_tags
      o import random; random.shuffle(array)
    - For the k-th parition 
      - populate submatrix with the tag considered and the count of onegrams 
        where each partition is (i * partition size) to (i * partition size + partition size)
        o refer to documentdb.py
    - PCA on the k-th partition and add/multiply the PCA values to the temp list
  - append the PCA values to the PCA analysis matrix
'''

ITERATIONS = 10000

test_data = [] 
# test_data[0] == java
# test_data[1] == android
# test_data[2] == c
# test_data[3] == python
# test_data[4] == .net

test_classes = ['java', 'android', 'c', 'python', '.net']
for class_label in test_classes:

    ith_document = copy.deepcopy(documentdb.unique_one_grams)
  
    index = documentdb.unique_tags.index(class_label) + 1 # offset to make up for headers in records
    one_grams = documentdb.records[index][1].split()
    for one_gram in one_grams:
        ith_document[one_gram] += 1 
    
    document_db.append(list(ith_document.values()))
