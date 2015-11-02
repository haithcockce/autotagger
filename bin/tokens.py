import sys
import csv
import string
import nltk
from nltk.util import bigrams
from nltk.util import trigrams

"""
TODO

. Finish the TagRecord class
    - update NGrams appropriately
    - find someway of writing this to a file
. Create an NGramRecord class? 
    - 1 gram to many tags relationship
        o TagRecord is the opposite: 1 tag to many ngrams
    - find a way of writing this to a file
. Line 90:
    - create the TagRecords and add them to a list of the tag records
    - This may require reworking a bit of code below
"""


class TagRecord:
    'Contains the tag, tag count of occurrance, and all unique 1grams associated with it along with their count' 

    def __init__(self, tag, one_gram_list):
        self.tag = ''
        self.tag_cnt = 0
        self.one_grams = {}
        # self.two_grams = {}
        # self.three_grams = {}
        for i in range(0, len(one_gram_list)):
            self.one_grams[one_gram_list[i]] = 1
            # self.two_grams[one_gram_list[i]] = 1
            # self.three_grams[one_gram_list[i]] = 1

    def getTag(self):
        return self.tag

    def getTagCount(self):
        return self.tag_cnt

    def updateTagCount(self):
        self.tag_cnt += 1

    def updateNGram(self, new_one_gram):
        if new_one_gram not in self.one_grams:
            self.one_grams[new_one_gram] = 1
        else # if new_one_gram in self.one_grams
            self.one_grams[new_one_gram] += 1

    def getOneGrams(self):
        return self.one_grams


PROJECT_PATH = '/home/dev/School/CSc-522/Project/autotagger/'
TAG_INDEX = 16

tokens_with_tags = []
records = []
tags = ''
tag_list = []
raw_data = []
# raw_data[0][0] = 'Id'
# raw_data[0][1] = 'BodyTokenized - 1Grams'
# raw_data[0][2] = 'BodyTokenized - 2Grams'
# raw_data[0][3] = 'BodyTokenized - 3Grams'
# raw_data[0][4] = 'TagsTokenized'

tags_1grams = []
tags_1grams[0][0] = None

# Open the raw CSV and read in all the records
csv_reader = csv.reader(open(PROJECT_PATH + 'data/plain-text/TEST-full.csv', 'r', newline=''))
for row in csv_reader:
    records.append(row)

# Building the tokenized tag list and body
for i in range(1, len(records)):
    tags = copy.deepcopy(records[i][TAG_INDEX])
    tags = tags.replace('<', '').replace('>', ' ') 

    tag_list[i][0] = copy.deepcopy(records[i][0])
    tag_list[i][1] = records[i][8].split()
    tag_list[i][2] = list(bigrams(records[i][8].split()))
    tag_list[i][3] = list(trigrams(records[i][8].split()))
    tag_list[i][4] = tags.split()

    # Building a count of the tags
    for n in range(0, len(tag_list[i][4])):
        


