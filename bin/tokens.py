import sys
import csv
import string
import nltk
import copy
from nltk.util import bigrams
from nltk.util import trigrams

"""
TODO

"""


class TagRecord:
    'Contains the tag, tag count of occurrance, and all unique 1grams associated with it along with their count' 

    def __init__(self, tag, one_gram_list):
        self.tag = ''
        self.tag_cnt = 1
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
        else: # if new_one_gram in self.one_grams
            self.one_grams[new_one_gram] += 1

    def getOneGrams(self):
        return self.one_grams


class OneGramRecord:
    'Contains the ngram, ngram count of occurrance, and all unique tags associated with it along with their count' 

    def __init__(self, one_gram, tags):
        self.one_gram = ''
        self.one_gram_cnt = 1
        self.tags = {}
        # self.two_grams = {}
        # self.three_grams = {}
        for i in range(0, len(tags)):
            self.tags[tags[i]] = 1
            # self.two_grams[one_gram_list[i]] = 1
            # self.three_grams[one_gram_list[i]] = 1

    def getTag(self):
        return self.one_gram

    def getTagCount(self):
        return self.one_gram_cnt

    def updateTagCount(self):
        self.one_gram_cnt += 1

    def updateNGram(self, new_tag):
        if new_tag not in self.tags:
            self.tags[new_tag] = 1
        else: # if new_one_gram in self.one_grams
            self.tags[new_tag] += 1

    def getOneGrams(self):
        return self.one_grams
        
PROJECT_PATH = '/home/dev/School/CSc-522/Project/autotagger/'

records = []
unique_tags_one_grams_dict = {}
unique_one_grams_tags_dict = {}


# Open the raw CSV and read in all the records
csv_reader = csv.reader(open(PROJECT_PATH + 'data/plain-text/full.csv', 'r', newline=''))
for row in csv_reader:
    records.append(row)

# Building the tokenized tag list and body
for i in range(1, len(records)):
  
    # Records that are not questions (id type 1) will not have tags
    if records[i][1] != '1':
        continue

    # Setting up unique tag dict
    tagstr = copy.deepcopy(records[i][16])
    tagstr = tagstr.replace('<', '').replace('>', ' ').lower()
    tag_list = tagstr.split();


    for tag in tag_list:
        if tag in unique_tags_one_grams_dict:
            unique_tags_one_grams_dict[tag] += (records[i][8].lower() + ' ')
        else:
            unique_tags_one_grams_dict[tag] = (records[i][8].lower() + ' ')

    # Building the tokenized body list and tags
    bodystr = copy.deepcopy(records[i][8])
    bodystr = bodystr.lower()
    one_grams = bodystr.split()
    
    for one_gram in one_grams:
        if one_gram in unique_one_grams_tags_dict:
            unique_one_grams_tags_dict[one_gram] += (tagstr + ' ')
        else:
            unique_one_grams_tags_dict[one_gram] = (tagstr + ' ')

row = []
header = []

# Write out the unique tag dictionary
csv_out = open(PROJECT_PATH + 'data/tags-ngrams/1-tag-many-one-grams', 'w', newline='')
csv_writer = csv.writer(csv_out, dialect='unix')
header = ['Tag', 'OneGrams']
csv_writer.writerow(header)

for key in iter(unique_tags_one_grams_dict):
    row = [key, unique_tags_one_grams_dict[key]]
    csv_writer.writerow(row)

# Write out the unique one gram dictionary
csv_out = open(PROJECT_PATH + 'data/tags-ngrams/1-one-gram-many-tags', 'w', newline='')
csv_writer = csv.writer(csv_out, dialect='unix')
header = ['OneGram', 'Tags']
csv_writer.writerow(header)

for key in iter(unique_one_grams_tags_dict):
    row = [key, unique_one_grams_tags_dict[key]]
    csv_writer.writerow(row)