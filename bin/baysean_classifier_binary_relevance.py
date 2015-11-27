import sys
import csv
import string
import copy
import question
import naive_bayse as nb
import sets
import os
import eval_classifier as ev
import binary_relevance as br
# comment out if you don't want dependency on joblib
#from joblib import Parallel, delayed

PROJECT_PATH = '/home/nclimer/autotagger/'
kRecordTypeFieldNumber = 1
kTagsFieldNumber = 16
kBodyFieldNumber = 8

kQuestion = '1'

# if set, filters only these tags. Else, uses all tags
tags_to_consider = ['c++', 'java']
questions=[]

csv_reader = csv.reader(open(PROJECT_PATH + 'data/plain-text/full.csv', 'r'))

all_tags = set()
# Preprocess data
for row in csv_reader:
  if (row[kRecordTypeFieldNumber]==kQuestion):
    tagstr = copy.deepcopy(row[kTagsFieldNumber])
    tagstr = tagstr.replace('<', '').replace('>', ' ').lower()
    tag_list = tagstr.split()
    if tags_to_consider and len(tags_to_consider) > 0:
      tag_list = [tag for tag in tag_list if tag in tags_to_consider]
    if not tag_list or len(tag_list) == 0:
      continue

    bodystr = copy.deepcopy(row[kBodyFieldNumber])
    bodystr = bodystr.lower()
    one_grams = bodystr.split()
    
    word_counts = {}

    for one_gram in one_grams:
      word_counts[one_gram] = word_counts.setdefault(one_gram, 0) + 1

    questions.append(question.Question(tag_list, word_counts))

questions = questions[:100]

all_tags = set()
for q in questions:
  all_tags.update(q.tag_list)

print all_tags
try:
  os.mkdir('./nieve_bayse_br_eval')
except:
  pass

def nieve_bayse_factory():
  return nb.NaiveBayseClassifier()
def binary_relevance_factory():
  return br.BinaryRelevanceClassifier(nieve_bayse_factory)
  

ev.leave_one_out(binary_relevance_factory, ev.eval_tp1, './nieve_bayse_br_eval', questions, all_tags)
