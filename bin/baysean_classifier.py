import sys
import csv
import string
import copy
import question
import naive_bayse
# comment out if you don't want dependency on joblib
from joblib import Parallel, delayed

PROJECT_PATH = '/home/njclimer/source/csc522/'
kRecordTypeFieldNumber = 1
kTagsFieldNumber = 16
kBodyFieldNumber = 8

kQuestion = '1'

# if set, filters only these tags. Else, uses all tags
tags_to_consider = ['c++', 'java']
questions=[]

csv_reader = csv.reader(open(PROJECT_PATH + 'data/plain-text/full.csv', 'r'))

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

nbc = naive_bayse.NaiveBayseClassifier()
nbc.Train(questions)
def classify(question):
  return nbc.Classify(question, False)
# use this line if you don't want dependency on joblib
#cls = [classify(question) for question in questions]
cls = Parallel(n_jobs=8)(delayed(classify)(question) for question in questions)
tp = {}
fp = {}
fn = {}
for i in range(1, len(cls)):
  c = cls[i]
  q = questions[i]
  if c in q.tag_list:
    tp[c] = tp.setdefault(c, 0) + 1
  else:
    fp[c] = fp.setdefault(c, 0) + 1
  for t in q.tag_list:
    if not t == c:
      fn[t] = fn.setdefault(t, 0) + 1

for c in nbc.prob_tag.keys():
  print "{} tp: {}, fp: {}, fn: {}".format(c, tp.get(c,0), fp.get(c, 0), fn.get(c, 0))
