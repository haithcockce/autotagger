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
import preprocess_to_questions as pq
# comment out if you don't want dependency on joblib
#from joblib import Parallel, delayed

PROJECT_PATH = '/home/njclimer/source/csc522/'
tags_to_consider = ['javascript', 'java', 'android', 'php', 'c#', 'python', 'jquery', 'html', 'ios']
questions = pq.read_questions(PROJECT_PATH)
questions = pq.filter_tags(questions, tags_to_consider)
print 'processing {} questions.'.format(len(questions))
# questions = questions[:1000]

all_tags = set()
for q in questions:
  all_tags.update(q.tag_list)

print all_tags
try:
  os.mkdir('./nieve_bayse_br_eval_doc')
except:
  pass

def nieve_bayse_factory():
  return nb.NaiveBayseClassifier(word_frequency=False)
def binary_relevance_factory():
  return br.BinaryRelevanceClassifier(nieve_bayse_factory)
  
ev.leave_one_out(binary_relevance_factory, ev.eval_tp1, './nieve_bayse_br_eval_doc', questions, all_tags, threads=3)
