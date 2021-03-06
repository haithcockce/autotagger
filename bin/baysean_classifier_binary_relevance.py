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

PROJECT_PATH = '/home/nclimer/autotagger/'
tags_to_consider = ['javascript', 'java', 'android', 'php', 'c#', 'python', 'jquery', 'html', 'ios']
questions = pq.read_questions(PROJECT_PATH)
questions = pq.filter_tags(questions, tags_to_consider)
# questions = questions[:100]

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
  
<<<<<<< HEAD
ev.leave_one_out(binary_relevance_factory, ev.eval_tp1, './nieve_bayse_br_eval', questions, all_tags, threads=1)
=======
ev.leave_one_out(binary_relevance_factory, ev.eval_tp1, './nieve_bayse_br_eval', questions, all_tags, threads=4)
>>>>>>> 1724f010a63a1910b1d320bb0b53f6f263add1f0
