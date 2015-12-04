import sys
import string
import copy
import question
import naive_bayse as nb
import sets
import os
import eval_classifier as ev
import preprocess_to_questions as pq
# comment out if you don't want dependency on joblib
#from joblib import Parallel, delayed

PROJECT_PATH = '/home/njclimer/source/csc522/'

tags_to_consider = ['c++', 'sql', 'angularjs']

questions = pq.read_questions(PROJECT_PATH)
questions = pq.tp1_filter(pq.filter_tags(questions, tags_to_consider))
questions = questions[:100]
print 'Evaulationg {} questions'.format(len(questions))
all_tags = set()
for q in questions:
  all_tags.update(q.tag_list)

print all_tags
try:
  os.mkdir('./nieve_bayse_doc_counts_eval')
except:
  pass

class nieve_bayse_factory:
  def __call__(self):
    return nb.NaiveBayseClassifier(1, word_frequency=False)

ev.leave_one_out(nieve_bayse_factory(), ev.eval_tp1, './nieve_bayse_doc_counts_eval', questions, all_tags, threads=1)
