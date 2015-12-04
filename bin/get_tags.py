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

questions = pq.read_questions(PROJECT_PATH)

print 'Evaulationg {} questions'.format(len(questions))
tag_counts = {}
for q in questions:
  for tag in q.tag_list:
    tag_counts[tag] = tag_counts.get(tag, 0) + 1

tags_to_keep = 10
tags = list(reversed(sorted([(count, tag) for tag, count in tag_counts.iteritems()])))
tags = tags[:tags_to_keep]
tags = [tag for _, tag in tags]
print tags
