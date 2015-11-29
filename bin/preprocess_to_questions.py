import question
import csv
import copy

def read_questions(PROJECT_PATH):
  kRecordTypeFieldNumber = 1
  kTagsFieldNumber = 16
  kBodyFieldNumber = 8
  
  kQuestion = '1'
  
  questions=[]
  
  csv_reader = csv.reader(open(PROJECT_PATH + 'data/plain-text/full.csv', 'r'))
  
  # Preprocess data
  for row in csv_reader:
    if (row[kRecordTypeFieldNumber]==kQuestion):
      tagstr = copy.deepcopy(row[kTagsFieldNumber])
      tagstr = tagstr.replace('<', '').replace('>', ' ').lower()
      tag_list = tagstr.split()
      
      if not tag_list or len(tag_list) == 0:
        continue
  
      bodystr = copy.deepcopy(row[kBodyFieldNumber])
      bodystr = bodystr.lower()
      questions.append(question.MakeQuestion(tag_list, bodystr))
  return questions

def vectorize_body(bodystr):
  one_grams = bodystr.split()
      
  word_counts = {}
  for one_gram in one_grams:
    word_counts[one_gram] = word_counts.setdefault(one_gram, 0) + 1
  return word_counts

def filter_tags(questions, tags):
  tag_set = set(tags)
  return [question.MakeQuestion(tag_set.intersection(q.tag_list), q.raw_words) for q in questions]

def tp1_filter(questions):
  return [q for q in questions if len(q.tag_list) == 1]
