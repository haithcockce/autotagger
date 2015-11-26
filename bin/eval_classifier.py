import sets

def tag_suggester_cost(tp, tn, fp, fn, qs, tags, ntags):
  tp_cost = -float(1)
  tn_cost = -1/float(len(qs))
  fp_cost = 1/float(ntags)
  fn_cost = float(1)
  return tp_cost*tp+tn_cost*tn+fp_cost*fp+fn_cost*fn
  
def hamming_loss(qs, pc, allc):
  s = 0
  for i, q in enumerate(qs):
     p = pc[i]
     fp = len(p.difference(q.tag_list))
     fn = len(q.tag_list.difference(p))
     s += fp + fn
  return s/float(len(allc)+len(qs))

def multi_accuracy(qs, pc, allc):
  s = 0
  for i, q in enumerate(qs):
    p = pc[i]
    tc = q.tag_list
    s += len(tc.intersection(p))/float(len(tc.union(p)))
  return s/len(qs)

def multi_precision(qs, pc, allc):
  s = 0
  for i, q in enumerate(qs):
    p = pc[i]
    tc = q.tag_list
    s += len(tc.intersection(p))/float(len(p))
  return s/len(qs)

def multi_recall(qs, pc, allc):
  s = 0
  for i, q in enumerate(qs):
    p = pc[i]
    tc = q.tag_list
    s += len(tc.intersection(p))/float(len(tc))
  return s/len(qs)    

def harmonic_mean(x, y):
  return 2/(1/float(x)+1/float(y))

def multi_f1(qs, pc, allc):
  return harmonic_mean(multi_precision(qs, pc, allc), multi_recall(qs, pc, allc))

def eval_multi(questions, tags, cl, folder):
  fname = folder + '/multi_eval.csv'
  f = open(fname, 'w')
  f.write('hamming_loss,precision,recall,f1\n')
  f.write('{},{},{},{}\n'.format(hamming_loss(questions, cl, tags), \
          multi_precision(questions, cl, tags),\
          multi_recall(questions, cl, tags),\
          multi_f1(questions, cl, tags)))
  f.close();


# assumes only one tag / question
def eval_tp1(questions, tags, cl, folder):
  confusion = {}
  for i in range(len(questions)):
    q = iter(questions[i].tag_list).next()
    matrix = confusion.setdefault(q, {})
    c = cl[i]
    for pc in c:
      matrix[pc] = matrix.get(pc, 0) + 1

  fname = folder + '/confusion_matrix.csv'
  f = open(fname, 'w')
  f.write('TrueTag,');
  
  for tag in tags:
    f.write(tag+',')
  f.write('\n')
  for tag in tags:
    f.write(tag+',')
    matrix = confusion.get(tag, {})
    for ptag in tags:
      f.write('{},'.format(matrix.get(ptag, 0)))
    f.write('\n')
  f.close();

  fname = folder + '/eval.csv'
  f = open(fname, 'w')
  f.write('tag,tp,tn,fp,fn,sensitivity,specificity,acc,precision,recall,f1\n')

  for true_tag in tags:
    tp = confusion.get(true_tag, 0).get(true_tag, 0)
    predicted_positive = 0;
    for ttag in tags:
      predicted_positive += confusion.get(ttag, {}).get(true_tag, 0)
    fp = predicted_positive - tp
    positive = 0
    for ptag in tags:
       positive += confusion.get(true_tag, {}).get(ptag, 0)
    fn = positive - tp
    tn = len(questions) - positive - predicted_positive + tp
    sensitivity = tp/float(tp+fn)
    specificity = tn/float(fp+tn)
    precision = tp/float(tp+fp)
    acc=(tp+tn)/float(len(questions))
    f1=(2*tp)/float(2*tp+fp+fn)
    recall=tp/float(tp+fn)
    f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(true_tag,tp,tn,fp,fn,sensitivity,specificity,acc,precision,recall,f1))
  f.close()
  eval_multi(questions, tags, cl, folder)

def leave_one_out(classifier_factory, eval_function, folder_name, questions, all_tags):
  def classify(i, eval_question):
    training_questions = questions[:i]+questions[i+1:]
    classifier = classifier_factory()
    classifier.Train(training_questions);
    return classifier.Classify(eval_question)
  cl = [classify(i, eval_question) for i, eval_question in enumerate(questions)]
  eval_function(questions, all_tags, cl, folder_name)
