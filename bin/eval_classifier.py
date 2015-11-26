import sets

# assumes only one tag / question
def eval_tp1(questions, tags, cl, folder):
  confusion = {}
  for i in range(len(questions)):
    q = questions[i].tag_list[0]
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
  f.write('tag,tp,tn,fp,fn,sensitivity,specificity,specificity,acc,precision,recall,f1\n')

  for true_tag in tags:
    f.write(true_tag);
    f.write(',')
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
    f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(tag,tp,tn,fp,fn,sensitivity,specificity,acc,precision,recall,f1))
  f.close()

def leave_one_out(classifier_factory, eval_function, folder_name, questions, all_tags):
  def classify(i, eval_question):
    training_questions = questions[:i]+questions[i+1:]
    classifier = classifier_factory()
    classifier.Train(training_questions);
    return classifier.Classify(eval_question)
  cl = [classify(i, eval_question) for i, eval_question in enumerate(questions)]
  eval_function(questions, all_tags, cl, folder_name)
