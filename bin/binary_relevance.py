import question
import sets

class BinaryRelevanceClassifier:
  def __init__(self, classifier_factory):
      self.classifier_factory = classifier_factory
  def Train(self, questions, alpha=1):
    # extract list of all tags.                                                                                    
    tag_list = set()
    for q in questions:
      tag_list.update(q.tag_list)
    self.classifiers = {}
    for tag in tag_list:
      tqs = [None]*len(questions)
      for i, q in enumerate(questions):
          if (tag in q.tag_list):
              tqs[i] = question.Question([tag], q.word_counts)
          else:
              tqs[i] = question.Question([None], q.word_counts)
      classifier = self.classifier_factory()
      classifier.Train(tqs)
      self.classifiers[tag] = classifier
  def Classify(self, question):
    tags = []
    for tag, classifier in self.classifiers.iteritems():
      tags += classifier.Classify(question)
    return tags
